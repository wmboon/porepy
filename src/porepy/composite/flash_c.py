"""Module containing implementation of the unified flash using (parallel) compiled
functions created with numba.

The flash system, including a non-parametric interior point method, is assembled and
compiled using :func:`numba.njit`, to enable an efficient solution of the equilibrium
problem.

The compiled functions are such that they can be used to solve multiple flash problems
in parallel.

Parallelization is achieved by applying Newton in parallel for multiple input.
The intended use is for larg compositional flow problems, where an efficient solution
to the local equilibrium problem is required.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_


"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Literal, Optional, Sequence

import numba
import numpy as np

from ._core import NUMBA_CACHE
from .composite_utils import COMPOSITE_LOGGER as logger
from .composite_utils import safe_sum
from .eos_compiler import EoSCompiler, extended_compositional_derivatives
from .mixture import BasicMixture
from .npipm_c import (
    convert_param_dict,
    initialize_npipm_nu,
    linear_solver,
    parallel_solver,
)
from .states import FluidState
from .utils_c import (
    insert_xy,
    normalize_fractions,
    parse_pT,
    parse_sat,
    parse_target_state,
    parse_xyz,
)

__all__ = [
    "Flash_c",
]


_import_start = time.time()


# region Helper methods


@numba.njit("float64[:](float64[:],float64[:,:])", fastmath=True, cache=True)
def _rr_poles(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Parameters:
        y: Phase fractions, assuming the first one belongs to the reference phase.
        K: Matrix of K-values per independent phase (row) per component (column)

    Returns:
        A vector of length ``num_comp`` containing the denominators in the RR-equation
        related to K-values per component.
        Each demoninator is given by :math:`1 + \\sum_{j\\neq r} y_j (K_{ji} - 1)`.

    """
    # tensordot is the fastes option for non-contigous arrays,
    # but currently unsupported by numba TODO
    # return 1 + np.tensordot(K.T - 1, y[1:], axes=1)
    return 1 + (K.T - 1) @ y[1:]  # K-values given for each independent phase


@numba.njit("float64(float64[:],float64[:])", fastmath=True, cache=True)
def _rr_binary_vle_inversion(z: np.ndarray, K: np.ndarray) -> float:
    """Inverts the Rachford-Rice equation for the binary 2-phase case.

    Parameters:
        z: ``shape=(num_comp,)``

            Vector of feed fractions.
        K: ``shape=(num_comp,)``

            Matrix of K-values per per component between vapor and liquid phase.

    Returns:
        The corresponding value of the vapor fraction.

    """
    ncomp = z.shape[0]
    n = np.sum((1 - K) * z)
    d = np.empty(ncomp)
    for i in range(ncomp):
        d[i] = (K[i] - 1) * np.sum(np.delete(K, i) - 1) * z[i]

    return n / np.sum(d)


# NOTE default caching not true because of dependency
@numba.njit("float64(float64[:],float64[:],float64[:,:])", cache=NUMBA_CACHE)
def _rr_potential(z: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Calculates the potential according to [1] for the j-th Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the potential is given by

    .. math::

        F = \\sum\\limits_{i} -(z_i ln(1 - (\\sum\\limits_{j\\neq R}(1 - K_{ij})y_j)))

    References:
        [1] `Okuno and Sepehrnoori (2010) <https://doi.org/10.2118/117752-PA>`_

    Parameters:
        z: ``len=n_c``

            Vector of feed fractions.
        y: ``len=n_p``

            Vector of molar phase fractions.
        K: ``shape=(n_p, n_c)``

            Matrix of K-values per independent phase (row) per component (column).

    Returns:
        The value of the potential based on above formula.

    """
    return np.sum(-z * np.log(np.abs(_rr_poles(y, K))))
    # F = [-np.log(np.abs(_rr_pole(i, y, K))) * z[i] for i in range(len(z))]
    # return np.sum(F)


# endregion
# region General flash equation independent of flash type and EoS


logger.debug(f"(import composite/flash_c.py) Compiling shared flash equations ..\n")


@numba.njit("float64[:](float64[:,:],float64[:],float64[:])", fastmath=True, cache=True)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y,z``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.
        z: ``shape=(num_comp,)``

            Overall fractions per component.

    Returns:
        An array with ``shape=(num_comp - 1,)`` containg the residual of the mass
        conservation equation (left-hand side of above equation) for each component,
        except the first one (in ``z``).

    """
    # tensordot is the fastes option for non-contigous arrays,
    # but currently unsupported by numba TODO
    # return (z - np.tensordot(y, x, axes=1))[1:]
    return (z - np.dot(y, x))[1:]


@numba.njit("float64[:,:](float64[:,:],float64[:])", fastmath=True, cache=True)
def mass_conservation_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_conservation_res`

    The Jacobian is of shape ``(num_comp - 1, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    Note:
        The Jacobian does not depend on the overall fractions ``z``, since they are
        assumed given and constant, hence only relevant for residual.

    """
    nphase, ncomp = x.shape

    # must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero
    jac = np.zeros((ncomp - 1, nphase - 1 + nphase * ncomp), dtype=np.float64)

    for i in range(ncomp - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase
        # hence d mass_i / d y_j = x_ij - x_ir
        jac[i, : nphase - 1] = x[1:, i + 1] - x[0, i + 1]  # i + 1 to skip ref component

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, nphase + i :: nphase] = y  # nphase -1 + i + 1 to skip ref component

    # -1 because of z - z(x,y) = 0
    # and above is dz(x,y) / dyx
    return (-1) * jac


@numba.njit("float64[:](float64[:,:],float64[:])", fastmath=True, cache=True)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.

    Returns:
        An array with ``shape=(num_phase,)`` containg the residual of the complementary
        condition per phase.

    """
    return y * (1 - np.sum(x, axis=1))


@numba.njit("float64[:,:](float64[:,:],float64[:])", fastmath=True, cache=True)
def complementary_conditions_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`complementary_conditions_res`

    The Jacobian is of shape ``(num_phase, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    """
    nphase, ncomp = x.shape
    # must fill with zeros, since matrix sparsely populated.
    d_ccs = np.zeros((nphase, nphase - 1 + nphase * ncomp), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    d_ccs[0, : nphase - 1] = (-1) * unities[0]
    d_ccs[0, nphase - 1 : nphase - 1 + ncomp] = y[0] * (-1)
    for j in range(1, nphase):
        # for the other phases, its slight easier since y_j * (1 - sum_i x_ij)
        d_ccs[j, j - 1] = unities[j]
        d_ccs[j, nphase - 1 + j * ncomp : nphase - 1 + (j + 1) * ncomp] = y[j] * (-1)

    return d_ccs


# endregion


class Flash_c:
    """A class providing efficient unified flash calculations using numba-compiled
    functions.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Flash equations are represented by callable residuals and Jacobians. Various
    flash types are assembled in a modular way by combining required, compiled equations
    into a solvable system.

    Since each system depends on the modelled phases and components, significant
    parts of the equilibrium problem must be compiled on the fly.

    This is a one-time action once the modelling process is completed.

    The supported flash types are than available until destruction.

    Supported flash types/specifications:

    1. ``'p-T'``: state definition in terms of pressure and temperature
    2. ``'p-h'``: state definition in terms of pressure and specific mixture enthalpy
    3. ``'v-h'``: state definition in terms of specific volume and enthalpy of the
       mixture

    Supported mixtures:

    1. non-reactive
    2. only 1 gas and 1 liquid phase
    3. arbitrary many components

    Multiple flash problems can be solved in parallel by passing vectorized state
    definitions.

    Parameters:
        mixture: A mixture model containing modelled components and phases.
        eos_compiler: An EoS compiler instance required to create a
            :class:`~porepy.composite.flash_compiler.FlashCompiler`.

    Raises:
        AssertionError: If not at least 2 components are present.
        AssertionError: If not 2 phases are modelled.

    """

    def __init__(
        self,
        mixture: BasicMixture,
        eos_compiler: EoSCompiler,
    ) -> None:
        nc = mixture.num_components
        np = mixture.num_phases

        assert np == 2, "Supports only 2-phase mixtures."
        assert nc >= 2, "Must have at least two components."

        # data used in initializers
        self._pcrits: list[float] = [comp.p_crit for comp in mixture.components]
        """A list containing critical pressures per component in ``mixture``."""
        self._Tcrits: list[float] = [comp.T_crit for comp in mixture.components]
        """A list containing critical temperatures per component in ``mixture``."""
        self._vcrits: list[float] = [comp.V_crit for comp in mixture.components]
        """A list containing critical volumes per component in ``mixture``."""
        self._omegas: list[float] = [comp.omega for comp in mixture.components]
        """A list containing acentric factors per component in ``mixture``."""
        self._phasetypes: list[int] = [phase.type for phase in mixture.phases]
        """A list containing the phase types per phase in ``mixture``."""

        self.npnc: tuple[int, int] = (np, nc)
        """Number of phases and components present in mixture."""

        self.eos_compiler: EoSCompiler = eos_compiler
        """Assembler and compiler of EoS-related expressions equation.
        passed at instantiation."""

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self.initializers: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the initialization procedure."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u1": 1.0,
            "u2": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta': 0.5`` linear decline in slack variable
        - ``'u1': 1.`` penalty for violating complementarity
        - ``'u2': 1.`` penalty for violating negativitiy of fractions

        Values can be set directly by modifying the values of this dictionary.

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 150,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search:

        - ``'kappa': 0.4``
        - ``'rho_0': 0.99``
        - ``'j_max': 150`` (maximal number of Armijo iterations)

        Values can be set directly by modifying the values of this dictionary.

        """

        self.initialization_parameters: dict[str, float | int] = {
            "N1": 3,
            "N2": 2,
            "N3": 5,
            "eps": 1e-3,
        }
        """Numbers of iterations for initialization procedures and other configurations

        - ``'N1'``: Int, default is 3. Iterations for fractions guess.
        - ``'N2'``: Int, default is 2. Iterations for state constraint (p/T update).
        - ``'N3'``: int, default is 5. Alterations between fractions guess and  p/T
          update.
        - ``'eps'``: Float, default is 1e-3.
          If not None, performs checks of the flash residual.
          If norm of residual reaches this tolerance, initialization is stopped.
          Used only for flashes other than p-T to unecessarily expensive initialization.

        """

        self.tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

        self.last_flash_stats: dict[str, Any] = dict()
        """Contains some information about the last flash procedure called.

        - ``'type'``: String. Type of the flash (p-T, p-h,...)
        - ``'init_time'``: Float. Real time taken to compute initial guess in seconds.
        - ``'minim_time'``: Float. Real time taken to solve the minimization problem in
          seconds.
        - ``'num_flash'``: Int. Number of flash problems solved (if vectorized input)
        - ``'num_max_iter'``: Int. Number of flash procedures which reached the
          prescribed number of iterations.
        - ``'num_failure'``: Int. Number of failed flash procedures
          (failure in evaluation of residual or Jacobian).
        - ``'num_diverged'``: Int. Number of flash procedures which diverged.

        """

    def _parse_and_complete_results(
        self,
        results: np.ndarray,
        state_input: dict[str, np.ndarray],
    ) -> FluidState:
        """Helper function to fill a result state with the results from the flash.

        Modifies and returns the passed result state structur containing flash
        specifications.

        Also, fills up secondary expressions for respective flash type.

        Sequences of quantities associated with phases, components or derivatives are
        stored as 2D arrays for convenience (row-wise per phase/component/derivative).

        """
        nphase, ncomp = self.npnc

        # Parsing phase compositions and molar phsae fractions
        result_state = FluidState(**state_input)
        y: list[np.ndarray] = list()
        x: list[np.ndarray] = list()
        for j in range(nphase):
            # values for molar phase fractions of independent phases
            if j < nphase - 1:
                y.append(results[:, -(1 + nphase * ncomp + nphase - 1) + j])
            # composition of phase j
            x_j = list()
            for i in range(ncomp):
                x_j.append(results[:, -(1 + (nphase - j) * ncomp) + i])
            x.append(np.array(x_j))

        result_state.y = np.vstack([1 - safe_sum(y), np.array(y)])

        # If T is unknown, it is always the last unknown before molar fractions
        if "T" not in state_input:
            result_state.T = results[:, -(1 + ncomp * nphase + nphase - 1 + 1)]

        # If v is a defined value, we fetch pressure and saturations
        if "v" in state_input:
            # If T is additionally unknown to p, p is the second last quantity before
            # molar fractions
            if "T" not in state_input:
                p_pos = 1 + ncomp * nphase + nphase - 1 + 2
            else:
                p_pos = 1 + ncomp * nphase + nphase - 1 + 1

            result_state.p = results[:, -p_pos]

            # saturations are stored before pressure (for independent phases)
            s: list[np.ndarray] = list()
            for j in range(nphase - 1):
                s.append(results[:, -(p_pos + nphase - 1 + j)])
            result_state.sat = [1 - safe_sum(s)] + s

        # Computing states for each phase after filling p, T and x
        result_state.phases = list()
        for j in range(nphase):
            result_state.phases.append(
                self.eos_compiler.compute_phase_state(
                    self._phasetypes[j], result_state.p, result_state.T, x[j]
                )
            )

        # if v not defined, evaluate saturations based on rho and y
        if "v" not in state_input:
            result_state.evaluate_saturations()
        result_state.sat = np.array(result_state.sat)
        # evaluate extensive properties of the fluid mixture
        result_state.evaluate_extensive_state()

        return result_state

    def log_last_stats(self):
        """Prints statistics found in :attr:`last_flash_stats` in the console."""
        logger.warn("--- Last flash stats:\n")
        for k, v in self.last_flash_stats.items():
            logger.warn(f"---\t{k}: {v}\n")
        print("")

    def compile(self, verbosity: int = 1) -> None:
        """Triggers the assembly and compilation of equilibrium equations, including
        the NPIPM approach.

        The order of equations is always as follows:

        1. ``num_comp -1`` mass constraints
        2. ``(num_phase -1) * num_comp`` isofugacity constraints
        3. state constraints (1 for each)
        4. ``num_phase`` complementarity conditions
        5. 1 NPIPM slack equation

        Important:
            This takes a considerable amount of time.
            The compilation is therefore separated from the instantiation of this class.

        Parameters:
            verbosity: ``default=1``

                Enable progress logs. Set to zero to disable.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc
        tol = self.tolerance
        phasetypes = self._phasetypes

        ## dimension of flash systems, excluding NPIPM
        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase
        # p-h flash: additional var T, additional equ enthalpy constraint
        ph_dim = pT_dim + 1
        # v-h flash: additional vars p, s_j j!= ref
        # additional equations volume constraint and density constraints
        vh_dim = ph_dim + 1 + (nphase - 1)

        ## Compilation start
        logger.info(
            f"Starting flash compilation (phases: {nphase}, components: {ncomp}):\n"
        )
        _start = time.time()
        prearg_val_c = self.eos_compiler.funcs.get("prearg_val", None)
        if prearg_val_c is None:
            logger.debug("Compiling residual pre-argument ..\n")
            prearg_val_c = self.eos_compiler.get_prearg_for_values()
        prearg_jac_c = self.eos_compiler.funcs.get("prearg_jac", None)
        if prearg_jac_c is None:
            logger.debug("Compiling Jacobian pre-argument ..\n")
            prearg_jac_c = self.eos_compiler.get_prearg_for_derivatives()
        phi_c = self.eos_compiler.funcs.get("phi", None)
        if phi_c is None:
            logger.debug("Compiling fugacity coefficient function ..\n")
            phi_c = self.eos_compiler.get_fugacity_function()
        d_phi_c = self.eos_compiler.funcs.get("d_phi", None)
        if d_phi_c is None:
            logger.debug("Compiling derivatives of fugacity coefficients ..\n")
            d_phi_c = self.eos_compiler.get_dpTX_fugacity_function()
        h_c = self.eos_compiler.funcs.get("h", None)
        if h_c is None:
            logger.debug("Compiling enthalpy function ..\n")
            h_c = self.eos_compiler.get_enthalpy_function()
        d_h_c = self.eos_compiler.funcs.get("d_h", None)
        if d_h_c is None:
            logger.debug("Compiling derivative of enthalpy function ..\n")
            d_h_c = self.eos_compiler.get_dpTX_enthalpy_function()

        rho_c = self.eos_compiler.funcs.get("rho", None)
        if rho_c is None:
            logger.debug("Compiling density function ..\n")
            rho_c = self.eos_compiler.get_density_function()
        d_rho_c = self.eos_compiler.funcs.get("d_rho", None)
        if d_rho_c is None:
            logger.debug("Compiling derivative of density function ..\n")
            d_rho_c = self.eos_compiler.get_dpTX_density_function()

        logger.debug("Compiling residual of isogucacity constraints ..\n")

        @numba.njit(
            "float64[:](float64[:,:], float64, float64, float64[:,:], float64[:,:])"
        )
        def isofug_constr_c(
            prearg: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(prearg[0], p, T, Xn[0])

            for j in range(1, nphase):
                phi_j = phi_c(prearg[j], p, T, Xn[j])
                # isofugacity constraint between phase j and phase r
                # NOTE fugacities are evaluated with normalized fractions
                isofug[(j - 1) * ncomp : j * ncomp] = X[j] * phi_j - X[0] * phi_r

            return isofug

        logger.debug("Compiling Jacobian of isogucacity constraints ..\n")

        @numba.njit(
            "float64[:,:]"
            + "(float64[:], float64[:], float64, float64, float64[:], float64[:])",
        )
        def d_isofug_block_j(
            prearg_res_j: np.ndarray,
            prearg_jac_j: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """

            phi_j = phi_c(prearg_res_j, p, T, Xn)
            d_phi_j = d_phi_c(prearg_res_j, prearg_jac_j, p, T, Xn)
            # NOTE phi depends on normalized fractions
            # extending derivatives from normalized fractions to extended ones
            for i in range(ncomp):
                d_phi_j[i] = extended_compositional_derivatives(d_phi_j[i], X)

            # product rule: x * dphi
            d_xphi_j = (d_phi_j.T * X).T
            # + phi * dx  (minding the first two columns which contain the p-T derivs)
            d_xphi_j[:, 2:] += np.diag(phi_j)

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64[:,:], float64[:, :],"
            + "float64, float64, float64[:,:], float64[:,:])"
        )
        def d_isofug_constr_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the derivative of the isofugacity constraints

            Formulation is always w.r.t. the reference phase r, assumed to be zero 0.

            Important:
                The derivative is taken w.r.t. to A, B, Z (among others).
                An forward expansion must be done after a call to this function.

            """
            d_iso = np.zeros((ncomp * (nphase - 1), 2 + ncomp * nphase))

            # creating derivative parts involving the reference phase
            d_xphi_r = d_isofug_block_j(prearg_res[0], prearg_jac[0], p, T, X[0], Xn[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(
                    prearg_res[1], prearg_jac[1], p, T, X[j], Xn[j]
                )

                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = -d_xphi_r[:, 2:]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        logger.debug("Compiling residual of enthalpy constraints ..\n")

        @numba.njit(
            "float64(float64[:,:], float64, float64, float64, float64[:], float64[:,:])"
        )
        def h_constr_res_c(
            prearg: np.ndarray,
            p: float,
            h: float,
            T: float,
            y: np.ndarray,
            xn: np.ndarray,
        ) -> float:
            """Helper function to evaluate the normalized residual of the enthalpy
            constraint.

            Note that ``h`` is the target value.

            """

            h_constr_res = h
            for j in range(xn.shape[0]):
                h_constr_res -= y[j] * h_c(prearg[j], p, T, xn[j])

            # for better conditioning, normalize enthalpy constraint if abs(h) > 1
            if np.abs(h) > 1.0:
                h_constr_res /= h

            return h_constr_res

        logger.debug("Compiling Jacobian of enthalpy constraints ..\n")

        @numba.njit(
            "float64[:]"
            + "(float64[:,:],float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:,:],float64[:,:])"
        )
        def h_constr_jac_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            h: float,
            T: float,
            y: np.ndarray,
            x: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Function to assemble the gradient of the enthalpy constraint w.r.t.
            temperature, molar phase fractions and extended phase compositions."""
            # gradient of sum_j y_j h_j(p, T, x_j)  w.r.t. p, T, y, x
            h_constr_jac = np.zeros(2 + nphase - 1 + nphase * ncomp)

            # treatment of reference phase enthalpy
            # enthalpy and its gradient of the reference phase
            h_0 = h_c(prearg_res[0], p, T, xn[0])
            # gradient of h_0 w.r.t to extended fraction
            d_h_0 = extended_compositional_derivatives(
                d_h_c(prearg_res[0], prearg_jac[0], p, T, xn[0]), x[0]
            )
            # contribution to p- and T-derivative of reference phase
            h_constr_jac[0] = y[0] * d_h_0[0]
            h_constr_jac[1] = y[0] * d_h_0[1]
            # y_0 = 1 - y_1 - y_2 ..., contribution is below
            # derivative w.r.t. composition in reference phase
            h_constr_jac[2 + nphase - 1 : 2 + nphase - 1 + ncomp] = y[0] * d_h_0[2:]

            for j in range(1, nphase):
                h_j = h_c(prearg_res[j], p, T, xn[j])
                d_h_j = extended_compositional_derivatives(
                    d_h_c(prearg_res[j], prearg_jac[j], p, T, xn[j]), x[j]
                )
                # contribution to p- and T-derivative of phase j
                h_constr_jac[0] += y[j] * d_h_j[0]
                h_constr_jac[1] += y[j] * d_h_j[1]

                # derivative w.r.t. y_j
                h_constr_jac[1 + j] = h_j - h_0  # because y_0 = 1 - y_1 - y_2 ...

                # derivative w.r.t. composition of phase j
                h_constr_jac[
                    2 + nphase - 1 + j * ncomp : 2 + nphase - 1 + (j + 1) * ncomp
                ] = (y[j] * d_h_j[2:])

            # for better conditioning, if abs(h) > 1, the constraint is 1 - h() / h
            if np.abs(h) > 1:
                h_constr_jac /= h
            return -h_constr_jac

        logger.debug("Compiling residual of volume constraints ..\n")

        @numba.njit(
            "float64[:]"
            + "(float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:],float64[:,:])"
        )
        def v_constr_res_c(
            prearg: np.ndarray,
            v: float,
            p: float,
            T: float,
            sat: np.ndarray,
            y: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Helper function to evaluate the residual of the volume constraint,
            including the phase fraction relations."""

            rho_j = np.array([rho_c(prearg[j], p, T, xn[j]) for j in range(nphase)])
            rho_mix = np.dot(sat, rho_j)

            res = np.empty(nphase, dtype=np.float64)
            # volume constraint
            res[0] = v * rho_mix - 1
            # nphase - 1 phase fraction relations
            res[1:] = (y - sat * rho_j / rho_mix)[1:]

            return res

        logger.debug("Compiling Jacobian of volume constraints ..\n")

        @numba.njit(
            "float64[:,:]"
            + "(float64[:,:],float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:],float64[:,:],float64[:,:])"
        )
        def v_constr_jac_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            v: float,
            p: float,
            T: float,
            sat: np.ndarray,
            y: np.ndarray,
            x: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Helper function to compute the Jacobian of the volume constraint and
            phase fraction relations.
            Returns derivatives w.r.t. sat, p, T, y, and x."""

            rho_j = np.array([rho_c(prearg_res[j], p, T, xn[j]) for j in range(nphase)])
            d_rho_j = np.array(
                [
                    extended_compositional_derivatives(
                        d_rho_c(prearg_res[j], prearg_jac[j], p, T, xn[j]), x[j]
                    )
                    for j in range(nphase)
                ]
            )
            rho_mix = np.dot(sat, rho_j)

            # rho_mix = sum_i s_i * rho_i
            dpT_rho_mix = np.sum([sat[j] * d_rho_j[j, :2] for j in range(nphase)])

            # 1 volume constraint, nphase-1 phase fraction relations, all derivatives
            jac = np.zeros((nphase, 2 * nphase + ncomp * nphase), dtype=np.float64)

            # derivatives of volume constraint w.r.t. independent s_j
            # s_r = 1 - sum_j!=r s_j
            # and v * (sum_i s_i * rho_i) - 1 = 0
            jac[0, : nphase - 1] = rho_j[1:] - rho_j[0]
            # derivatives of volume constraint w.r.t. p and T
            jac[0, nphase - 1 : nphase + 1] = dpT_rho_mix
            # derivatives of v constr w.r.t. x_r
            jac[0, 2 * nphase : 2 * nphase + ncomp] = sat[0] * d_rho_j[0, 2:]

            for j in range(1, nphase):
                # derivatives volume constr w.r.t. x_j for independent phases
                jac[0, 2 * nphase + j * ncomp : 2 * nphase + (j + 1) * ncomp] = (
                    sat[j] * d_rho_j[j, 2:]
                )

                # outer derivative of rho_j * dpTx(1 / rho_mix)
                outer_j = -rho_j[j] / rho_mix**2

                # derivatives of phase fraction relations for each independent phase.
                # y_j - sat_j * rho_j / (sum_i s_i * rho_i)
                # First, derivatives w.r.t. saturations
                # With s_0 = 1 - sum_(i > 0) s_i it holds for k > 0
                # ds_k (s_j rho_j / (sum_i s_i * rho_i)) =
                # delta_kj * rho_j / (sum_i s_i * rho_i)
                # + s_j * (- rho_j / (sum_i s_i * rho_i)^2 * (rho_k - rho_0))
                jac[j, : nphase - 1] = outer_j * (rho_j[1:] - rho_j[0])
                jac[j, j - 1] += rho_j[j] / rho_mix

                # derivatives of phase fraction relations w.r.t. p, T
                # With s_0 = 1 - sum_(i > 0) s_i and rho_mix = sum_i s_i * rho_i
                # dpt (rho_j(p, T) / rho_mix) =
                # dpt(rho_j(p,T)) / rho_mix
                # + rho_j * (-1 / rho_mix^2 * dpt(rho_mix))
                jac[j, nphase - 1 : nphase + 1] = sat[j] * (
                    d_rho_j[j, :2] / rho_mix + outer_j * dpT_rho_mix
                )

                # derivatives of phase fraction relation w.r.t. x_ik
                # for all phases k, and j > 0
                # dx_ik (rho_j(x_ij) / rho_mix) =
                # delta_kj * (dx_ik(rho_j(x_ij)) / rho_mix)
                # + rho_j * (-1 / rho_mix^2 * (
                #   dx_ik(sum_l s_l rho_l(x_il)))
                # )
                for k in range(nphase):
                    jac[j, 2 * nphase + k * ncomp : 2 * nphase + (k + 1) * ncomp] = (
                        sat[j] * outer_j * sat[k] * d_rho_j[k, 2:]
                    )
                    if k == j:
                        jac[
                            j, 2 * nphase + k * ncomp : 2 * nphase + (k + 1) * ncomp
                        ] += (sat[j] * d_rho_j[k, 2:] / rho_mix)

            # volume constraint is scaled with target volume
            jac[0] *= v

            # multiply fraction relations with -1 because y_j (-) s_j rho_j / rho_mix
            jac[1:] *= -1
            # derivatives of phase fraction relations w.r.t. independent y_j
            jac[1:, nphase + 1 : 2 * nphase] = np.eye(nphase - 1)

            return jac

        logger.debug("Compiling p-T flash ..\n")

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian of proper dimension
            jac = np.zeros((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )
            prearg_jac = np.array(
                [prearg_jac_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :
            ] = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)[:, 2:]

            return jac

        logger.debug("Compiling p-h flash ..\n")

        @numba.njit("float64[:](float64[:])")
        def F_ph(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            h, p = parse_target_state(X_gen, (nphase, ncomp))
            _, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(ph_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)

            # complementarity always last for NPIPM to work
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )
            # state constraints always after isofug and befor complementary cond.
            res[-(nphase + 1)] = h_constr_res_c(prearg, p, h, T, y, xn)

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_ph(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            h, p = parse_target_state(X_gen, (nphase, ncomp))
            _, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian of proper dimension
            jac = np.zeros((ph_dim, ph_dim), dtype=np.float64)

            jac[: ncomp - 1, 1:] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:, 1:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )
            prearg_jac = np.array(
                [prearg_jac_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            d_iso = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)

            # derivatives w.r.t. T
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), 0] = d_iso[:, 1]
            # derivatives w.r.t. fractions
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase:] = d_iso[:, 2:]

            d_h_constr = h_constr_jac_c(prearg_res, prearg_jac, p, h, T, y, x, xn)
            jac[-(nphase + 1), 0] = d_h_constr[1]
            jac[-(nphase + 1), nphase:] = d_h_constr[2:]

            return jac

        logger.debug("Compiling v-h flash ..\n")

        @numba.njit("float64[:](float64[:])")
        def F_vh(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            v, h = parse_target_state(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))
            sat = parse_sat(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(vh_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)

            # complementarity always last for NPIPM to work
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )

            # state constraints always after isofug and befor complementary cond.
            # h constraint
            res[ncomp - 1 + ncomp * (nphase - 1)] = h_constr_res_c(
                prearg, p, h, T, y, xn
            )
            # v constraint including closure for saturations (rho y_j = rho_j s_j)
            res[ncomp + ncomp * (nphase - 1) : -nphase] = v_constr_res_c(
                prearg, v, p, T, sat, y, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_vh(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            v, h = parse_target_state(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))
            sat = parse_sat(X_gen, (nphase, ncomp))

            # declare Jacobian of proper dimension
            jac = np.zeros((vh_dim, vh_dim), dtype=np.float64)

            jac[: ncomp - 1, nphase + 1 :] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:, nphase + 1 :] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = np.array(
                [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )
            prearg_jac = np.array(
                [prearg_jac_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
            )

            # isofugacity constraints
            d_iso = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)
            # derivatives w.r.t. p, T
            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 : nphase + 1
            ] = d_iso[:, :2]
            # derivatives w.r.t. x
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), 2 * nphase :] = d_iso[
                :, 2:
            ]

            # enthalpy constraint
            d_h_constr = h_constr_jac_c(prearg_res, prearg_jac, p, h, T, y, x, xn)
            jac[ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :] = d_h_constr

            # volume constraint
            jac[ncomp + ncomp * (nphase - 1) : -nphase] = v_constr_jac_c(
                prearg_res, prearg_jac, v, p, T, sat, y, x, xn
            )

            return jac

        logger.debug("Storing compiled equations ..\n")

        self.residuals.update(
            {
                "p-T": F_pT,
                "p-h": F_ph,
                "v-h": F_vh,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
                "p-h": DF_ph,
                "v-h": DF_vh,
            }
        )

        p_crits = np.array(self._pcrits)
        T_crits = np.array(self._Tcrits)
        v_crits = np.array(self._vcrits)
        omegas = np.array(self._omegas)

        logger.debug("Compiling p-T initialization ..\n")

        @numba.njit("float64[:](float64[:],int32, int32)")
        def guess_fractions(
            X_gen: np.ndarray, N1: int, guess_K_vals: int
        ) -> np.ndarray:
            """Guessing fractions for a single flash configuration"""
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # pseudo-critical quantities
            T_pc = np.sum(z * T_crits)
            p_pc = np.sum(z * p_crits)

            # storage of K-values (first phase assumed reference phase)
            K = np.zeros((nphase - 1, ncomp))
            K_tol = 1e-10  # tolerance to bind K-values away from 0

            if guess_K_vals != 0:
                for j in range(nphase - 1):
                    K[j, :] = (
                        np.exp(5.37 * (1 + omegas) * (1 - T_crits / T)) * p_crits / p
                        + K_tol
                    )
            else:
                xn = normalize_fractions(x)
                prearg = np.array(
                    [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
                )
                # fugacity coefficients reference phase
                phi_r = phi_c(prearg[0], p, T, xn[0])
                for j in range(1, nphase):
                    phi_j = phi_c(prearg[j], p, T, xn[j])
                    K_jr = phi_r / phi_j + K_tol
                    K[j - 1, :] = K_jr

            # starting iterations using Rachford Rice
            for n in range(N1):
                # solving RR for molar phase fractions
                if nphase == 2:
                    # only one independent phase assumed
                    K_ = K[0]
                    if ncomp == 2:
                        y_ = _rr_binary_vle_inversion(z, K_)
                    else:
                        raise NotImplementedError(
                            "Multicomponent RR solution not implemented."
                        )

                    # copy the original value s.t. different corrections
                    # do not interfer with eachother
                    # _y = float(y_)
                    negative = y_ < 0.0
                    exceeds = y_ > 1.0
                    invalid = exceeds | negative

                    # correction of invalid gas phase values
                    if invalid:
                        # assuming gas saturated for correction using RR potential
                        y_test = np.array([0.0, 1.0], dtype=np.float64)
                        rr_pot = _rr_potential(z, y_test, K)
                        # checking if y is feasible
                        # for more information see Equation 10 in
                        # `Okuno et al. (2010) <https://doi.org/10.2118/117752-PA>`_
                        t_i = _rr_poles(y_test, K)
                        cond_1 = t_i - z >= 0.0
                        # tests holds for arbitrary number of phases
                        # reflected by implementation, despite nph == 2
                        cond_2 = K * z - t_i <= 0.0
                        gas_feasible = np.all(cond_1) & np.all(cond_2)

                        if rr_pot > 0.0:
                            y_ = 0.0
                        elif (rr_pot < 0.0) & gas_feasible:
                            y_ = 1.0

                        # clearly liquid
                        if (T < T_pc) & (p > p_pc):
                            y_ = 0.0
                        # clearly gas
                        elif (T > T_pc) & (p < p_pc):
                            y_ = 1.0

                        # Correction based on negative flash
                        # value of y_ must be between innermost poles
                        # K_min = np.min(K_)
                        # K_max = np.max(K_)
                        # y_1 = 1 / (1 - K_max)
                        # y_2 = 1 / (1 - K_min)
                        # if y_1 <= y_2:
                        #     y_feasible = y_1 < _y < y_2
                        # else:
                        #     y_feasible = y_2 < _y < y_1

                        # if y_feasible & negative:
                        #     y_ = 0.0
                        # elif y_feasible & exceeds:
                        #     y_ = 1.0

                        # If all K-values are smaller than 1 and gas fraction is negative,
                        # the liquid phase is clearly saturated
                        # Vice versa, if fraction above 1 and K-values greater than 1.
                        # the gas phase is clearly saturated
                        if negative & np.all(K_ < 1.0):
                            y_ = 0.0
                        elif exceeds & np.all(K_ > 1.0):
                            y_ = 1.0

                        # assert corrections did what they have to do
                        assert (
                            0.0 <= y_ <= 1.0
                        ), "y fraction estimate outside bound [0, 1]."
                    y[1] = y_
                    y[0] = 1.0 - y_
                else:
                    raise NotImplementedError(
                        "Fractions guess for more than 2 phases not implemented."
                    )

                # resolve compositions
                t = _rr_poles(y, K)
                x[0] = z / t  # fraction in reference phase
                x[1:] = K * x[0]  # fraction in indp. phases

                # update K-values if another iteration comes
                if n < N1 - 1:
                    xn = normalize_fractions(x)
                    prearg = np.array(
                        [
                            prearg_val_c(phasetypes[j], p, T, xn[j])
                            for j in range(nphase)
                        ]
                    )
                    # fugacity coefficients reference phase
                    phi_r = phi_c(prearg[0], p, T, xn[0])
                    for j in range(1, nphase):
                        phi_j = phi_c(prearg[j], p, T, xn[j])
                        K_jr = phi_r / phi_j + K_tol
                        K[j - 1, :] = K_jr

            return insert_xy(X_gen, x, y, (nphase, ncomp))

        @numba.njit("float64[:,:](float64[:,:],int32, int32)", parallel=True)
        def pT_initializer(X_gen: np.ndarray, N1: int, guess_K_vals: int) -> np.ndarray:
            """p-T initializer as a parallelized loop over all flash configurations."""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                # for f in range(nf):
                X_gen[f] = guess_fractions(X_gen[f], N1, guess_K_vals)
            return X_gen

        logger.debug("Compiling p-h initialization ..\n")

        @numba.njit("float64[:](float64[:], int32)")
        def update_T_guess(X_gen: np.ndarray, N2: int) -> np.ndarray:
            """Updating T guess by iterating on h-constr w.r.t. T using Newton and some
            corrections"""
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))
            h, _ = parse_target_state(X_gen, (nphase, ncomp))
            xn = normalize_fractions(x)

            for _ in range(N2):
                prearg_res = np.array(
                    [prearg_val_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
                )
                prearg_jac = np.array(
                    [prearg_jac_c(phasetypes[j], p, T, xn[j]) for j in range(nphase)]
                )

                h_constr_res = h_constr_res_c(prearg_res, p, h, T, y, xn)
                if np.abs(h_constr_res) < tol:
                    break
                else:
                    dT_h_constr = h_constr_jac_c(
                        prearg_res, prearg_jac, p, h, T, y, x, xn
                    )[
                        1
                    ]  # T-derivative
                    dT = 0 - h_constr_res / dT_h_constr  # Newton iteration

                    # corrections to unfeasible updates because of decoupling
                    if np.abs(dT) > T:
                        dT = 0.1 * T * np.sign(dT)
                    dT *= 1 - np.abs(dT) / T
                    # TODO this correction is only valid for VLE
                    if h_constr_res > 0 and y[1] > 1e-3:
                        dT *= 0.4
                    T += dT

            # inserting the updated T in the generic thd argument
            # Minding the order z, state_1, state_2, (s),(p),(T), y, x
            X_gen[-(nphase - 1 + nphase * ncomp + 1)] = T
            return X_gen

        @numba.njit(
            "float64[:,:](float64[:,:], int32, int32, int32, float64)",
            parallel=True,
        )
        def ph_initializer(
            X_gen: np.ndarray, N1: int, N2: int, N3: int, eps: float
        ) -> np.ndarray:
            """p-h initializer as a parallelized loop over all configurations"""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                xf = X_gen[f]
                _, _, z = parse_xyz(xf, (nphase, ncomp))
                T_pc = np.sum(z * T_crits)  # pseudo-critical T approximation as start
                xf[-(ncomp * nphase + nphase - 1) - 1] = T_pc
                xf = guess_fractions(xf, N1, 1)

                for _ in range(N3):
                    xf = update_T_guess(xf, N2)
                    xf = guess_fractions(xf, N1, 0)

                    # abort if residual already small enough
                    res = F_ph(xf)
                    if np.linalg.norm(res) <= eps:
                        break

                X_gen[f] = xf
            return X_gen

        logger.debug("Compiling h-v flash initialization ..\n")

        self.initializers.update(
            {
                "p-T": pT_initializer,
                "p-h": ph_initializer,
            }
        )

        _end = time.time()
        logger.info(
            f"Flash compilation completed (elapsed time: {_end - _start}(s)).\n\n"
        )

    def flash(
        self,
        z: Sequence[np.ndarray],
        p: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        initial_state: Optional[FluidState] = None,
        mode: Literal["linear", "parallel"] = "linear",
        verbosity: int = 0,
    ) -> tuple[FluidState, np.ndarray, np.ndarray]:
        """Performes the flash for given feed fractions and state definition.

        Exactly 2 thermodynamic state must be defined in terms of ``p, T, h`` or ``v``
        for an equilibrium problem to be well-defined.

        One state must relate to pressure or volume.
        The other to temperature or energy.

        Supported combination:

        - p-T
        - p-h
        - v-h

        Parameters:
            z: ``len=num_comp - 1``

                A squence of feed fractions per component, except reference component.
            p: Pressure at equilibrium.
            T: Temperature at equilibrium.
            h: Specific enthalpy of the mixture at equilibrium,
            v: Specific volume of the mixture at equilibrium,
            initial_state: ``default=None``

                If not given, an initial guess is computed for the unknowns of the flash
                type.

                If given, it must have at least values for phase fractions and
                compositions.
                Molar phase fraction for reference phase **must not** be provided.

                It must have additionally values for temperature, for
                a state definition where temperature is not known at equilibrium.

                It must have additionally values for pressure and saturations, for
                state definitions where pressure is not known at equilibrium.
                Saturation for reference phase **must not** be provided.
            mode: ``default='linear'``

                Mode of solving the equilibrium problems for multiple state definitions
                given by arrays.

                - ``'linear'``: A classical loop over state defintions (row-wise).
                - ``'parallel'``: A parallelized loop, intended for larger amounts of
                  problems.

            verbosity: ``default=0``

                For logging information about progress. Note that as of now, there is
                no support for logs during solution procedures in the loop since
                compiled code is exectuded.

        Raises:
            ValueError: If an insufficient amount of feed fractions is passed or they
                violate the unity constraint.
            NotImplementedError: If an unsupported combination of insufficient number of
                of thermodynamic states is passed.

        Returns:
            A 3-tuple containing the results, success flags and number of iterations as
            returned by :func:`newton`.
            The results are stored in a fluid state structure.

            Important:
                If the equilibrium state is not defined in terms of pressure or
                temperature, the resulting volume or enthalpy values of the fluid might
                differ slightly from the input values, due to precision and convergence
                criterion.
                Extensive properties are always returned in terms of the computed
                pressure or temperature.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc

        for i, z_ in enumerate(z):
            if np.any(z_ <= 0) or np.any(z_ >= 1):
                raise ValueError(
                    f"Violation of strict bound (0,1) for feed fraction {i} detected."
                )

        z_sum = safe_sum(z)
        if len(z) == ncomp - 1:
            if not np.all(z_sum < 1.0):
                raise ValueError(
                    f"{ncomp - 1} ({ncomp}) feed fractions violate unity constraint."
                )
        elif len(z) == ncomp:
            if not np.all(z_sum == 1.0):
                raise ValueError(
                    f"{ncomp} ({ncomp}) feed fractions violate unity constraint."
                )
            z = z[1:]
        else:
            raise ValueError(f"Expecting at least {ncomp - 1} feed fractions.")

        flash_type: Literal["p-T", "p-h", "v-h"]
        f_dim: int  # Dimension of flash system (unknowns & equations including NPIPM)
        NF: int  # number of vectorized target states
        X0: np.ndarray  # vectorized, generic flash argument
        gen_arg_dim: int  # number of required values for a flash
        init_args: tuple  # Parameters for initialization procedure

        if p is not None and T is not None and (h is None and v is None):
            flash_type = "p-T"
            f_dim = nphase - 1 + nphase * ncomp + 1
            NF = (z_sum + p + T).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + f_dim
            state_1 = p
            state_2 = T
            init_args = (self.initialization_parameters["N1"], 1)
        elif p is not None and h is not None and (T is None and v is None):
            flash_type = "p-h"
            f_dim = nphase - 1 + nphase * ncomp + 1 + 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + 1 + f_dim
            state_1 = h
            state_2 = p
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
                self.initialization_parameters["eps"],
            )
        elif v is not None and h is not None and (T is None and v is None):
            flash_type = "v-h"
            f_dim = nphase - 1 + nphase * ncomp + 2 + nphase - 1 + 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + nphase - 1 + 2 + f_dim
            state_1 = v
            state_2 = h
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
                self.initialization_parameters["eps"],
            )
        else:
            raise NotImplementedError(
                f"Unsupported flash with state definitions {p, T, h, v}"
            )

        logger.info(f"Determined flash type: {flash_type}\n")

        logger.debug("Assembling generic flash arguments ..")
        X0 = np.zeros((NF, gen_arg_dim))
        for i, z_i in enumerate(z):
            X0[:, i] = z_i
        X0[:, ncomp - 1] = state_1
        X0[:, ncomp] = state_2

        if initial_state is None:
            logger.info("Computing initial state ..")
            start = time.time()
            # exclude NPIPM variable (last column) from initialization
            X0[:, :-1] = self.initializers[flash_type](X0[:, :-1], *init_args)
            end = time.time()
            init_time = end - start
            logger.info(f"Initial state computed (elapsed time: {init_time} (s)).\n")
        else:
            init_time = 0.0
            logger.info("Parsing initial state ..")
            # parsing phase compositions and molar fractions
            for j in range(nphase):
                # values for molar phase fractions except for reference phase
                if j < nphase - 1:
                    X0[:, -(1 + nphase * ncomp + nphase - 1 + j)] = initial_state.y[j]
                # composition of phase j
                for i in range(ncomp):
                    X0[:, -(1 + (nphase - j) * ncomp + i)] = initial_state.phases[j].x[
                        i
                    ]

            # If T is unknown, get provided guess for T
            if "T" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 1)] = initial_state.T
            # If v is given, get provided guess for p and saturations
            if "v" in flash_type:
                # If T is additionally unknown to p, p is the second last quantity before
                # molar fractions
                if "T" not in flash_type:
                    p_pos = 1 + ncomp * nphase + nphase - 1 + 2
                else:
                    p_pos = 1 + ncomp * nphase + nphase - 1 + 1
                X0[:, -p_pos] = initial_state.p
                for j in range(nphase - 1):
                    X0[:, -(p_pos + nphase - 1 + j)] = initial_state.sat[j]

            # parsing molar phsae fractions

        logger.info("Computing initial guess for slack variable ..")
        X0 = initialize_npipm_nu(X0, (nphase, ncomp))

        F = self.residuals[flash_type]
        DF = self.jacobians[flash_type]
        solver_params = convert_param_dict(
            {
                "f_dim": f_dim,
                "num_phase": nphase,
                "num_comp": ncomp,
                "tol": self.tolerance,
                "max_iter": self.max_iter,
                "rho": self.armijo_parameters["rho"],
                "kappa": self.armijo_parameters["kappa"],
                "j_max": self.armijo_parameters["j_max"],
                "u1": self.npipm_parameters["u1"],
                "u2": self.npipm_parameters["u2"],
                "eta": self.npipm_parameters["eta"],
            }
        )

        logger.info("Solving ..\n")
        start = time.time()
        if mode == "linear":
            results, success, num_iter = linear_solver(X0, F, DF, solver_params)
        elif mode == "parallel":
            results, success, num_iter = parallel_solver(X0, F, DF, solver_params)
        else:
            raise ValueError(f"Unknown mode of compuation {mode}")
        end = time.time()
        minim_time = end - start
        logger.info(f"{flash_type} flash done (elapsed time: {minim_time} (s)).\n\n")

        self.last_flash_stats = {
            "type": flash_type,
            "init_time": init_time,
            "minim_time": minim_time,
            "num_flash": NF,
            "num_max_iter": int(np.sum(success == 1)),
            "num_failure": int(np.sum(success == 2) + np.sum(success == 3)),
            "num_diverged": int(np.sum(success == 4)),
        }
        if verbosity >= 2:
            self.log_last_stats()

        z_ = X0[:, : ncomp - 1].T
        state_input = {
            "z": np.vstack([1 - np.sum(z_, axis=0), z_]),
            flash_type[0]: X0[:, ncomp - 1],
            flash_type[2]: X0[:, ncomp],
        }
        return (
            self._parse_and_complete_results(results, state_input),
            success,
            num_iter,
        )


_import_end = time.time()

logger.debug(
    "(import composite/flash_c.py)"
    + f" Done (elapsed time: {_import_end - _import_start} (s)).\n\n"
)

del _import_start, _import_end
