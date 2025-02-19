r"""
This module contains the setup for a 3d verification test for the coupled
thermo-poromechanical problem. The problem is defined on a unit square domain, and
consists of a fluid flow equation, a mechanical equation, and an energy equation.

The problem definition contains a heterogeneity in the permeability and Lamé parameters.
To define the manufactured solution, we introduce the auxiliary function

.. math::
    f(x, y, z, t) = t * x * (1 - x) * (x - 1 / 2) * sin(2 * pi * y) * sin(2 * pi * z)

Define the characteristic function $$\\chi$$, which is 1 if $$x > 0.5$$, $$y > 0.5$$, $$z
> 0.5$$, and 0 otherwise. Also, define a heterogeneity factor $$\kappa$$. The exact
solutions for the primary variables pressure, displacement, and temperature are then
defined as

.. math::
    p(x, y, z, t) = f / ((1 - \\chi) + \\chi * \\kappa),

    u_x(x, y, z, t) = p,

    u_y(x, y, z, t) = p,

    T(x, y, t) = f

The permeability and the Lamé parameters are also made heterogeneous (though with an
inverse scaling, so that the derived fluxes and stresses are continuous across the
interface between the regions with different material parameters). The temperature is
not scaled with the heterogeneity, since this would be difficult to make work with the
fluid conductivity not being heterogeneous. Also note that all the primary variables are
constructed to be zero at the interface, so that they are continuous across the
interface.

In addition to the heterogeneity, a notable feature of the problem is the tensorial Biot
coefficient and thermal expansion tensor. These are also spatially heterogeneous.

"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sym

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from tests.functional.setups.manu_thermoporomech_nofrac_2d import (
    ManuThermoPoroMechDataSaving,
    SourceTerms,
    ManuThermoPoroMechSaveData,
)


class ManuThermoPoroMechExactSolution3d:
    """Class containing the exact manufactured solution for the verification setup.

    The exact solutions for the primary variables pressure, displacement, temperature,
    are defined below, as well as the exact solutions for the secondary variables Darcy
    flux, energy flux and thermoporoelastic force.

    A heterogeneity is introduced in the permeability and Lamé parameters, so that these
    take different values in the region x > 0.5 and y > 0.5. The primary variables
    pressure and displacement are scaled with the heterogeneity, but in a reciprocal
    way, so that the derived fluxes and stresses are continuous across the interface
    between the regions with different material parameters. Moreover, the primary
    variables are constructed to be zero at the interface, so that they are continuous
    across the interface. This is necessary to ensure that the exact solution can be
    used to verify convergence of the numerical solution.

    The temperature variable was not scaled with the heterogeneity, since this would be
    difficult to make work with the fluid conductivity (thus effective heterogeneity)
    not being heterogeneous.

    The problem is defined with tensorial Biot's coefficient and thermal expansion
    tensor. These are also spatially heterogeneous.

    """

    def __init__(self, setup: pp.PorePyModel):
        """Constructor of the class."""

        # Heterogeneity factor.
        heterogeneity: float = setup.params.get("heterogeneity")

        # Physical parameters, fetched from the material constants in the setup object.

        # The parameters for mechanical stiffness and permeability can be made
        # heterogeneous, by means of the parameter 'hetereogeneity'. The values fetched
        # from the material constants 'solid' and 'fluid' are used as base values, and
        # expanded by the heterogeneity factor below
        #
        # Lamé parameters
        lame_lmbda_base = setup.solid.lame_lambda
        lame_mu_base = setup.solid.shear_modulus
        # Permeability
        permeability_base = setup.solid.permeability

        # Biot coefficient. Will be used to define the Biot tensor below.
        alpha = setup.solid.biot_coefficient
        # Reference density and compressibility for fluid.
        reference_fluid_density = setup.fluid.reference_component.density
        fluid_compressibility = setup.fluid.reference_component.compressibility
        # Density of the solid.
        solid_density = setup.solid.density

        # Reference porosity
        phi_0 = setup.solid.porosity
        # Specific heat capacity of the fluid
        fluid_specific_heat = setup.fluid.reference_component.specific_heat_capacity
        # Specific heat capacity of the solid
        solid_specific_heat = setup.solid.specific_heat_capacity
        # Reference pressure and temperature
        p_0 = setup.reference_variable_values.pressure
        T_0 = setup.reference_variable_values.temperature

        # Thermal expansion coefficients
        fluid_thermal_expansion = setup.fluid.reference_component.thermal_expansion
        solid_thermal_expansion = setup.solid.thermal_expansion

        # Conductivity for the fluid and solid
        fluid_conductivity = setup.fluid.reference_component.thermal_conductivity
        solid_conductivity = setup.solid.thermal_conductivity

        # Fluid viscosity
        mu_f = setup.fluid.reference_component.viscosity

        ## Done with fetching constants. Now, introduce heterogeneities and define
        # the exact solutions for the primary variables.

        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")
        pi = sym.pi

        # Characteristic function: 1 if x > 0.5 and y > 0.5, 0 otherwise
        char_func = sym.Piecewise((1, ((x > 0.5) & (y > 0.5) & (z > 0.5))), (0, True))

        def make_heterogeneous(v, invert: bool):
            # Helper function to include the heterogeneity into a function.
            if invert:
                return v / ((1 - char_func) + char_func * heterogeneity)
            else:
                return v * ((1 - char_func) + char_func * heterogeneity)

        # Base for the exact pressure and displacement solutions. Note the compatibility
        # condition for the heterogeneous primary variables and material parameters: The
        # scaling with the heterogeneity is reciprocal in the primary variables and
        # material parameters, so that the derived fluxes and stresses are continuous
        # across the interface between the regions with different material parameters.
        # Moreover, the primary variables are constructed to be zero at the interface,
        # so that they are continuous across the interface.
        #
        # Pressure
        p_base = (
            t * x * (1 - x) * (x - 1 / 2) * sym.sin(2 * pi * y) * sym.sin(2 * pi * z)
        )
        p = make_heterogeneous(p_base, True)
        # Displacement
        u_base = [p_base, p_base, p_base]
        u = [
            make_heterogeneous(u_base[0], True),
            make_heterogeneous(u_base[1], True),
            make_heterogeneous(u_base[2], True),
        ]

        # Temperature is not scaled with the heterogeneity, since this would be
        # difficult to make work with the fluid conductivity not being heterogeneous.
        T = p_base

        # Heterogeneous material parameters
        permeability = make_heterogeneous(permeability_base, False)
        lame_lmbda = make_heterogeneous(lame_lmbda_base, False)
        lame_mu = make_heterogeneous(lame_mu_base, False)
        #  Solid Bulk modulus (heterogeneous)
        K_d = lame_lmbda + (2 / 3) * lame_mu

        # Stress tensors for the fluid and thermal stress (the 'Biot tensor' and its
        # thermal counterpart). These should be symmetric and positive definite. It was
        # easier to define these directly than to use the function make_heterogeneous.
        fluid_stress_tensor = [
            [
                alpha * (10 * char_func + 1 * (1 - char_func)),
                alpha * 0.2 * (1 - char_func),
                alpha * 0.15 * (1 - char_func),
            ],
            [
                alpha * 0.2 * (1 - char_func),
                alpha * (1.0 / 10 * char_func + 1 * (1 - char_func)),
                alpha * 0.2 * (1 - char_func),
            ],
            [
                alpha * 0.15 * (1 - char_func),
                alpha * 0.2 * (1 - char_func),
                alpha * (1.0 / 10 * char_func + 1 * (1 - char_func)),
            ],
        ]
        thermal_stress_tensor = [
            [
                solid_thermal_expansion * (10 * char_func + 1 * (1 - char_func)),
                solid_thermal_expansion * 0.1 * (1 - char_func),
                solid_thermal_expansion * 0.2 * (1 - char_func),
            ],
            [
                solid_thermal_expansion * 0.1 * (1 - char_func),
                0.1
                * solid_thermal_expansion
                * (1.0 / 10 * char_func + 1 * (1 - char_func)),
                solid_thermal_expansion * 0.1 * (1 - char_func),
            ],
            [
                solid_thermal_expansion * 0.2 * (1 - char_func),
                solid_thermal_expansion * 0.1 * (1 - char_func),
                0.7
                * solid_thermal_expansion
                * (1.0 / 10 * char_func + 1 * (1 - char_func)),
            ],
        ]

        # Define secondary quantities, with an aim of defining the exact source terms
        # for the balance equations (found by plugging the imposed exact solutions into
        # the equations).
        #
        # NOTE: The below expressions are mainly taken from Section 3 of
        #
        # Stefansson et al (2024): Flexible and rigorous numerical modelling of
        # multiphysics processes in fractured porous media using PorePy.
        #
        # Reference to specific equations, e.g., (13) are given in the comments below.
        # Some expressions, mainly those relating to the tensorial Biot coefficient, are
        # taken from Coussy (2004), and are referenced as such.

        # Exact fluid density - Eq (13)
        fluid_density = reference_fluid_density * sym.exp(
            fluid_compressibility * (p - p_0) - fluid_thermal_expansion * (T - T_0)
        )

        # Exact gradient of the displacement
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y), sym.diff(u[0], z)],
            [sym.diff(u[1], x), sym.diff(u[1], y), sym.diff(u[1], z)],
            [sym.diff(u[2], x), sym.diff(u[2], y), sym.diff(u[2], z)],
        ]
        # The alteration of porosity due to mechanical deformation (note: This should be
        # computed from the displacement gradients, and not from the symmetrized
        # version). This is taken from Coussy, Eq 4.19b.
        alpha_div_u = (
            fluid_stress_tensor[0][0] * grad_u[0][0]
            + fluid_stress_tensor[0][1] * grad_u[0][1]
            + fluid_stress_tensor[0][2] * grad_u[0][2]
            + fluid_stress_tensor[1][0] * grad_u[1][0]
            + fluid_stress_tensor[1][1] * grad_u[1][1]
            + fluid_stress_tensor[1][2] * grad_u[1][2]
            + fluid_stress_tensor[2][0] * grad_u[2][0]
            + fluid_stress_tensor[2][1] * grad_u[2][1]
            + fluid_stress_tensor[2][2] * grad_u[2][2]
        )
        # Porosity - (26)
        phi = (
            phi_0
            + ((alpha - phi_0) * (1 - alpha) / K_d) * (p - p_0)
            + alpha_div_u
            - (alpha - phi_0) * solid_thermal_expansion * (T - T_0)
        )

        ## The mass flux and source term for the mass balance

        # Exact darcy flux (11) (no gravity)
        q_darcy = [
            -1 * (permeability / mu_f) * sym.diff(p, x),
            -1 * (permeability / mu_f) * sym.diff(p, y),
            -1 * (permeability / mu_f) * sym.diff(p, z),
        ]

        # Exact mass flux (1)
        mf = [
            fluid_density * q_darcy[0],
            fluid_density * q_darcy[1],
            fluid_density * q_darcy[2],
        ]
        # Exact divergence of the mass flux
        div_mf = sym.diff(mf[0], x) + sym.diff(mf[1], y) + sym.diff(mf[2], z)
        # Exact mass accumulation (1)
        accum_flow = sym.diff(phi * fluid_density, t)
        # Exact mass source (1)
        source_flow = accum_flow + div_mf

        ## The energy flux and source term for the energy balance

        # Exact enthalpy (15)
        fluid_enthalpy = fluid_specific_heat * (T - T_0)
        # Advective heat flux (19)
        q_adv = [
            fluid_density * fluid_enthalpy * q_darcy[0],
            fluid_density * fluid_enthalpy * q_darcy[1],
            fluid_density * fluid_enthalpy * q_darcy[2],
        ]

        # Weighted average of the solid and fluid conductivities (3, properly
        # interpreted)
        conductivity = solid_conductivity * (1 - phi) + fluid_conductivity * phi

        # Exact Fourier flux - (18)
        q_fourier = [
            -1 * conductivity * sym.diff(T, x),
            -1 * conductivity * sym.diff(T, y),
            -1 * conductivity * sym.diff(T, z),
        ]

        # Total flux of energy:
        q_energy_total = [
            q_adv[0] + q_fourier[0],
            q_adv[1] + q_fourier[1],
            q_adv[2] + q_fourier[2],
        ]

        # Exact divergence of the energy flux
        div_ef = (
            sym.diff(q_energy_total[0], x)
            + sym.diff(q_energy_total[1], y)
            + sym.diff(q_energy_total[2], z)
        )

        # Specific internal energy of fluid (16)
        specific_internal_energy_fluid = fluid_enthalpy - p / fluid_density
        # Specific internal energy of solid (17)
        specific_internal_energy_rock = solid_specific_heat * (T - T_0)

        # Exact energy accumulation (2-3)
        accum_energy = sym.diff(
            (
                phi * fluid_density * specific_internal_energy_fluid
                + (1 - phi) * solid_density * specific_internal_energy_rock
            ),
            t,
        )

        # Exact energy source (2)
        source_energy = accum_energy + div_ef

        ## Momentum balance

        # Exact transpose of the gradient of the displacement
        trans_grad_u = [
            [grad_u[0][0], grad_u[1][0], grad_u[2][0]],
            [grad_u[0][1], grad_u[1][1], grad_u[2][1]],
            [grad_u[0][2], grad_u[1][2], grad_u[2][2]],
        ]

        # Exact (symmetrized) strain
        epsilon = [
            [
                0.5 * (grad_u[0][0] + trans_grad_u[0][0]),
                0.5 * (grad_u[0][1] + trans_grad_u[0][1]),
                0.5 * (grad_u[0][2] + trans_grad_u[0][2]),
            ],
            [
                0.5 * (grad_u[1][0] + trans_grad_u[1][0]),
                0.5 * (grad_u[1][1] + trans_grad_u[1][1]),
                0.5 * (grad_u[1][2] + trans_grad_u[1][2]),
            ],
            [
                0.5 * (grad_u[2][0] + trans_grad_u[2][0]),
                0.5 * (grad_u[2][1] + trans_grad_u[2][1]),
                0.5 * (grad_u[2][2] + trans_grad_u[2][2]),
            ],
        ]

        # Exact trace (in the linear algebra sense) of the strain
        tr_epsilon = epsilon[0][0] + epsilon[1][1] + epsilon[2][2]

        # Exact elastic stress
        sigma_elas = [
            [
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[0][0],
                2 * lame_mu * epsilon[0][1],
                2 * lame_mu * epsilon[0][2],
            ],
            [
                2 * lame_mu * epsilon[1][0],
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[1][1],
                2 * lame_mu * epsilon[1][2],
            ],
            [
                2 * lame_mu * epsilon[2][0],
                2 * lame_mu * epsilon[2][1],
                lame_lmbda * tr_epsilon + 2 * lame_mu * epsilon[2][2],
            ],
        ]

        # Exact thermoporoelastic stress (24)
        sigma_total = [
            [
                sigma_elas[0][0]
                - fluid_stress_tensor[0][0] * p
                - thermal_stress_tensor[0][0] * T,
                sigma_elas[0][1]
                - fluid_stress_tensor[0][1] * p
                - thermal_stress_tensor[0][1] * T,
                sigma_elas[0][2]
                - fluid_stress_tensor[0][2] * p
                - thermal_stress_tensor[0][2] * T,
            ],
            [
                sigma_elas[1][0]
                - fluid_stress_tensor[1][0] * p
                - thermal_stress_tensor[1][0] * T,
                sigma_elas[1][1]
                - fluid_stress_tensor[1][1] * p
                - thermal_stress_tensor[1][1] * T,
                sigma_elas[1][2]
                - fluid_stress_tensor[1][2] * p
                - thermal_stress_tensor[1][2] * T,
            ],
            [
                sigma_elas[2][0]
                - fluid_stress_tensor[2][0] * p
                - thermal_stress_tensor[2][0] * T,
                sigma_elas[2][1]
                - fluid_stress_tensor[2][1] * p
                - thermal_stress_tensor[2][1] * T,
                sigma_elas[2][2]
                - fluid_stress_tensor[2][2] * p
                - thermal_stress_tensor[2][2] * T,
            ],
        ]

        # Mechanics source term
        source_mech = [
            sym.diff(sigma_total[0][0], x)
            + sym.diff(sigma_total[0][1], y)
            + sym.diff(sigma_total[0][2], z),
            sym.diff(sigma_total[1][0], x)
            + sym.diff(sigma_total[1][1], y)
            + sym.diff(sigma_total[1][2], z),
            sym.diff(sigma_total[2][0], x)
            + sym.diff(sigma_total[2][1], y)
            + sym.diff(sigma_total[2][2], z),
        ]

        ## Public attributes
        # Primary variables
        self.p = p  # pressure
        self.u = u  # displacement
        self.T = T  # temperature
        # Secondary variables
        self.sigma_total = sigma_total  # poroelastic (total) stress
        self.q_darcy = q_darcy  # Darcy flux
        self.q_energy = q_energy_total  # Energy flux

        # Source terms
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.source_flow = source_flow  # Source term entering the mass balance
        self.source_energy = source_energy  # Source term entering the energy balance

        # Heterogeneous material parameters. Make these available, so that a model can
        # be populated with these parameters.
        self.biot_tensor = fluid_stress_tensor  # Biot tensor
        self.thermal_stress_tensor = thermal_stress_tensor  # Thermal expansion tensor
        self.k = permeability  # permeability
        self.lame_lmbda = lame_lmbda  # Lamé parameter
        self.lame_mu = lame_mu  # Lamé parameter

    # -----> Primary and secondary variables
    def pressure(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact pressure at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at the
            cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        p_fun: Callable = sym.lambdify((x, y, z, t), self.p, "numpy")

        # Cell-centered pressures
        p_cc: np.ndarray = p_fun(cc[0], cc[1], cc[2], time)

        return p_cc

    def displacement(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact displacement at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_cells, )`` containing the exact displacements
            at the cell centers for the given ``time``.

        Notes:
            The returned displacement is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        u_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.u[0], "numpy"),
            sym.lambdify((x, y, z, t), self.u[1], "numpy"),
            sym.lambdify((x, y, z, t), self.u[2], "numpy"),
        ]

        # Cell-centered displacements
        u_cc: list[np.ndarray] = [
            u_fun[0](cc[0], cc[1], cc[2], time),
            u_fun[1](cc[0], cc[1], cc[2], time),
            u_fun[2](cc[0], cc[1], cc[2], time),
        ]

        # Flatten array
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        return u_flat

    def temperature(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact temperature at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact temperature at the
            cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        p_fun: Callable = sym.lambdify((x, y, z, t), self.T, "numpy")

        # Cell-centered temperatures
        p_cc: np.ndarray = p_fun(cc[0], cc[1], cc[2], time)

        return p_cc

    def darcy_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact Darcy flux at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy fluxes at
            the face centers for the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd.face_normals``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.q_darcy[0], "numpy"),
            sym.lambdify((x, y, z, t), self.q_darcy[1], "numpy"),
            sym.lambdify((x, y, z, t), self.q_darcy[2], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], fc[2], time) * fn[0]
            + q_fun[1](fc[0], fc[1], fc[2], time) * fn[1]
            + q_fun[2](fc[0], fc[1], fc[2], time) * fn[2]
        )

        return q_fc

    def energy_flux(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact energy flux at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd.num_faces, )`` containing the exact Darcy fluxes at
            the face centers for the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd.face_normals``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.q_energy[0], "numpy"),
            sym.lambdify((x, y, z, t), self.q_energy[1], "numpy"),
            sym.lambdify((x, y, z, t), self.q_energy[2], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc: np.ndarray = (
            q_fun[0](fc[0], fc[1], fc[2], time) * fn[0]
            + q_fun[1](fc[0], fc[1], fc[2], time) * fn[1]
            + q_fun[2](fc[0], fc[1], fc[2], time) * fn[2]
        )

        return q_fc

    def thermoporoelastic_force(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact thermoporoelastic force at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact poroealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned thermoporoelastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun: list[list[Callable]] = [
            [
                sym.lambdify((x, y, z, t), self.sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[2][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][2], "numpy"),
            ],
        ]

        # Face-centered thermoporoelastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y + sigma_xz * n_z) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[0][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_yx * n_x + sigma_yy * n_y + sigma_yz * n_z) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[1][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_zx * n_x + sigma_zy * n_y + sigma_zz * n_z) * face_area
            sigma_total_fun[2][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[2][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[2][2](fc[0], fc[1], fc[2], time) * fn[2],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    # -----> Sources
    def mechanics_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the momentum balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the momentum balance equation with ``shape=(
            3 * sd.num_cells, )``.

        Notes:
            The returned array is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_mech_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.source_mech[0], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[1], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[2], "numpy"),
        ]

        # Evaluate and integrate source
        source_mech: list[np.ndarray] = [
            source_mech_fun[0](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[1](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[2](cc[0], cc[1], cc[2], time) * vol,
        ]

        # Flatten array
        source_mech_flat: np.ndarray = np.asarray(source_mech).ravel("F")

        # Invert sign according to sign convention.
        return -source_mech_flat

    def flow_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the fluid mass balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the fluid mass balance equation with ``shape=(
            sd.num_cells, )``.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_flow_fun: Callable = sym.lambdify(
            (x, y, z, t), self.source_flow, "numpy"
        )

        # Evaluate and integrate source
        source_flow: np.ndarray = source_flow_fun(cc[0], cc[1], cc[2], time) * vol

        return source_flow

    def energy_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the energy balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the energy balance equation with ``shape=(
            sd.num_cells, )``.

        """
        x, y, z, t = sym.symbols("x y z t")

        cc = sd.cell_centers
        vol = sd.cell_volumes

        source_energy_fun = sym.lambdify((x, y, z, t), self.source_energy, "numpy")

        source_energy = source_energy_fun(cc[0], cc[1], cc[2], time) * vol
        return source_energy


# -----> Geometry
class UnitCubeGrid(pp.ModelGeometry):
    """Class for setting up the geometry of the unit cube domain.

    The domain may be assigned different material parameters in the region x > 0.5, y >
    0.5, z > 0.5 To ensure the region with different material parameters is the same in
    all refinement levels, we want to have the lines x=0.5, y=0.5, z=0.5 as grid lines.
    This is achieved by different means: For a Cartesian grid, we simply have to make
    sure the number of cells in all directions is even (this is done by the default
    meshing parameters provided in self.meshing_arguments(), but will have to be taken
    care of by the user if the default parameters is overridden). For a simplex grid,
    the lines are defined as fractures in self.set_fractures(), and marked as
    constraints in self.meshing_kwargs().

    Furthermore, if the grid nodes are perturbed, the perturbation should not be applied
    to the nodes on the boundary of the domain, nor to the nodes at x=0.5, y=0.5, z=0.5.
    The latter is needed to ensure the region with the different material parameters is
    the same in all realizations of the perturbed grid. This is achieved in
    self.set_geometry().

    """

    params: dict
    """Simulation model parameters."""

    def set_geometry(self) -> None:
        super().set_geometry()

        sd = self.mdg.subdomains()[0]
        x, y, z = sd.nodes[0], sd.nodes[1], sd.nodes[2]
        h = np.min(sd.cell_diameters())

        pert_rate = self.params.get("perturbation", 0.0)

        # Nodes to perturb: Not on the boundary, and not at x=0.5, y=0.5, z=0.5.
        pert_nodes = np.logical_not(
            np.logical_or.reduce(
                (np.isin(x, [0, 0.5, 1]), np.isin(y, [0, 0.5, 1], np.isin(z, [0, 1])))
            )
        )
        # Set the random seed
        np.random.seed(42)
        # Perturb the nodes
        x[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
        y[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)
        z[pert_nodes] += pert_rate * h * (np.random.rand(pert_nodes.sum()) - 0.5)

        sd.compute_geometry()

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.25}
        return self.params.get("meshing_arguments", default_mesh_arguments)

    def set_fractures(self) -> None:
        """The geometry contains no fractures per se, but we set fractures for simplex
        grids to conform to material heterogeneities. See class documentation for
        details.
        """

        if self.params["grid_type"] == "simplex":
            self._fractures = pp.fracture_sets.orthogonal_fractures_3d(size=1)
        else:
            # No need to do anything for Cartesian grids.
            self._fractures = []

    def meshing_kwargs(self) -> dict:
        """Set meshing arguments."""
        if self.params["grid_type"] == "simplex":
            # Mark the fractures added as constraints (not to be represented as
            # lower-dimensional objects).
            return {"constraints": [0, 1, 2]}
        else:
            return {}


# -----> Solution strategy
class ManuThermoPoroMechSolutionStrategy3d(
    pp.thermoporomechanics.SolutionStrategyThermoporomechanics
):
    """Solution strategy for the verification setup."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.exact_sol: ManuThermoPoroMechExactSolution3d
        """Exact solution object."""

        self.stress_variable: str = "thermoporoelastic_force"
        """Keyword to access the thermoporoelastic force."""
        
        self.results: list[ManuThermoPoroMechSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""        

    def set_materials(self):
        """Set material parameters."""
        super().set_materials()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ManuThermoPoroMechExactSolution3d(self)

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        super().before_nonlinear_loop()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, time_step_index=0
        )

        # Flow source
        flow_source = self.exact_sol.flow_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_flow", values=flow_source, data=data, time_step_index=0
        )
        # Energy source
        energy_source = self.exact_sol.energy_source(sd=sd, time=t)
        pp.set_solution_values(
            name="source_energy", values=energy_source, data=data, time_step_index=0
        )

    def _is_nonlinear_problem(self) -> bool:
        """The problem is non-linear due to the coupling between fluid flux and
        advective energy transport.
        """
        return True

    def bulk_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The bulk modulus will be spatially varying if the Lame parameters are so."""
        x, y, z = sym.symbols("x y z")

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            xc = sd.cell_centers
            lame_lmbda = sym.lambdify((x, y, z), self.exact_sol.lame_lmbda, "numpy")(
                xc[0], xc[1], xc[2]
            )
            lame_mu = sym.lambdify((x, y, z), self.exact_sol.lame_mu, "numpy")(
                xc[0], xc[1], xc[2]
            )

            bulk = lame_lmbda + (2 / 3) * lame_mu
            # Special case: If the bulk modulus is constant (no spatial variation), the
            # lambdify function returns a float or int. In this case, we need to
            # broadcast the value to all cells.
            if isinstance(bulk, (float, int)):
                bulk = bulk * np.ones(sd.num_cells)

            return pp.wrap_as_dense_ad_array(bulk, name="bulk_modulus")

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem.

        The parent class' definitions of permeability, stiffness parameters, and the Biot
        and thermal stress tensors are owerwritten.
        """
        super().set_discretization_parameters()

        x, y, z = sym.symbols("x y z")

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            xc = sd.cell_centers

            # Set permeability
            k_xx = sym.lambdify((x, y, z), self.exact_sol.k, "numpy")(
                xc[0], xc[1], xc[2]
            )
            if isinstance(k_xx, (float, int)):
                k_xx = k_xx * np.ones(sd.num_cells)

            perm = pp.SecondOrderTensor(kxx=k_xx)
            data[pp.PARAMETERS][self.darcy_keyword]["second_order_tensor"] = perm

            # Set stiffness matrix
            lame_lmbda = sym.lambdify((x, y, z), self.exact_sol.lame_lmbda, "numpy")(
                xc[0], xc[1], xc[2]
            )
            lame_mu = sym.lambdify((x, y, z), self.exact_sol.lame_mu, "numpy")(
                xc[0], xc[1], xc[2]
            )
            if isinstance(lame_lmbda, (float, int)):
                lame_lmbda = lame_lmbda * np.ones(sd.num_cells)
            if isinstance(lame_mu, (float, int)):
                lame_mu = lame_mu * np.ones(sd.num_cells)
            stiffness = pp.FourthOrderTensor(lmbda=lame_lmbda, mu=lame_mu)
            data[pp.PARAMETERS][self.stress_keyword]["fourth_order_tensor"] = stiffness

            # Set the Biot tensor
            a_xx = sym.lambdify((x, y, z), self.exact_sol.biot_tensor[0][0], "numpy")(
                xc[0], xc[1], xc[2]
            )
            a_xy = sym.lambdify((x, y, z), self.exact_sol.biot_tensor[0][1], "numpy")(
                xc[0], xc[1], xc[2]
            )
            a_yy = sym.lambdify((x, y, z), self.exact_sol.biot_tensor[1][1], "numpy")(
                xc[0], xc[1], xc[2]
            )
            # Special case: If the Biot tensor is constant (no spatial variation), the
            # lambdify function returns a float or int. In this case, we need to
            # broadcast the value to all cells.
            if isinstance(a_xx, (float, int)):
                a_xx = a_xx * np.ones(sd.num_cells)
            if isinstance(a_yy, (float, int)):
                a_yy = a_yy * np.ones(sd.num_cells)
            if isinstance(a_xy, (float, int)):
                a_xy = a_xy * np.ones(sd.num_cells)
            biot_alpha = pp.SecondOrderTensor(kxx=a_xx, kyy=a_yy, kxy=a_xy)

            # Set the thermal stress tensor
            b_xx = sym.lambdify(
                (x, y, z), self.exact_sol.thermal_stress_tensor[0][0], "numpy"
            )(xc[0], xc[1], xc[2])
            b_xy = sym.lambdify(
                (x, y, z), self.exact_sol.thermal_stress_tensor[0][1], "numpy"
            )(xc[0], xc[1], xc[2])
            b_yy = sym.lambdify(
                (x, y, z), self.exact_sol.thermal_stress_tensor[1][1], "numpy"
            )(xc[0], xc[1], xc[2])
            # Special case: If the thermal stress tensor is constant (no spatial
            # variation), the lambdify function returns a float or int. In this case, we
            # need to broadcast the value to all cells.
            if isinstance(b_xx, (float, int)):
                b_xx = b_xx * np.ones(sd.num_cells)
            if isinstance(b_yy, (float, int)):
                b_yy = b_yy * np.ones(sd.num_cells)
            if isinstance(b_xy, (float, int)):
                b_xy = b_xy * np.ones(sd.num_cells)
            thermal_stress = pp.SecondOrderTensor(kxx=b_xx, kyy=b_yy, kxy=b_xy)

            # Mapping from scalar to vector variables, to be used for discretization
            scalar_vector_mapping = {
                self.darcy_keyword: biot_alpha,
                self.enthalpy_keyword: thermal_stress,
            }
            data[pp.PARAMETERS][self.stress_keyword][
                "scalar_vector_mappings"
            ] = scalar_vector_mapping


class ManuThermoPoroMechSetup3d(  # type: ignore[misc]
    UnitCubeGrid,
    SourceTerms,
    ManuThermoPoroMechDataSaving,
    ManuThermoPoroMechSolutionStrategy3d,
    pp.Thermoporomechanics,
):
    pass
