import numpy as np
import time
import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)

from .constitutive_description.BrineConstitutiveDescription import (
    FluidMixture,
    SecondaryEquations,
)
from .geometry_description.geometry_market import SimpleGeometry as ModelGeometry

class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        sides = self.domain_boundary_sides(sd)
        facet_idx = sides.all_bf
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        p_inlet = 20.0
        p_outlet = 15.0
        p = p_outlet * np.ones(boundary_grid.num_cells)
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        h_inlet = 2.0
        h_outlet = 1.5
        h = h_outlet * np.ones(boundary_grid.num_cells)
        h[inlet_idx] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.1
        z_inlet = 1.0e-4
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[inlet_idx] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[inlet_idx] = z_inlet
            return z_NaCl

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)
        t_inlet = 620.0
        t_outlet = 620.0
        T = t_outlet * np.ones(boundary_grid.num_cells)
        T[inlet_idx] = t_inlet
        T[outlet_idx] = t_outlet
        return T


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 15.0
        return np.ones(sd.num_cells) * p_init

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h_init = 1.5
        return np.ones(sd.num_cells) * h_init

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.1
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class ModelEquations(
    PrimaryEquationsCF,
    SecondaryEquations,
):
    """Collecting primary flow and transport equations, and secondary equations
    which provide substitutions for independent saturations and partial fractions.
    """

    def set_equations(self):
        """Call to the equation. Parent classes don't use super(). User must provide
        proper order resultion.

        I don't know why, but the other models are doing it this way was well.
        Maybe it has something to do with the sparsity pattern.

        """
        # Flow and transport in MD setting
        PrimaryEquationsCF.set_equations(self)
        # local elimination of dangling secondary variables
        SecondaryEquations.set_equations(self)


class DriesnerBrineFlowModel(
    ModelGeometry,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):

    @property
    def vtk_sampler(self):
        return self._vtk_sampler

    @vtk_sampler.setter
    def vtk_sampler(self, vtk_sampler):
        self._vtk_sampler = vtk_sampler

    @property
    def vtk_sampler_ptz(self):
        return self._vtk_sampler_phz

    @vtk_sampler_ptz.setter
    def vtk_sampler_ptz(self, vtk_sampler):
        self._vtk_sampler_phz = vtk_sampler

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        return saturation**2

    def temperature_function(self, triplet) -> pp.ad.Operator:
        T_vals, _ = self.temperature_func(*triplet)
        return T_vals

    def after_nonlinear_convergence(self, iteration_counter) -> None:
        tb = time.time()
        _, residual = self.equation_system.assemble(evaluate_jacobian=True)
        res_norm = np.linalg.norm(residual)
        te = time.time()
        print("Elapsed time assemble: ", te - tb)
        print("Time step converged with residual norm: ", res_norm)
        print("Number of iterations: ", iteration_counter)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")
        super().after_nonlinear_convergence(iteration_counter)

    def after_simulation(self):
        self.exporter.write_pvd()

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        petsc_solver_q = self.params.get("petsc_solver_q", False)

        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])

        tb = time.time()
        if petsc_solver_q:
            from petsc4py import PETSc
            # from sklearn.utils import sparsefuncs

            # scale down the equations
            # Pressure residuals in [Kilo Tone / s]: P in [MPa]
            # Compositional residuals in [Kilo Tone / s]: z in [-]
            # Energy residuals in [Kilo Watt / s]: h in [KJ / Kg]
            jac_g, res_g = self.linear_system

            PETSc_jac_g = PETSc.Mat().createAIJ(
                size=jac_g.shape,
                csr=((jac_g.indptr, jac_g.indices, jac_g.data)),
            )

            # solving ls
            st = time.time()
            ksp = PETSc.KSP().create()
            ksp.setOperators(PETSc_jac_g)
            b = PETSc_jac_g.createVecLeft()
            b.array[:] = res_g
            x = PETSc_jac_g.createVecRight()

            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")

            ksp.setConvergenceHistory()
            ksp.solve(b, x)
            sol = x.array
        else:
            csr_mat, res_g = self.linear_system
            sol = super().solve_linear_system()
        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            raise ValueError("Case not implemented yet.")
        te = time.time()
        print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
        print("Pressure residual norm: ", np.linalg.norm(res_g[p_dof_idx]))
        print("Composition residual norm: ", np.linalg.norm(res_g[z_dof_idx]))
        print("Enthalpy residual norm: ", np.linalg.norm(res_g[h_dof_idx]))
        print("Temperature residual norm: ", np.linalg.norm(res_g[t_dof_idx]))
        print("Elapsed time linear solve: ", te - tb)

        self.postprocessing_overshoots(sol)
        self.postprocessing_enthalpy_overshoots(sol)
        return sol

    def postprocessing_overshoots(self, delta_x):

        zmin, zmax, hmin, hmax, pmin, pmax = self.vtk_sampler.search_space.bounds
        z_scale, h_scale, p_scale = self.vtk_sampler.conversion_factors
        zmin /= z_scale
        zmax /= z_scale
        hmin /= h_scale
        hmax /= h_scale
        pmin /= p_scale
        pmax /= p_scale

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])
        xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
        xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        new_p = np.where(new_p < 0.0, 0.0, new_p)
        new_p = np.where(new_p > 100.0, 100.0, new_p)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.where(new_z < 0.0, 0.0, new_z)
        new_z = np.where(new_z > 0.35, 0.35, new_z)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.where(new_h < 0.0, 0.0, new_h)
        new_h = np.where(new_h > 4.0, 4.0, new_h)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.where(new_t < 0.0, 0.0, new_t)
        new_t = np.where(new_t > 1273.15, 1273.15, new_t)
        delta_x[t_dof_idx] = new_t - t_0

        # secondary fractions
        for dof_idx in [s_dof_idx, xw_v_dof_idx, xw_l_dof_idx, xs_v_dof_idx, xs_l_dof_idx]:
            new_q = delta_x[dof_idx] + x0[dof_idx]
            new_q = np.where(new_q < 0.0, 0.0, new_q)
            new_q = np.where(new_q > 1.0, 1.0, new_q)
            delta_x[dof_idx] = new_q - x0[dof_idx]

        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return

    def postprocessing_enthalpy_overshoots(self, delta_x):
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]
        max_dH = 1.0

        dh_overshoots_idx = np.where(np.abs(delta_x[h_dof_idx]) > max_dH)[0]
        if dh_overshoots_idx.shape[0] > 0:
            tb = time.time()
            p0_red = p_0[dh_overshoots_idx]
            h0_red = h_0[dh_overshoots_idx]
            z0_red = z_0[dh_overshoots_idx]
            t = delta_x[t_dof_idx] + t_0
            t_red = t[dh_overshoots_idx]
            h, idx = self.bisection_method(p0_red, z0_red, t_red)
            dh = h - h0_red[idx]
            new_dh = np.where(np.abs(delta_x[h_dof_idx][dh_overshoots_idx][idx]) > max_dH, dh, delta_x[h_dof_idx][dh_overshoots_idx][idx])
            delta_x[h_dof_idx][dh_overshoots_idx][idx] = new_dh
            te = time.time()
            print("Elapsed time for bisection enthalpy correction: ", te - tb)
        return

    def bisection_method(self, p, z, t_target, tol=1e-1, max_iter=100):
        a = np.zeros_like(t_target)
        b = 4.0 * np.ones_like(t_target)
        f_res = lambda H_val: t_target - self.temperature_function(
            np.vstack([p, H_val, z]))

        fa_times_fb = f_res(a) * f_res(b)
        idx = np.where(fa_times_fb < 0.0)[0]
        if idx.shape[0] == 0:
            return np.empty_like(a), idx
        else:
            if np.any(np.logical_and(fa_times_fb > 0.0, np.isclose(fa_times_fb, 0.0))):
                print("Bisection:: Some cells are ignored because fa_times_fb > 0.0 is true.")

            f_res = lambda H_val: t_target[idx] - self.temperature_function(
                np.vstack([p[idx], H_val, z[idx]]))
            a = a[idx]
            b = b[idx]

        for it in range(max_iter):
            c = (a + b) / 2.0
            f_c = f_res(c)

            if np.all(np.logical_or(np.abs(f_c) < tol, np.abs(b - a) < tol)):
                return c, idx

            f_a = f_res(a)
            idx_n = np.where(f_a * f_c < 0)
            idx_p = np.where(f_a * f_c >= 0)
            b[idx_n] = c[idx_n]
            a[idx_p] = c[idx_p]

        raise RuntimeError("Bisection:: Maximum number of iterations reached without convergence.")

