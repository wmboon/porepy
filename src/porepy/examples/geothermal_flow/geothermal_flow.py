"""Example implementing a multi-phase multi component flow of H2O-NaCl using Driesner
correlations and a tracer-like as constitutive descriptions.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturation, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""

from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import time

import numpy as np
from model_configuration.DriesnerModelConfiguration import (
    DriesnerBrineFlowModel as FlowModel,
)
from vtk_sampler import VTKSampler

import porepy as pp

day = 86400
tf = 0.01 * day
dt = 0.0001 * day
dynamic_time_step_q = False

if dynamic_time_step_q:
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=False,
        dt_min_max = (dt, 0.1 * day),
        iter_optimal_range = (5, 10),
        iter_relax_factors = (0.5,1.5),
        recomp_factor = 0.25,
        iter_max=50,
        print_info=True,
    )
else:
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )

solid_constants = pp.SolidConstants(
    {
        "permeability": 5.0e-14,
        "porosity": 0.1,
        "thermal_conductivity": 1.8,
        "density": 2650.0,
        "specific_heat_capacity": 1000.0,
    }
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-2,
    "max_iterations": 50,
    "petsc_solver_q": True,
}


class GeothermalFlowModel(FlowModel):

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
        new_p = np.where(new_p > pmax, pmax, new_p)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.where(new_z < 0.0, 0.0, new_z)
        new_z = np.where(new_z > zmax, zmax, new_z)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.where(new_h < 0.0, 0.0, new_h)
        new_h = np.where(new_h > hmax, hmax, new_h)
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
        p_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[0]])
        z_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[1]])
        h_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[2]])
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
                print("Bisection:: some cells are ignored.")

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

        raise RuntimeError("Maximum number of iterations reached without convergence.")


# Instance of the computational model
model = GeothermalFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
brine_sampler = VTKSampler(file_name)
brine_sampler.conversion_factors = (1.0, 1.0e3, 10.0)  # (z,h,p)
model.vtk_sampler = brine_sampler


tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])

# Check conservation of overall mass across boundaries
# external_bc_idx = bc_sides.all_bf
# assert np.isclose(np.sum(mn[external_bc_idx]), 0.0, atol=1.0e-10)


