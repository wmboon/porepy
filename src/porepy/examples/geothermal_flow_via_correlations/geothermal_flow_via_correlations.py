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
import time

import scipy.sparse as sps
import matplotlib.pyplot as plt
os.environ["NUMBA_DISABLE_JIT"] = str(0)

import numpy as np
import porepy as pp

tracer_like_setting_q = False
if tracer_like_setting_q:
    from TracerModelConfiguration import TracerFlowModel as FlowModel
else:
    from DriesnerBrineOBL import DriesnerBrineOBL
    from DriesnerModelConfiguration import DriesnerBrineFlowModel as FlowModel

day = 86400
t_scale = 0.1
tf = 2.5 * day * t_scale
dt = 0.025 * day * t_scale
t_eps = 10.0
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {"permeability": 5.0e-14, "porosity": 0.1, "thermal_conductivity": 1.8, 'density': 2650.0, 'specific_heat_capacity': 1000.0}
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "petsc_solver_q": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-3,
    "max_iterations": 500,
}


class GeothermalFlowModel(FlowModel):

    # def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
    #     super().after_nonlinear_iteration(nonlinear_increment)
    #     self.update_secondary_quantities()
    #     # After updating the fluid properties, update discretizations
    #     self.update_discretizations()


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

        p_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[0]])
        z_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[1]])
        h_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[2]])
        t_dof_idx = self.equation_system.dofs_of([self.secondary_variables_names[0]])

        tb = time.time()
        if petsc_solver_q:
            from petsc4py import PETSc

            csr_mat, res_g = self.linear_system

            jac_g = PETSc.Mat().createAIJ(
                size=csr_mat.shape,
                csr=((csr_mat.indptr, csr_mat.indices, csr_mat.data)),
            )

            # solving ls
            st = time.time()
            ksp = PETSc.KSP().create()
            ksp.setOperators(jac_g)
            b = jac_g.createVecLeft()
            b.array[:] = res_g
            x = jac_g.createVecRight()

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
        print("Residual norm at x_k: ", np.linalg.norm(res_g))
        print("Pressure residual norm at x_k: ", np.linalg.norm(res_g[p_dof_idx]))
        print("Composition residual norm at x_k: ", np.linalg.norm(res_g[z_dof_idx]))
        print("Enthalpy residual at norm x_k: ", np.linalg.norm(res_g[h_dof_idx]))
        print("Temperature residual at norm x_k: ", np.linalg.norm(res_g[t_dof_idx]))
        print("Elapsed time linear solve: ", te - tb)

        tb = time.time()
        self.postprocessing_overshoots(sol)
        te = time.time()
        print("Elapsed time for bisection enthalpy correction: ", te - tb)
        return sol

    def postprocessing_overshoots(self, delta_x):
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[0]])
        z_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[1]])
        h_dof_idx = self.equation_system.dofs_of([self.primary_variable_names[2]])
        t_dof_idx = self.equation_system.dofs_of([self.secondary_variables_names[0]])
        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]
        max_dH = 1.0e6

        # control overshoots in:
        # pressure
        new_p = delta_x[p_dof_idx] + p_0
        new_p = np.where(new_p < 0.0, 1.0, new_p)
        new_p = np.where(new_p > 100.0e6, 100.0e6, new_p)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.where(new_z < 1.0e-3, 1.0e-3, new_z)
        new_z = np.where(new_z > 0.3, 0.3, new_z)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.where(new_h < 100.0, 100.0, new_h)
        new_h = np.where(new_h > 4.0e6, 4.0e6, new_h)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.where(new_t < 1.0, 1.0, new_t)
        new_t = np.where(new_t > 1200.0, 1200.0, new_t)
        delta_x[t_dof_idx] = new_t - t_0

        # if np.where(np.abs(delta_x[h_dof_idx]) > max_dH)[0].shape[0] > 0:
        #     print('PostprocessingOvershoots:: Apply bisection correction.')
        #     t = delta_x[t_dof_idx] + t_0
        #     h, idx = self.bisection_method(p_0, z_0, t)
        #     dh = h - h_0[idx]
        #     new_dh = np.where(np.abs(delta_x[h_dof_idx][idx]) > max_dH, dh, delta_x[h_dof_idx][idx])
        #     delta_x[h_dof_idx][idx] = new_dh

        return

    def bisection_method(self, p, z, t_target, tol=1e-3, max_iter=50):
        a = np.zeros_like(t_target)
        b = 4.0e6 * np.ones_like(t_target)
        f_res = lambda H_val : t_target - self.temperature_function(np.vstack([p, H_val, z]))

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


if tracer_like_setting_q:
    model = GeothermalFlowModel(params)
else:
    model = GeothermalFlowModel(params)
    file_name = "binary_files/XHP_l0_modified.vtk"
    brine_obl = DriesnerBrineOBL(file_name)
    brine_obl.fields_constant_extension = ['S_l','S_v', 'Xl', 'Xv']
    brine_obl.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
    model.obl = brine_obl

    if False:
        # h = np.arange(1.5e3, 4.0e6, 0.025e6)
        # p = 20.0e6 * np.ones_like(h)
        # z_NaCl = (0.01 + 1.0e-5) * np.ones_like(h)

        z_NaCl = np.arange(-0.5, 0.5, 0.01)
        h = 1.5e6 * np.ones_like(z_NaCl)
        p = 20.0e6 * np.ones_like(z_NaCl)


        par_points = np.array((z_NaCl, h, p)).T
        brine_obl.sample_at(par_points)

        T = brine_obl.sampled_could.point_data["Temperature"]
        plt.plot(
            z_NaCl,
            T,
            label="T(H)",
            color="blue",
            linestyle="-",
            marker="o",
            markerfacecolor="blue",
            markersize=5,
        )

        s_l = brine_obl.sampled_could.point_data["S_l"]
        s_v = brine_obl.sampled_could.point_data["S_v"]
        plt.plot(z_NaCl, s_l, label='Liquid', color='blue', linestyle='-', marker='o',
                 markerfacecolor='blue', markersize=5)
        plt.plot(z_NaCl, s_v, label='Vapor', color='red', linestyle='-', marker='o',
                 markerfacecolor='red', markersize=5)

        h_l = brine_obl.sampled_could.point_data["H_l"]
        h_v = brine_obl.sampled_could.point_data["H_v"]
        plt.plot(z_NaCl, h_l, label='Liquid', color='blue', linestyle='-', marker='o',
                 markerfacecolor='blue', markersize=5)
        plt.plot(z_NaCl, h_v, label='Vapor', color='red', linestyle='-', marker='o',
                 markerfacecolor='red', markersize=5)

        X_l = brine_obl.sampled_could.point_data["Xl"]
        X_v = brine_obl.sampled_could.point_data["Xv"]
        plt.plot(z_NaCl, X_l, label='Liquid', color='blue', linestyle='-', marker='o',
                 markerfacecolor='blue', markersize=5)
        plt.plot(z_NaCl, X_v, label='Vapor', color='red', linestyle='-', marker='o',
                 markerfacecolor='red', markersize=5)
        plt.legend()
        plt.show()

        aka = 0


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
print("Final simulation time: ", tf)
print("Time step size: ", dt)

sds = model.mdg.subdomains()
flux_op = model.darcy_flux(sds) # this is a facet integrated quantity
mn_flux = flux_op.value(model.equation_system) / sds[0].face_areas
bc_sides = model.domain_boundary_sides(sds[0])
print("normal flux east: ", mn_flux[bc_sides.east])
print("normal flux west: ", mn_flux[bc_sides.west])
print("normal flux north: ", mn_flux[bc_sides.north])
print("normal flux south: ", mn_flux[bc_sides.south])


# load external jacobian
with open('data.npy', 'rb') as f:
    data = np.load(f)
with open('indices.npy', 'rb') as f:
    indices = np.load(f)
with open('indptr.npy', 'rb') as f:
    indptr = np.load(f)

single_physics_jac = sps.csr_matrix((data, indices, indptr))

model.assemble_linear_system()
jac, res = model.linear_system
test = model.primary_equation_names
trial = model.primary_variable_names

i_idx = model.equation_system.assembled_equation_indices[test[0]]
j_idx = model.equation_system.dofs_of(trial[0:1])

multi_physics_jac = jac[i_idx,:][:,j_idx]
aka = 0

