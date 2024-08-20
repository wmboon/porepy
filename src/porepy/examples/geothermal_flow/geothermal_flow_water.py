from __future__ import annotations

import time

import numpy as np

from model_configuration.DConfigSteamWaterPhasesLowPa import (
    DriesnerWaterFlowModel as FlowModel,
)
from vtk_sampler import VTKSampler

import porepy as pp

day = 86400
tf = 91250.0 * day # final time [750 years]
dt = 912.50 * day # time step size [75 years]
# Pure water single liquid phase
# day = 86400 #seconds in a day.
# tf = 91250.0 * day # final time [250 years]
# dt = 912.50 * day # time step size [2,5 years]
# time_manager = pp.TimeManager(
#     schedule=[0.0, tf],
#     dt_init=dt,
#     constant_dt=True,
#     iter_max=50,
#     print_info=True,
# )

# Pure water and steam - 2Phases - Low pressure gradient and temperature
day = 86400 #seconds in a day.
tf = 730000.0 * day # final time [250 years]
dt = 73000.0 * day # time step size [2,5 years]
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "permeability": 1.0e-15,
        "porosity": 0.1,
        "thermal_conductivity": 0.16,
        "density": 2700.0,
        "specific_heat_capacity": 880.0,
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
    "nl_convergence_tol_res": 1.0e-3,
    "max_iterations": 50,
}

class GeothermalWaterFlowModel(FlowModel):

    def after_nonlinear_convergence(self, iteration_counter) -> None:
        super().after_nonlinear_convergence(iteration_counter)
        print("Number of iterations: ", iteration_counter)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        petsc_solver_q = self.params.get("petsc_solver_q", False)
        eq_idx_map = self.equation_system.assembled_equation_indices
        p_dof_idx = eq_idx_map['pressure_equation']
        z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
        h_dof_idx = eq_idx_map['total_energy_balance']
        t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
        xw_v_dof_idx = eq_idx_map['elimination_of_x_H2O_gas_on_grids_[0]']
        xw_l_dof_idx = eq_idx_map['elimination_of_x_H2O_liq_on_grids_[0]']
        xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']
        xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']

        tb = time.time()
        _, res_g = self.linear_system
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
        print("Saturation residual norm: ", np.linalg.norm(res_g[s_dof_idx]))
        print("x_H2O_gas residual norm: ", np.linalg.norm(res_g[xw_v_dof_idx]))
        print("x_H2O_liq residual norm: ", np.linalg.norm(res_g[xw_l_dof_idx]))
        print("x_NaCl_gas residual norm: ", np.linalg.norm(res_g[xs_v_dof_idx]))
        print("x_NaCl_liq residual norm: ", np.linalg.norm(res_g[xs_l_dof_idx]))
        print("Elapsed time linear solve: ", te - tb)

        return sol

# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 2
# file_name_prefix = "/Users/michealoguntola/porepy/src/porepy/examples/geothermal_flow/model_configuration/constitutive_description/driesner_vtk_files/"
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name_phz = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
file_name_ptz = (
    file_name_prefix + "XTP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)
constant_extended_fields = ['S_v', 'S_l', 'S_h', 'Xl', 'Xv']
brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.constant_extended_fields = constant_extended_fields
brine_sampler_phz.conversion_factors = (1.0, 1.0e-3, 1.0e-5)  # (z,h,p)
model.vtk_sampler = brine_sampler_phz

brine_sampler_ptz = VTKSampler(file_name_ptz)
brine_sampler_ptz.constant_extended_fields = constant_extended_fields
brine_sampler_ptz.conversion_factors = (1.0, 1.0, 1.0e-5)  # (z,t,p)
brine_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
model.vtk_sampler_ptz = brine_sampler_ptz

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