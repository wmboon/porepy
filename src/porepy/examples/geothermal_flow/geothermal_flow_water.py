from __future__ import annotations

import time

import numpy as np

from model_configuration.DConfigSteamWaterPhasesLowPa import (
    DriesnerWaterFlowModel as FlowModel,
)
from vtk_sampler import VTKSampler
from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
import matplotlib.pyplot as plt
import porepy as pp

# Pure water and steam - 2Phases - Low pressure gradient and temperature
day = 86400 #seconds in a day.
tf = 730000.0 * day # final time [2000 years]
dt = 7300.0 * day # time step size [2 years]
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
        "thermal_conductivity": 2.0,
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
    "nl_convergence_tol_res": 1.0e-4,
    "max_iterations": 400,
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

    def temperature_function(self, triplet) -> pp.ad.Operator:
        T_vals, _ = self.temperature_func(*triplet)
        return T_vals

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""

        eq_idx_map = self.equation_system.assembled_equation_indices
        p_dof_idx = eq_idx_map['pressure_equation']
        z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
        h_dof_idx = eq_idx_map['total_energy_balance']
        t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']

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
        print("Elapsed time linear solve: ", te - tb)

        def newton_increment_constraint(res_norm):
            if res_norm < 0.001:
                return 1.0
            elif 0.001 <= res_norm < np.pi:
                return 1.0/np.pi
            elif np.pi <= res_norm < 10.0*np.pi:
                return 1.0 / res_norm
            else:
                return 1.0/10.0*np.pi
        enthalpy_alpha = newton_increment_constraint(np.linalg.norm(res_g[h_dof_idx]))
        print("Enthalpy residual norm and constraint: ",
              (np.linalg.norm(res_g[h_dof_idx]), enthalpy_alpha))
        self.postprocessing_overshoots(sol, enthalpy_alpha)
        # sol = self.increment_from_projected_solution()
        return sol

    def load_and_project_reference_data(self):

        # doi: 10.1111/gfl.12080
        p_data = np.genfromtxt('pressure_data_fig6A.csv', delimiter=',', skip_header=1)
        t_data = np.genfromtxt('temperature_data_fig6A.csv', delimiter=',', skip_header=1)
        sl_data = np.genfromtxt('saturation_liq_data_fig6B.csv', delimiter=',', skip_header=1)

        p_data[:, 0] *= 1.0e3
        t_data[:, 0] *= 1.0e3
        sl_data[:, 0] *= 1.0e3

        p_data[:, 1] *= 1.0e6
        t_data[:, 1] += 273.15

        xc = self.mdg.subdomains()[0].cell_centers.T
        p_proj = np.interp(xc[:, 0], p_data[:, 0], p_data[:, 1])
        t_proj = np.interp(xc[:, 0], t_data[:, 0], t_data[:, 1])
        s_proj = np.interp(xc[:, 0], sl_data[:, 0], sl_data[:, 1])

        # triple point of water
        T_ref = 273.16
        P_ref = 611.657
        liquid = IAPWS95Liquid(T=T_ref, P=P_ref, zs=[1])
        gas = IAPWS95Gas(T=T_ref, P=P_ref, zs=[1])
        flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

        def bisection(p, s, tol=1e-8, max_iter=1000):
            a = 0.0
            b = 1.0

            def func(p, s, v):
                PV = flasher.flash(P=p, VF=v)
                assert len(PV.betas_volume) == 2
                res = s - PV.betas_volume[0]
                return res

            if func(p, s, a) * func(p, s, b) >= 0:
                raise ValueError("f(a) and f(b) must have opposite signs")

            for _ in range(max_iter):
                c = (a + b) / 2
                if abs(func(p, s, c)) < tol or (b - a) / 2 < tol:
                    return c
                elif func(p, s, c) * func(p, s, a) < 0:
                    b = c
                else:
                    a = c
            raise RuntimeError("Maximum iterations exceeded")

        h_data = []
        for i, pair in enumerate(zip(p_proj, t_proj)):
            s_v = 1.0 - s_proj[i]
            if np.isclose(s_v, 0.0) or np.isclose(s_v, 1.0):
                PT = flasher.flash(P=pair[0], T=pair[1])
                h_data.append(PT.H_mass())
            else:
                vf = bisection(pair[0], s_v)
                PT = flasher.flash(P=pair[0], VF=vf)
                h_data.append(PT.H_mass())
        h_proj = np.array(h_data)
        return p_proj, h_proj, t_proj, s_proj

    def increment_from_projected_solution(self):

        # triple point of water
        T_ref = 273.16
        P_ref = 611.657
        # MW_H2O = iapws_constants.MWs[0] * 1.0e-3  # [Kg/mol]
        liquid = IAPWS95Liquid(T=T_ref, P=P_ref, zs=[1])
        gas = IAPWS95Gas(T=T_ref, P=P_ref, zs=[1])
        flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])


        zmin, zmax, hmin, hmax, pmin, pmax = self.vtk_sampler.search_space.bounds
        z_scale, h_scale, p_scale = self.vtk_sampler.conversion_factors
        zmin /= z_scale
        zmax /= z_scale
        hmin /= h_scale
        hmax /= h_scale
        pmin /= p_scale
        pmax /= p_scale

        x0 = self.equation_system.get_variable_values(iterate_index=0)
        x_k = np.zeros_like(x0)

        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])
        xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
        xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        # project data
        p_proj, h_proj, t_proj, s_proj = self.load_and_project_reference_data()
        z_proj = np.zeros_like(s_proj)
        x_k[p_dof_idx] = p_proj
        x_k[z_dof_idx] = z_proj
        x_k[h_dof_idx] = h_proj

        x_k[t_dof_idx] = t_proj
        x_k[s_dof_idx] = 1.0 - s_proj
        x_k[xw_l_dof_idx] = np.ones_like(s_proj) - z_proj
        x_k[xw_v_dof_idx] = np.ones_like(s_proj) - z_proj
        x_k[xs_l_dof_idx] = z_proj
        x_k[xs_v_dof_idx] = z_proj

        delta_x = x_k - x0
        return delta_x

    def postprocessing_overshoots(self, delta_x, enthalpy_alpha):

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
        new_p = np.where(new_p > 100.0e6, 100.0e6, new_p)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.where(new_z < 0.0, 0.0, new_z)
        new_z = np.where(new_z > 0.5, 0.5, new_z)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.where(new_h < 0.0, 0.0, new_h)
        new_h = np.where(new_h > 4.0e6, 4.0e6, new_h)
        delta_x[h_dof_idx] = (new_h - h_0) * enthalpy_alpha

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
            delta_x[dof_idx] = (new_q - x0[dof_idx])

        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return

# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 0
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

#
# P_proj, H_proj, T_proj, S_proj = model.load_and_project_reference_data()
#
# z_proj = (1.0e-4) * np.ones_like(S_proj)
# par_points = np.array((z_proj, H_proj, P_proj)).T
# model.vtk_sampler.sample_at(par_points)
# H_vtk = model.vtk_sampler.sampled_could.point_data['H']
# T_vtk = model.vtk_sampler.sampled_could.point_data['Temperature']
# S_vtk = model.vtk_sampler.sampled_could.point_data['S_l']
#
# def draw_and_save_comparison(T_proj,T_vtk,S_proj,S_vtk,H_proj,H_vtk):
#     # plot the data
#     figure_data = {
#         'T': ('temperarure_at_2000_years.png', 'T - Fig. 6A P. WEIS (2014)', 'T - VTKsample + GEOMAR'),
#         'S': ('liquid_saturation_at_2000_years.png', 's_l - Fig. 6B P. WEIS (2014)', 's_l - VTKsample + GEOMAR'),
#         'H': ('enthalpy_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'H - VTKsample + GEOMAR'),
#     }
#     fields_data = {
#         'T': (T_proj,T_vtk),
#         'S': (S_proj,S_vtk),
#         'H': (H_proj,H_vtk),
#     }
#
#     xc = model.mdg.subdomains()[0].cell_centers.T
#     cell_vols = model.mdg.subdomains()[0].cell_volumes
#     for item in fields_data.items():
#         field, data = item
#         file_name, label_ref, label_vtk = figure_data[field]
#         x = xc[:, 0]
#         y1 = data[0]
#         y2 = data[1]
#
#         l2_norm = np.linalg.norm((data[0] - data[1])*cell_vols) / np.linalg.norm(data[0] *cell_vols)
#
#         plt.plot(x, y1, label=label_ref)
#         plt.plot(x, y2, label=label_vtk, linestyle='--')
#
#         plt.xlabel('Distance [Km]')
#         plt.title('Relative l2_norm = ' + str(l2_norm))
#         plt.legend()
#         plt.savefig(file_name)
#         plt.clf()
#
# draw_and_save_comparison(T_proj,T_vtk,S_proj,S_vtk,H_proj,H_vtk)

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

P_num = model.equation_system.get_variable_values(['pressure'],time_step_index=0)
H_num = model.equation_system.get_variable_values(['enthalpy'],time_step_index=0)
S_num = 1.0 - model.equation_system.get_variable_values(['s_gas'],time_step_index=0)
T_num = model.equation_system.get_variable_values(['temperature'],time_step_index=0)

def draw_and_save_comparison_numeric(T_proj,T_num,S_proj,S_num,H_proj,H_num,P_proj,P_num):
    # plot the data
    figure_data = {
        'T': ('pp_temperarure_at_2000_years.png', 'T - Fig. 6A P. WEIS (2014)', 'T - numeric + GEOMAR'),
        'S': ('pp_liquid_saturation_at_2000_years.png', 's_l - Fig. 6B P. WEIS (2014)', 's_l - numeric + GEOMAR'),
        'H': ('pp_enthalpy_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'H - numeric '),
        'P': ('pp_pressure_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'P - numeric '),
    }
    fields_data = {
        'T': (T_proj,T_num),
        'S': (S_proj,S_num),
        'H': (H_proj,H_num),
        'P': (P_proj, P_num),
    }

    xc = model.mdg.subdomains()[0].cell_centers.T
    cell_vols = model.mdg.subdomains()[0].cell_volumes
    for item in fields_data.items():
        field, data = item
        file_name, label_ref, label_vtk = figure_data[field]
        x = xc[:, 0]
        y1 = data[0]
        y2 = data[1]

        l2_norm = np.linalg.norm((data[0] - data[1])*cell_vols) / np.linalg.norm(data[0] *cell_vols)

        plt.plot(x, y1, label=label_ref)
        plt.plot(x, y2, label=label_vtk, linestyle='--')

        plt.xlabel('Distance [Km]')
        plt.title('Relative l2_norm = ' + str(l2_norm))
        plt.legend()
        plt.savefig(file_name)
        plt.clf()

P_proj, H_proj, T_proj, S_proj = model.load_and_project_reference_data()
draw_and_save_comparison_numeric(T_proj,T_num,S_proj,S_num,H_proj,H_num,P_proj,P_num)