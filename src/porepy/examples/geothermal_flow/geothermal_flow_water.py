from __future__ import annotations

import time
from typing import Any
import numpy as np

from model_configuration.DConfigSteamWaterPhasesLowPa import (
    DriesnerWaterFlowModel as FlowModel,
)

from vtk_sampler import VTKSampler
from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
import matplotlib.pyplot as plt
import porepy as pp
from porepy.models.compositional_flow import update_phase_properties

# scale
M_scale = 1.0e-6
s_tol = 9.0
day = 86400 #seconds in a day.
year = 365.0 * day
tf = 2000.0 * year # final time [2000 years]
# dt = 2000.0 * year # time step size [2000 years]
dt = 1.0 * year # time step size [1.0 years]
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
        "thermal_conductivity": 2.0*M_scale,
        "density": 2700.0,
        "specific_heat_capacity": 880.0*M_scale,
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
    "nl_convergence_mass_tol_res": s_tol * 1.0e-5,
    "nl_convergence_energy_tol_res": s_tol * 1.0e-5,
    "nl_convergence_temperature_tol_res": s_tol * 1.0e-1,
    "nl_convergence_fractions_tol_res": s_tol * 1.0e-3,
    "max_iterations": 100,
}

class GeothermalWaterFlowModel(FlowModel):

    def after_nonlinear_convergence(self, iteration_counter) -> None:
        day = 86400
        year = 365.0 * day
        super().after_nonlinear_convergence(iteration_counter)
        print("Number of iterations: ", iteration_counter)
        print("Time value [years]: ", self.time_manager.time / year)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def after_simulation(self):
        self.exporter.write_pvd()

    def temperature_function(self, triplet) -> pp.ad.Operator:
        T_vals, _ = self.temperature_func(*triplet)
        return T_vals

    def beta_mass_function(self, triplet) -> pp.ad.Operator:
        beta_vals = self.beta_mass_func(*triplet)
        return beta_vals

    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""

        eq_idx_map = self.equation_system.assembled_equation_indices
        eq_p_dof_idx = eq_idx_map['pressure_equation']
        eq_z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
        eq_h_dof_idx = eq_idx_map['total_energy_balance']
        eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
        eq_xw_v_dof_idx = eq_idx_map['elimination_of_x_H2O_gas_on_grids_[0]']
        eq_xw_l_dof_idx = eq_idx_map['elimination_of_x_H2O_liq_on_grids_[0]']
        eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
        eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']

        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        s_dof_idx = self.equation_system.dofs_of(['s_gas'])
        xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
        xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
        xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
        xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

        # primary system
        eq_p_idx = np.concatenate([eq_p_dof_idx, eq_z_dof_idx, eq_h_dof_idx, ])
        var_p_idx = np.concatenate([p_dof_idx, z_dof_idx, h_dof_idx])

        eq_s_idx = np.concatenate([eq_t_dof_idx, eq_s_dof_idx, eq_xw_v_dof_idx, eq_xw_l_dof_idx, eq_xs_v_dof_idx, eq_xs_l_dof_idx])
        var_s_idx = np.concatenate([t_dof_idx, s_dof_idx, xw_v_dof_idx, xw_l_dof_idx, xs_v_dof_idx, xs_l_dof_idx])

        jac_g, res_g = self.linear_system
        print("Overall residual norm at x_k: ", np.linalg.norm(res_g))
        print("Pressure residual norm: ", np.linalg.norm(res_g[eq_p_dof_idx]))
        print("Composition residual norm: ", np.linalg.norm(res_g[eq_z_dof_idx]))
        print("Enthalpy residual norm: ", np.linalg.norm(res_g[eq_h_dof_idx]))
        print("Temperature residual norm: ", np.linalg.norm(res_g[eq_t_dof_idx]))
        print("Saturation residual norm: ", np.linalg.norm(res_g[eq_s_dof_idx]))
        print("Xw_v residual norm: ", np.linalg.norm(res_g[eq_xw_v_dof_idx]))
        print("Xw_l residual norm: ", np.linalg.norm(res_g[eq_xw_l_dof_idx]))
        print("Xs_v residual norm: ", np.linalg.norm(res_g[eq_xs_v_dof_idx]))
        print("Xs_l residual norm: ", np.linalg.norm(res_g[eq_xs_l_dof_idx]))


        tb = time.time()
        # global solve
        # delta_x = super().solve_linear_system().copy()

        # partial solve
        delta_x = np.zeros_like(res_g)
        jac_p = jac_g[eq_p_idx[:, None], var_p_idx]
        res_p = res_g[eq_p_idx]
        self.linear_system = (jac_p, res_p)
        delta_x_p = super().solve_linear_system()
        delta_x[var_p_idx] = delta_x_p

        # equilibrate secondary fields
        # [var_s_idx] = self.rectify_secondary_fields(delta_x)
        # delta_x[var_s_idx] = -1.0 * res_g[eq_s_idx]

        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            raise ValueError("Case not implemented yet.")
        te = time.time()

        # project_sol_Q = False
        # if project_sol_Q:
        #     delta_x = self.increment_from_projected_solution()
            # x0 = self.equation_system.get_variable_values(iterate_index=0)
            # x_proj = delta_x + x0
            # self.equation_system.set_variable_values(values=x_proj, iterate_index=0)
            # self.update_all_constitutive_expressions()
            # jac_g , res_g = self.equation_system.assemble(evaluate_jacobian=True)
            # self.linear_system = (jac_g , res_g)
        print("Elapsed time linear solve: ", te - tb)
        # self.recompute_secondary_residuals(delta_x, res_g)

        self.postprocessing_overshoots(delta_x)

        line_search_Q = True
        if line_search_Q:
            tb = time.time()
            x = self.equation_system.get_variable_values(iterate_index=0).copy()
            # self.postprocessing_secondary_variables_increments(x, delta_x, res_g)
            # Line search: backtracking to satisfy Armijo condition per field

            dofs_idx = {
                'p': (eq_p_dof_idx, p_dof_idx),
                'z': (eq_z_dof_idx, z_dof_idx),
                'h': (eq_h_dof_idx, h_dof_idx),
                't': (eq_t_dof_idx, t_dof_idx),
                's': (eq_s_dof_idx, s_dof_idx),
                'xw_v': (eq_xw_v_dof_idx, xw_v_dof_idx),
                'xw_l': (eq_xw_l_dof_idx, xw_l_dof_idx),
                'xs_v': (eq_xs_v_dof_idx, xs_v_dof_idx),
                'xs_l': (eq_xs_l_dof_idx, xs_l_dof_idx),
            }

            fields_idx = {
                'p': 0,
                'z': 1,
                'h': 2,
                # 't': 3,
                # 's': 4,
                # 'xw_v': 5,
                # 'xw_l': 6,
                # 'xs_v': 7,
                # 'xs_l': 8,
            }

            eps_tol = 0.0
            field_to_skip = []
            for item in fields_idx.items():
                field_name, field_idx = item
                eq_idx, _ = dofs_idx[field_name]
                if np.linalg.norm(res_g[eq_idx]) < eps_tol:
                    field_to_skip.append(field_name)
            print('No line search performed on the fields: ', field_to_skip)
            max_searches = 25
            beta = 2.0/3.0  # reduction factor for alpha
            c = 1.0e-6  # Armijo condition constant
            alpha = np.ones(3) # initial step size
            k = 0
            x_k = x + delta_x

            Armijo_condition = [True, True, True]
            for i, field_name in enumerate(fields_idx.keys()):
                if field_name in field_to_skip:
                    Armijo_condition[i] = False
            Armijo_condition = np.array(Armijo_condition)
            while np.any(Armijo_condition) and (len(field_to_skip) < 5):
                for item in fields_idx.items():
                    field_name, field_idx = item
                    if field_name in field_to_skip:
                        continue
                    _, dof_idx = dofs_idx[field_name]
                    x_k[dof_idx] = x[dof_idx] + alpha[field_idx] * delta_x[dof_idx]
                # set new state
                x_k[var_s_idx] = self.rectify_secondary_fields(x_k - x)
                self.equation_system.set_variable_values(values=x_k, iterate_index=0)
                self.update_secondary_quantities()
                self.update_discretizations()
                res_g_k = self.equation_system.assemble(evaluate_jacobian=False)
                for item in fields_idx.items():
                    field_name, field_idx = item
                    if field_name in field_to_skip:
                        continue
                    eq_idx, dof_idx = dofs_idx[field_name]
                    Armijo_condition[field_idx] = np.any(np.linalg.norm(res_g_k[eq_idx]) > np.linalg.norm(res_g[eq_idx]) + c * alpha[field_idx] * np.dot(res_g[eq_idx], delta_x[dof_idx]))
                    # Armijo_condition[field_idx] = np.any(np.linalg.norm(res_g_k) > np.linalg.norm(res_g) + c * np.mean(alpha) * np.dot(res_g, delta_x))
                    if Armijo_condition[field_idx]:
                        alpha[field_idx] *= beta
                k+=1
                if k == max_searches:
                    print("The backtracking line search has reached the maximum number of iterations.")
                    break
            print("alphas per field: ", alpha)
            # # Scaled the increment per field
            # for item in fields_idx.items():
            #     field_name, field_idx = item
            #     if field_name in field_to_skip:
            #         continue
            #     _, dof_idx = dofs_idx[field_name]
            #     delta_x[dof_idx] *= alpha[field_idx]
            # adjusted increment
            delta_x = x_k - x
            self.equation_system.set_variable_values(values=x, iterate_index=0)
            # self.postprocessing_thermal_overshoots(delta_x)
            # self.rectify_secondary_fields(delta_x)
            # if k == max_searches:
            #     res_tol_mass = self.params['nl_convergence_mass_tol_res']
            #     res_tol_energy = self.params['nl_convergence_energy_tol_res']
            #     res_tol_fractions = self.params['nl_convergence_fractions_tol_res']
            #     res_tol = np.max([res_tol_mass, res_tol_energy, res_tol_fractions])
            #     res_mass_norm = np.linalg.norm(
            #         res_g[np.concatenate([eq_p_dof_idx, eq_z_dof_idx])])
            #     res_energy_norm = np.linalg.norm(res_g[eq_h_dof_idx])
            #     res_fractions_norm = np.linalg.norm(res_g[eq_s_idx])
            #     primary_residuals = [res_mass_norm, res_energy_norm, res_fractions_norm]
            #     converged_state_Q = np.all(np.array(primary_residuals) < res_tol)
            #     if converged_state_Q:
            #         # this method aims to correct secondary variables
            #         self.postprocessing_secondary_variables_increments(x, delta_x, res_g_k)

            te = time.time()
            print("Elapsed time for backtracking line search: ", te - tb)
        print("End of solution procedure")
        print("")
        print("")
        return delta_x

    def load_and_project_reference_data(self):

        # doi: 10.1111/gfl.12080
        file_prefix = 'verification_low_salt_content/'
        p_data = np.genfromtxt(file_prefix + 'fig_6a_pressure.csv', delimiter=',', skip_header=1)
        t_data = np.genfromtxt(file_prefix+ 'fig_6a_temperature.csv', delimiter=',', skip_header=1)
        sl_data = np.genfromtxt(file_prefix + 'fig_6b_liquid_saturation.csv', delimiter=',', skip_header=1)

        p_data[:, 0] *= 1.0e3
        t_data[:, 0] *= 1.0e3
        sl_data[:, 0] *= 1.0e3

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
                PV = flasher.flash(P=p*1.0e6, VF=v)
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
                PT = flasher.flash(P=pair[0]*1.0e6, T=pair[1])
                h_data.append(PT.H_mass()*1.0e-6)
            else:
                vf = bisection(pair[0], s_v)
                PT = flasher.flash(P=pair[0]*1.0e6, VF=vf)
                h_data.append(PT.H_mass()*1.0e-6)
        h_proj = np.array(h_data)
        return p_proj, h_proj, t_proj, s_proj

    def increment_from_projected_solution(self):

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
        new_p = np.where(new_p < 1.0e-6, 1.0e-6, new_p)
        new_p = np.where(new_p > 100.0, 100.0, new_p)
        delta_x[p_dof_idx] = new_p - p_0

        # composition
        new_z = delta_x[z_dof_idx] + z_0
        new_z = np.where(new_z < 0.0, 0.0, new_z)
        new_z = np.where(new_z > 0.5, 0.5, new_z)
        delta_x[z_dof_idx] = new_z - z_0

        # enthalpy
        new_h = delta_x[h_dof_idx] + h_0
        new_h = np.where(new_h < 1.0e-6, 1.0e-6, new_h)
        new_h = np.where(new_h > 4.0, 4.0, new_h)
        delta_x[h_dof_idx] = new_h - h_0

        # temperature
        new_t = delta_x[t_dof_idx] + t_0
        new_t = np.where(new_t < 100.0, 100.0, new_t)
        new_t = np.where(new_t > 1273.15, 1273.15, new_t)
        delta_x[t_dof_idx] = (new_t - t_0)

        # secondary fractions
        for dof_idx in [s_dof_idx, xw_v_dof_idx, xw_l_dof_idx, xs_v_dof_idx, xs_l_dof_idx]:
            new_q = delta_x[dof_idx] + x0[dof_idx]
            new_q = np.where(new_q < 0.0, 0.0, new_q)
            new_q = np.where(new_q > 1.0, 1.0, new_q)
            delta_x[dof_idx] = new_q - x0[dof_idx]


        te = time.time()
        print("Elapsed time for postprocessing overshoots: ", te - tb)
        return

    def postprocessing_thermal_overshoots(self, delta_x):

        dh_max = 1.0e-3
        x_n_m_one = self.equation_system.get_variable_values(time_step_index=0)
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
        t_dof_idx = self.equation_system.dofs_of(['temperature'])
        p_0 = x0[p_dof_idx]
        z_0 = x0[z_dof_idx]
        h_0 = x0[h_dof_idx]
        t_0 = x0[t_dof_idx]

        p_k = delta_x[p_dof_idx] + x0[p_dof_idx]
        z_k = delta_x[z_dof_idx] + x0[z_dof_idx]
        h_k = delta_x[h_dof_idx] * x0[h_dof_idx]  # delayed enthalpy
        par_points = np.array((z_k, h_k, p_k)).T
        self.vtk_sampler.sample_at(par_points)

        rho_v_k = self.vtk_sampler.sampled_could.point_data['Rho_v']
        s_k = self.vtk_sampler.sampled_could.point_data['S_v']
        Rho_k = self.vtk_sampler.sampled_could.point_data['Rho']
        beta_mass_v = s_k * rho_v_k / Rho_k

        multiphase_Q = np.logical_and(beta_mass_v > 0.0, beta_mass_v < 1.0)
        h_overshoots_Q = np.abs(delta_x[h_dof_idx]) > dh_max
        overshoots_idx = np.where(np.logical_and(h_overshoots_Q,multiphase_Q))[0]
        if overshoots_idx.shape[0] > 0:
            tb = time.time()
            p0_red = p_0[overshoots_idx]
            h0_red = h_0[overshoots_idx]
            z0_red = z_0[overshoots_idx]
            beta_red = beta_mass_v[overshoots_idx]
            h, idx = self.bisection_method(p0_red, z0_red, beta_red)
            if idx.shape[0] != 0:
                print("Applying enthalpy correction.")
                new_dh = h - h0_red[idx]
                delta_x[h_dof_idx[overshoots_idx[idx]]] = new_dh
            te = time.time()
            print("Elapsed time for bisection enthalpy correction: ", te - tb)
        return

    def bisection_method(self, p, z, beta_target, tol=1e-3, max_iter=300):
        a = (1.0e-6) * np.zeros_like(beta_target)
        b = 4.0 * np.ones_like(beta_target)
        f_res = lambda H_val: beta_target - self.beta_mass_function(
            np.vstack([p, H_val, z]))

        fa_times_fb = f_res(a) * f_res(b)
        idx = np.where(fa_times_fb < 0.0)[0]
        if idx.shape[0] == 0:
            return np.empty_like(a), idx
        else:
            if np.any(np.logical_and(fa_times_fb > 0.0, np.isclose(fa_times_fb, 0.0))):
                print("Bisection:: Some cells are ignored because fa_times_fb > 0.0 is true.")

            f_res = lambda H_val: beta_target[idx] - self.beta_mass_function(
                np.vstack([p[idx], H_val, z[idx]]))
            a = a[idx]
            b = b[idx]

        for it in range(max_iter):
            c = (a + b) / 2.0
            f_c = f_res(c)

            if np.all(np.logical_or(np.abs(f_c) < tol, np.abs(b - a) < tol)):
                return c, idx

            f_a = f_res(a)
            idx_n = np.where(f_a * f_c < 0.0)
            idx_p = np.where(np.logical_or(f_a * f_c > 0.0, np.isclose(f_a * f_c, 0.0)))
            b[idx_n] = c[idx_n]
            a[idx_p] = c[idx_p]

        raise RuntimeError("Bisection:: Maximum number of iterations reached without convergence.")

    def postprocessing_secondary_variables_increments(self, x0, delta_x, res_g):

        eq_idx_map = self.equation_system.assembled_equation_indices
        eq_p_dof_idx = eq_idx_map['pressure_equation']
        eq_z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
        eq_h_dof_idx = eq_idx_map['total_energy_balance']

        eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
        eq_xw_v_dof_idx = eq_idx_map['elimination_of_x_H2O_gas_on_grids_[0]']
        eq_xw_l_dof_idx = eq_idx_map['elimination_of_x_H2O_liq_on_grids_[0]']
        eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
        eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']

        res_tol_mass = self.params['nl_convergence_mass_tol_res']
        res_tol_energy = self.params['nl_convergence_energy_tol_res']
        res_tol_temperature = self.params['nl_convergence_temperature_tol_res']
        res_tol = np.max([res_tol_mass, res_tol_energy, res_tol_temperature])
        res_p_norm = np.linalg.norm(res_g[eq_p_dof_idx])
        res_z_norm = np.linalg.norm(res_g[eq_z_dof_idx])
        res_h_norm = np.linalg.norm(res_g[eq_h_dof_idx])
        primary_residuals = [res_p_norm,res_z_norm,res_h_norm]
        converged_state_Q = np.all(np.array(primary_residuals) < res_tol)
        print('Secondary_variables_increments:: converged_state_Q: ', converged_state_Q)

        res_t_norm = np.linalg.norm(res_g[eq_t_dof_idx])
        res_s_norm = np.linalg.norm(res_g[eq_s_dof_idx])
        res_xw_v_norm = np.linalg.norm(res_g[eq_xw_v_dof_idx])
        res_xw_l_norm = np.linalg.norm(res_g[eq_xw_l_dof_idx])
        res_xs_v_norm = np.linalg.norm(res_g[eq_xs_v_dof_idx])
        res_xs_l_norm = np.linalg.norm(res_g[eq_xs_l_dof_idx])
        secondary_residuals = [res_t_norm, res_s_norm, res_xw_v_norm,res_xw_l_norm,res_xs_v_norm,res_xs_l_norm]
        secondary_converged_states = np.array(secondary_residuals) < res_tol
        if converged_state_Q:

            tb = time.time()
            # x0 = self.equation_system.get_variable_values(iterate_index=0)
            p_dof_idx = self.equation_system.dofs_of(['pressure'])
            z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
            h_dof_idx = self.equation_system.dofs_of(['enthalpy'])
            t_dof_idx = self.equation_system.dofs_of(['temperature'])
            s_dof_idx = self.equation_system.dofs_of(['s_gas'])
            xw_v_dof_idx = self.equation_system.dofs_of(['x_H2O_gas'])
            xw_l_dof_idx = self.equation_system.dofs_of(['x_H2O_liq'])
            xs_v_dof_idx = self.equation_system.dofs_of(['x_NaCl_gas'])
            xs_l_dof_idx = self.equation_system.dofs_of(['x_NaCl_liq'])

            p_k = delta_x[p_dof_idx] + x0[p_dof_idx]
            z_k = delta_x[z_dof_idx] + x0[z_dof_idx]
            h_k = delta_x[h_dof_idx] + x0[h_dof_idx]
            par_points = np.array((z_k, h_k, p_k)).T
            self.vtk_sampler.sample_at(par_points)
            t_k = self.vtk_sampler.sampled_could.point_data['Temperature']
            s_k = self.vtk_sampler.sampled_could.point_data['S_v']
            Xw_v_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xv']
            Xw_l_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xl']
            Xs_v_k = self.vtk_sampler.sampled_could.point_data['Xv']
            Xs_l_k = self.vtk_sampler.sampled_could.point_data['Xl']

            delta_t = t_k - x0[t_dof_idx]
            delta_s = s_k - x0[s_dof_idx]
            delta_Xw_v = Xw_v_k - x0[xw_v_dof_idx]
            delta_Xw_l = Xw_l_k - x0[xw_l_dof_idx]
            delta_Xs_v = Xs_v_k - x0[xs_v_dof_idx]
            delta_Xs_l = Xs_l_k - x0[xs_l_dof_idx]

            # def newton_increment_constraint(res_norm):
            #     if res_norm < 0.01:
            #         return 1.0
            #     elif 0.01 <= res_norm < np.pi:
            #         return 1.0/np.pi
            #     elif np.pi <= res_norm < 10.0*np.pi:
            #         return 1.0 / res_norm
            #     else:
            #         return 1.0/10.0*np.pi

            deltas = [delta_t,delta_s,delta_Xw_v,delta_Xw_l,delta_Xs_v,delta_Xs_l]
            dofs_idx = [t_dof_idx,s_dof_idx,xw_v_dof_idx,xw_l_dof_idx,xs_v_dof_idx,xs_l_dof_idx]
            # update deltas
            for k_field, conv_state in enumerate(secondary_converged_states):
                if conv_state:
                    continue
                delta = deltas[k_field]
                dof_idx = dofs_idx[k_field]
                # alpha_scale = newton_increment_constraint(secondary_residuals[k_field])
                # delta_x[dof_idx] = delta * alpha_scale
                delta_x[dof_idx] = delta
            te = time.time()
            print("Elapsed time for postprocessing secondary increments: ", te - tb)
        return

    def recompute_secondary_residuals(self, delta_x, res_g):

        eq_idx_map = self.equation_system.assembled_equation_indices
        eq_p_dof_idx = eq_idx_map['pressure_equation']
        eq_z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
        eq_h_dof_idx = eq_idx_map['total_energy_balance']

        eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
        eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
        eq_xw_v_dof_idx = self.equation_system.dofs_of(
            ['elimination_of_x_H2O_gas_on_grids_[0]'])
        eq_xw_l_dof_idx = self.equation_system.dofs_of(
            ['elimination_of_x_H2O_liq_on_grids_[0]'])
        eq_xs_v_dof_idx = self.equation_system.dofs_of(
            ['elimination_of_x_NaCl_liq_on_grids_[0]'])
        eq_xs_l_dof_idx = self.equation_system.dofs_of(
            ['elimination_of_x_NaCl_gas_on_grids_[0]'])

        res_tol = self.params['nl_convergence_tol_res']
        res_p_norm = np.linalg.norm(res_g[eq_p_dof_idx])
        res_z_norm = np.linalg.norm(res_g[eq_z_dof_idx])
        res_h_norm = np.linalg.norm(res_g[eq_h_dof_idx])
        primary_residuals = [res_p_norm, res_z_norm, res_h_norm]
        converged_state_Q = np.all(np.array(primary_residuals) < res_tol)
        print('converged_state_Q: ', converged_state_Q)

        res_t_norm = np.linalg.norm(res_g[eq_t_dof_idx])
        res_s_norm = np.linalg.norm(res_g[eq_s_dof_idx])
        res_xw_v_norm = np.linalg.norm(res_g[eq_xw_v_dof_idx])
        res_xw_l_norm = np.linalg.norm(res_g[eq_xw_l_dof_idx])
        res_xs_v_norm = np.linalg.norm(res_g[eq_xs_v_dof_idx])
        res_xs_l_norm = np.linalg.norm(res_g[eq_xs_l_dof_idx])
        secondary_residuals = [res_t_norm, res_s_norm, res_xw_v_norm, res_xw_l_norm,
                               res_xs_v_norm, res_xs_l_norm]
        secondary_converged_states = np.array(secondary_residuals) < res_tol

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

        par_points = np.array((x0[z_dof_idx], x0[h_dof_idx], x0[p_dof_idx])).T
        self.vtk_sampler.sample_at(par_points)
        t_k = self.vtk_sampler.sampled_could.point_data['Temperature']
        s_k = self.vtk_sampler.sampled_could.point_data['S_v']
        Xw_v_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xv']
        Xw_l_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xl']
        Xs_v_k = self.vtk_sampler.sampled_could.point_data['Xv']
        Xs_l_k = self.vtk_sampler.sampled_could.point_data['Xl']

        res_t = x0[t_dof_idx] - t_k
        res_s = x0[s_dof_idx] - s_k
        res_Xw_v = x0[xw_v_dof_idx] - Xw_v_k
        res_Xw_l = x0[xw_l_dof_idx] - Xw_l_k
        res_Xs_v = x0[xs_v_dof_idx] - Xs_v_k
        res_Xs_l = x0[xs_l_dof_idx] - Xs_l_k

        residues = [res_t, res_s, res_Xw_v, res_Xw_l, res_Xs_v, res_Xs_l]
        dofs_idx = [eq_t_dof_idx, eq_s_dof_idx, eq_xw_v_dof_idx, eq_xw_l_dof_idx, eq_xs_v_dof_idx,
                    eq_xs_l_dof_idx]
        # update deltas
        for k_field, conv_state in enumerate(secondary_converged_states):
            if conv_state:
                continue
            residue = residues[k_field]
            dof_idx = dofs_idx[k_field]
            delta_x[dof_idx] = residue
        te = time.time()
        print("Elapsed time for postprocessing secondary increments: ", te - tb)

    def rectify_secondary_fields(self, delta_x):

        tb = time.time()
        x0 = self.equation_system.get_variable_values(iterate_index=0)
        p_dof_idx = self.equation_system.dofs_of(['pressure'])
        z_dof_idx = self.equation_system.dofs_of(['z_NaCl'])
        h_dof_idx = self.equation_system.dofs_of(['enthalpy'])

        p_k = delta_x[p_dof_idx] + x0[p_dof_idx]
        z_k = delta_x[z_dof_idx] + x0[z_dof_idx]
        h_k = delta_x[h_dof_idx] + x0[h_dof_idx]
        par_points = np.array((z_k, h_k, p_k)).T
        self.vtk_sampler.sample_at(par_points)
        t_k = self.vtk_sampler.sampled_could.point_data['Temperature']
        s_k = self.vtk_sampler.sampled_could.point_data['S_v']
        Xw_v_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xv']
        Xw_l_k = 1.0 - self.vtk_sampler.sampled_could.point_data['Xl']
        Xs_v_k = self.vtk_sampler.sampled_could.point_data['Xv']
        Xs_l_k = self.vtk_sampler.sampled_could.point_data['Xl']

        secondary_x_k = np.concatenate([t_k,s_k,Xw_v_k,Xw_l_k,Xs_v_k,Xs_l_k])

        te = time.time()
        print("Elapsed time for postprocessing secondary increments: ", te - tb)
        return secondary_x_k

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, float, bool, bool]:
        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = bool(np.any(np.isnan(nonlinear_increment)))
            converged: bool = not diverged
            residual_norm: float = np.nan if diverged else 0.0
            nonlinear_increment_norm: float = np.nan if diverged else 0.0
        else:
            # First a simple check for nan values.
            if np.any(np.isnan(nonlinear_increment)):
                # If the solution contains nan values, we have diverged.
                return np.nan, np.nan, False, True

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )
            # Residual based norm
            eq_idx_map = self.equation_system.assembled_equation_indices
            eq_p_dof_idx = eq_idx_map['pressure_equation']
            eq_z_dof_idx = eq_idx_map['mass_balance_equation_NaCl']
            eq_h_dof_idx = eq_idx_map['total_energy_balance']
            eq_t_dof_idx = eq_idx_map['elimination_of_temperature_on_grids_[0]']
            eq_s_dof_idx = eq_idx_map['elimination_of_s_gas_on_grids_[0]']
            eq_xw_v_dof_idx = eq_idx_map['elimination_of_x_H2O_gas_on_grids_[0]']
            eq_xw_l_dof_idx = eq_idx_map['elimination_of_x_H2O_liq_on_grids_[0]']
            eq_xs_v_dof_idx = eq_idx_map['elimination_of_x_NaCl_liq_on_grids_[0]']
            eq_xs_l_dof_idx = eq_idx_map['elimination_of_x_NaCl_gas_on_grids_[0]']

            # mass system
            eq_mass_idx = np.concatenate([eq_p_dof_idx, eq_z_dof_idx])

            # energy system
            eq_energy_idx = np.concatenate([eq_h_dof_idx])

            # temperature system
            eq_temperature_idx = np.concatenate([eq_t_dof_idx])

            # fractions system
            eq_fractions_idx = np.concatenate(
                [eq_s_dof_idx, eq_xw_v_dof_idx, eq_xw_l_dof_idx, eq_xs_v_dof_idx,
                 eq_xs_l_dof_idx])

            residual_norm = np.linalg.norm(residual)
            res_norm_mass = np.linalg.norm(residual[eq_mass_idx])
            res_norm_energy= np.linalg.norm(residual[eq_energy_idx])
            res_norm_temperature = np.linalg.norm(residual[eq_temperature_idx])
            res_norm_fractions = np.linalg.norm(residual[eq_fractions_idx])

            # Check convergence requiring both the increment and residual to be small.
            converged_inc = nonlinear_increment_norm < nl_params["nl_convergence_tol"]

            converged_res_mass = res_norm_mass < nl_params["nl_convergence_mass_tol_res"]
            converged_res_energy = res_norm_energy < nl_params["nl_convergence_energy_tol_res"]
            converged_res_temperature = res_norm_temperature < nl_params["nl_convergence_temperature_tol_res"]
            converged_res_fractions = res_norm_fractions < nl_params[
                "nl_convergence_fractions_tol_res"]
            converged_res = converged_res_mass and converged_res_energy and converged_res_temperature and converged_res_fractions
            converged = converged_inc and converged_res
            diverged = False

            if converged:
                separator = "-" * 40
                message = f"{separator}\n Solution procedure has successfully converged.\n{separator}"
                print(message)
                print("Overall residual norm: ", np.linalg.norm(residual))
                print("Mass residual norm: ", res_norm_mass)
                print("Energy residual norm: ", res_norm_energy)
                print("Temperature residual norm: ", res_norm_temperature)
                print("Fractions residual norm: ", res_norm_fractions)


        # Log the errors (here increments and residuals)
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm, residual_norm
        )

        return residual_norm, nonlinear_increment_norm, converged, diverged

    def update_secondary_quantities(self) -> None:

        # the dependencies for this model are all paremetrized with the same triplet
        # X = (pressure, enthalpy, overall_composition)
        # collect all dependencies at once
        self.vtk_sampler.mutex_state = False
        grid_id_to_dependencies = {}
        for _, expr, func, domains, _ in self._constitutive_eliminations.values():
            for g in domains:
                X = [x([g]).value(self.equation_system) for x in expr._dependencies]
                grid_id_to_dependencies[g.id] = X
        P, H, Z = grid_id_to_dependencies[0]
        par_points = np.array((Z, H, P)).T
        self.vtk_sampler.sample_at(par_points)
        self.vtk_sampler.mutex_state = True
        self.update_all_constitutive_expressions()
        self.update_thermodynamic_properties_of_phases()
        self.vtk_sampler.mutex_state = False

    def update_all_constitutive_expressions(self) -> None:
        ni = self.iterate_indices.size
        for _, expr, func, domains, _ in self._constitutive_eliminations.values():
            for g in domains:
                X = [x([g]).value(self.equation_system) for x in expr._dependencies]

                vals, diffs = func(*X)

                expr.progress_iterate_values_on_grid(vals, g, depth=ni)
                # NOTE with depth=0, no shift in iterate sense is performed
                expr.progress_iterate_derivatives_on_grid(diffs, g)

    def update_thermodynamic_properties_of_phases(self) -> None:

        subdomains = self.mdg.subdomains()
        for phase in self.fluid_mixture.phases:
            dep_vals = [
                d(subdomains).value(self.equation_system)
                for d in self.dependencies_of_phase_properties(phase)
            ]
            state = phase.compute_properties(*dep_vals)
            # Set current iterate indices of values and derivatives
            update_phase_properties(phase, state)


# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 2
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name_phz = (
    file_name_prefix + "XHP_l" + str(parametric_space_ref_level) + "_modified_low_salt_content.vtk"
)
file_name_ptz = (
    file_name_prefix + "XTP_l" + str(parametric_space_ref_level) + "_modified.vtk"
)

constant_extended_fields = ['S_v', 'S_l', 'S_h', 'Xl', 'Xv']
brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.constant_extended_fields = constant_extended_fields
brine_sampler_phz.conversion_factors = (1.0, 1.0e3, 10.0)  # (z,h,p)
model.vtk_sampler = brine_sampler_phz

brine_sampler_ptz = VTKSampler(file_name_ptz)
brine_sampler_ptz.constant_extended_fields = constant_extended_fields
brine_sampler_ptz.conversion_factors = (1.0, 1.0, 10.0)  # (z,t,p)
brine_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
model.vtk_sampler_ptz = brine_sampler_ptz

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

P_proj, H_proj, T_proj, S_proj = model.load_and_project_reference_data()

z_proj = (1.0e-3) * np.ones_like(S_proj)
par_points = np.array((z_proj, H_proj, P_proj)).T
model.vtk_sampler.sample_at(par_points)
H_vtk = model.vtk_sampler.sampled_could.point_data['H'] * 1.0e-6
T_vtk = model.vtk_sampler.sampled_could.point_data['Temperature']
S_vtk = model.vtk_sampler.sampled_could.point_data['S_l']

# xdata = np.array((H_proj,P_proj)).T
# ds_data = np.linalg.norm(xdata - xdata[0],axis = 1)
# plt.plot(ds_data, S_proj, label='Saturation along parametric space')

def draw_and_save_comparison(T_proj,T_vtk,S_proj,S_vtk,H_proj,H_vtk):
    # plot the data
    file_prefix = 'verification_low_salt_content/'
    figure_data = {
        'T': (file_prefix + 'temperature_at_2000_years.png', 'T - Fig. 6A P. WEIS (2014)', 'T - VTKsample + GEOMAR'),
        'S': (file_prefix + 'liquid_saturation_at_2000_years.png', 's_l - Fig. 6B P. WEIS (2014)', 's_l - VTKsample + GEOMAR'),
        'H': (file_prefix + 'enthalpy_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'H - VTKsample + GEOMAR'),
    }
    fields_data = {
        'T': (T_proj,T_vtk),
        'S': (S_proj,S_vtk),
        'H': (H_proj,H_vtk),
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

draw_and_save_comparison(T_proj,T_vtk,S_proj,S_vtk,H_proj,H_vtk)

# project solution as initial guess
# x = model.equation_system.get_variable_values(iterate_index=0).copy()
# delta_x = model.increment_from_projected_solution()
# x_k = x + delta_x
# model.equation_system.set_variable_values(values=x_k, iterate_index=0)

# assert False
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
    file_prefix = 'verification_low_salt_content/'
    figure_data = {
        'T': (file_prefix + 'numeric_temperature_at_2000_years.png', 'T - Fig. 6A P. WEIS (2014)', 'T - numeric + GEOMAR'),
        'S': (file_prefix +'numeric_liquid_saturation_at_2000_years.png', 's_l - Fig. 6B P. WEIS (2014)', 's_l - numeric + GEOMAR'),
        'H': (file_prefix +'numeric_enthalpy_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'H - numeric '),
        'P': (file_prefix +'numeric_pressure_at_2000_years.png', 'H - Fig. 6A P. WEIS (2014)', 'P - numeric '),
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