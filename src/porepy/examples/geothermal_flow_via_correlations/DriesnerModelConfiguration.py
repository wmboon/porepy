import BrineConstitutiveDescription
import numpy as np
from Geometries import SimpleGeometry as ModelGeometry

import porepy as pp
import porepy.compositional as ppc
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)


class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        sides = self.domain_boundary_sides(sd)
        facet_idx = sides.north + sides.south
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        facet_idx = np.concatenate(self.get_inlet_outlet_sides(sd))
        sides = self.domain_boundary_sides(sd)
        facet_idx = sides.east + sides.west
        return pp.BoundaryCondition(sd, facet_idx, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, outlet_idx = self.get_inlet_outlet_sides(boundary_grid)

        p_inlet = 20.0e6
        p_outlet = 18.0e6
        xc = boundary_grid.cell_centers.T
        l = 10.0

        def p_linear(xv):
            p_v = p_inlet * (1 - xv[0] / l) + p_outlet * (xv[0] / l)
            return p_v

        p = np.fromiter(map(p_linear, xc), dtype=float)
        return p
        p[inlet_idx] = p_inlet
        p[outlet_idx] = p_outlet
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        h_inlet = 2.8e6
        h_outlet = 1.5e6
        xc = boundary_grid.cell_centers.T
        l = 10.0

        def h_linear(xv):
            h_v = h_inlet * (1 - xv[0] / l) + h_outlet * (xv[0] / l)
            return h_v

        h = np.fromiter(map(h_linear, xc), dtype=float)
        h[inlet_idx] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        inlet_idx, _ = self.get_inlet_outlet_sides(boundary_grid)
        z_init = 0.05
        z_inlet = 0.05
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
        sides = self.domain_boundary_sides(boundary_grid)
        t_inlet = 600
        t_outlet = 600

        t = t_outlet * np.ones(boundary_grid.num_cells)
        t[sides.south] = t_inlet
        t[sides.north] = t_outlet
        return t

class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_inlet = 18.0e6
        p_outlet = 18.0e6
        xc = sd.cell_centers.T
        l = 10.0

        def p_linear(xv):
            p_v = p_inlet * (1 - xv[0] / l) + p_outlet * (xv[0] / l)
            return p_v

        p = np.fromiter(map(p_linear, xc), dtype=float)
        return p

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h_inlet = 1.5e6
        h_outlet = 1.5e6
        xc = sd.cell_centers.T
        l = 10.0

        def h_linear(xv):
            h_v = h_inlet * (1 - xv[0] / l) + h_outlet * (xv[0] / l)
            return h_v

        h = np.fromiter(map(h_linear, xc), dtype=float)
        return h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.05
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class SecondaryEquations(BrineConstitutiveDescription.SecondaryEquations):
    pass


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
    BrineConstitutiveDescription.FluidMixture,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        return saturation

    def temperature_function(self, primary_state: np.ndarray) -> np.ndarray:
        T_vals, _ = self.temperature_func(*primary_state)
        return T_vals

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl
