"""This module implements a multiphase-multicomponent flow model with phase change
using the Soereide model to define the fluid mixture

References:
    [1] Ingolf Søreide, Curtis H. Whitson,
        Peng-Robinson predictions for hydrocarbons, CO2, N2, and H2 S with pure water
        and NaCI brine,
        Fluid Phase Equilibria,
        Volume 77,
        1992,
        https://doi.org/10.1016/0378-3812(92)85105-H

"""
from __future__ import annotations
from typing import Sequence

import porepy as pp
import porepy.composite as ppc
from porepy.composite.base import Component
from porepy.composite.eos_compiler import EoSCompiler

from porepy.models.fluid_mixture_equilibrium import MixtureMixin, EquilibriumMixin
from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler


class SoereideMixture(MixtureMixin):
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2.

    """
    def get_components(self) -> Sequence[Component]:

        chems = ["H2O", "CO2"]
        species = ppc.load_species(chems)
        components = [
            ppc.peng_robinson.H2O.from_species(species[0]),
            ppc.peng_robinson.CO2.from_species(species[1]),
        ]
        return components

    def get_phase_configuration(
        self, components: Sequence[Component]
    ) -> Sequence[tuple[EoSCompiler, int, str]]:
        # This takes some time
        eos = PengRobinsonCompiler(components)
        return [(eos, 0, 'liq'), (eos, 1, 'gas')]


class GeothermalFlow(
    SoereideMixture,
    EquilibriumMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Geothermal flow using a fluid defined by the Soereide model."""
