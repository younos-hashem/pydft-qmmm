"""Plugin for computing region I-region II bonded interactions with the link-atom method
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.plugins.plugin import CompositeCalculatorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.common import Results
    import mypy_extensions
    CalculateMethod = Callable[
        [
            mypy_extensions.DefaultArg(
                bool | None,
                "return_forces", # noqa: F821
            ),
            mypy_extensions.DefaultArg(
                bool | None,
                "return_components", # noqa: F821
            ),
        ],
        Results,
    ]

class LINK(CompositeCalculatorPlugin):
    def __init__(
            self,
            boundary_atoms = list[tuple[int,list[int]]]
    ) -> None:
        self._boundary_atoms = boundary_atoms

    def modify(
        self,
        calculator: CompositeCalculator,
    ) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will
                modified by the plugin.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        calculator.calculate = self._modify_calculate(
            calculator.calculate,
        )
        # Grab the QM and MM calculators interfaces so we can do them separately
        for calc in calculator.calculators:
            if isinstance(calc, InterfaceCalculator):
                if isinstance(calc.interface, QMInterface):
                    self.qm_interface = calc.interface
                    self.qm_calculator = calc
                elif isinstance(calc.interface, MMInterface):
                    self.mm_interface = calc.interface
                    self.mm_calculator = calc

    def _modify_calculate(
            self,
            calculate: CalculateMethod,
    ) -> CalculateMethod:
        """Modify the calculate routine to use the link-atom method.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements
            the link-atom method using H atoms on the boundary
        """
        def inner(
                return_forces: bool | None = True,
                return_components: bool | None = True,
        ) -> Results:
            # Get original charges
            original_charges = system.charges
            # Prepare array of "shifted charges" to use later
            shifted_charges = original_charges
            for i, b_pair in enumerate(boundary_atoms):
                q_0 = shifted_charges[b_pair[1][0]]
                shifted_charges[b_pair[1][0]] = 0 # zero out M1 charge
                n = len(b_pair[1]) - 1.0 # number connected to M1
                if n > 0:
                    for j in range(n):
                    # redistribute charges
                    shifted_charges[b_pair[1][j+1]] += q_0/n

            # from the stock calculate method
            energy = 0.
            forces = np.zeros(system.forces.shape)
            components: Components = dict() # figure out what this means
            
            # do the MM calculation first, with the real charges
            results = mm_calculator.calculate()
            energy += results.energy
            forces += results.forces
            components[f"{mm_calculator.name}_0"] = results.energy # there is probably a more general way for this
            components["."*1] = results.components

            system.charges(shifted_charges) # shift the charges

            results = qm_calculator.calculate()
            energy += results.energy
            forces += results.forces
            components[f"{qm_calculator.name}_1"] = results.energy
            components["."*2] = results.components

            system.charges(original_charges) # re-shift the charges
            results = Results(energy, forces, components)

