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
from pydft_qmmm.common import Results

if TYPE_CHECKING:
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.common import Components
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
            boundary_atoms = list[tuple[int,list[int]]],
            distance = int
    ) -> None:
        self._boundary_atoms = boundary_atoms
        self._direct_pairs = []
        self.distance = distance # TEMPORARY
        self.fictitious = []
        for pair in self._boundary_atoms:
            self._direct_pairs.append([pair[0], pair[1][0]])

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
        # Grab the QM and MM calculators interfaces so we can do them separately
        for calc in calculator.calculators:
            if isinstance(calc, InterfaceCalculator):
                if isinstance(calc.interface, QMInterface):
                    self.qm_interface = calc.interface
                    self.qm_calculator = calc
                elif isinstance(calc.interface, MMInterface):
                    self.mm_interface = calc.interface
                    self.mm_calculator = calc
        # Force calculation sequence to do MM, then QM
        self.calculation_sequence = dict()
        self.calculation_sequence[f"{self.mm_calculator.name}_{0}"] = self.mm_calculator
        self.calculation_sequence[f"{self.qm_calculator.name}_{1}"] = self.qm_calculator

        ## Create arrays of original and shifted charges
        # Get original charges
        original_charges = self.system.charges
        # Prepare array of "shifted charges" to use later
        shifted_charges = list(original_charges)
        for i, b_pair in enumerate(self._boundary_atoms):
            q_0 = shifted_charges[b_pair[1][0]]
            shifted_charges[b_pair[1][0]] = 0 # zero out M1 charge
            n = len(b_pair[1]) - 1.0 # number connected to M1
            for j in range(int(n)):
                # redistribute charges
                shifted_charges[b_pair[1][j+1]] += q_0/n
        self.charges = [shifted_charges, original_charges] # put shifted first for MM calc
        print(f"ORIGINAL CHARGES: {self.charges[1]}")
        print(f"SHIFTED CHARGES : {self.charges[0]}")


        calculator.calculate = self._modify_calculate(
            calculator.calculate,
        )

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

            self.fictitious = self.generate_fictitious()
            self.qm_interface.set_fictitious(self.fictitious)

            # Stripped from the stock calculate method in composite_calculator.py
            energy = 0.
            forces = np.zeros(self.system.forces.shape)
            components: Components = dict() # figure out what this means
            for i, (name, calc) in enumerate(
                    self.calculation_sequence.items(),
            ):
#                results = calc.calculate()
#                energy += results.energy
#                forces += results.forces
                calc_energy, calc_forces, calc_components = self.get_forces(name, calc)
                energy += calc_energy
                forces += calc_forces
                components[name] = calc_energy
                components["."*(i+1)] = calc_components
                self.qm_interface.update_charges(self.charges[i])
            results = Results(energy, forces, components)
            return results
        return inner
    
    def get_forces(self, name, calc):
        results = calc.calculate()
        energy = results.energy
        forces = np.zeros(self.system.forces.shape)
        if isinstance(calc.interface, QMInterface):
            forces = results.forces[:len(self.system.positions), :]
            forces_fictitious = results.forces[len(self.system.positions):, :]
            forces += self.distribute_forces(forces_fictitious)
        else:
            forces = results.forces
        return (energy, forces, results.components)
    
    def distribute_forces(self, fictitious_forces):
        distributed = np.zeros(self.system.forces.shape)
        for i, pair in enumerate(self._direct_pairs):
            n = self.system.positions[pair[1]] - self.system.positions[pair[0]]
            g = np.linalg.norm(self.fictitious[i][0] - self.system.positions[pair[0]]) / np.linalg.norm(n)
            n = n/np.linalg.norm(n)

            distributed[pair[1]] = -g * np.dot(fictitious_forces[i], n)*n + g*fictitious_forces[i]
            distributed[pair[0]] = g * np.dot(fictitious_forces[i], n)*n + (1-g)*fictitious_forces[i]
        return distributed

    def generate_fictitious(self):
        # Get list of H atoms to give to Psi4
        fictitious = []
        for pair in self._direct_pairs:
            pos = self.system.positions[pair[1]] - self.system.positions[pair[0]]
            pos = self.system.positions[pair[0]] + self.distance * pos/np.linalg.norm(pos)
            fictitious.append([pos, "H"]) # just add H atoms; could potentially add support for other kinds of link atoms
        return fictitious
