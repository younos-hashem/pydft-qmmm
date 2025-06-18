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

import xml.etree.ElementTree as ET

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
    """Modify calculation to implement the link-atom scheme.

    Args:
        boundary_atoms: A list of boundary atoms and MM nearest
            neighbors, where each entry is of the form
            [[Q1, M1], [M2_1, M2_2, ..., M2_n]]
        distance: Distance along Q-M axis to place H atom
        forcefield: A list of OpenMM force field XML files
    """
    def __init__(
            self,
            boundary_atoms = list[tuple[tuple[int,int], tuple[int,...]]],
            distance = float,
            forcefield = list[str]
    ) -> None:
        self._boundary_atoms = boundary_atoms
        self.distance = distance
        self.fictitious = []
        self._direct_pairs = [pairs[0] for pairs in self._boundary_atoms]
        self.ffs = []
        for file in forcefield:
            tree = ET.parse(file)
            if tree.getroot().tag == 'ForceField':
                self.ffs.append(tree)
        self.atoms = {}

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
        original_charges = self.system.charges.base
        # Prepare array of "shifted charges" to use later
        shifted_charges = original_charges.copy()
        for i, b_pair in enumerate(self._boundary_atoms):
            q_0 = shifted_charges[b_pair[0][1]]
            shifted_charges[b_pair[0][1]] = 0 # zero out M1 charge
            n = len(b_pair[1])*1. # number connected to M1
            for j in range(int(n)):
                # redistribute charges
                shifted_charges[b_pair[1][j]] += q_0/n
        self.charges = [shifted_charges, original_charges] # put shifted first for MM calc


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
            the link-atom method using H atoms on the boundary.
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
            components: Components = dict()
            for i, (name, calc) in enumerate(
                    self.calculation_sequence.items(),
            ):
                calc_energy, calc_forces, calc_components = self.get_forces(calc)
                energy += calc_energy
                forces += calc_forces
                components[name] = calc_energy
                components["."*(i+1)] = calc_components
                self.qm_interface.update_charges(self.charges[i])
            results = Results(energy, forces, components)
            return results
        return inner
    
    def get_forces(self, calc):
        """Get energy, forces, and components from a calculator, with
        forces redistributed if necessary.
        
        Args:
            calc: The calculator to get forces from.
        
        Returns:
            energy: The energy from the calculator.
            forces: If MM, the forces from the calculator. If QM, the
                forces from the calculator, redistributed according to
                distribute_forces().
            results.components: The components from the calculator.
        """
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
        """Distribute the forces from the fictitious atoms onto the real ones.

        Args:
            fictitious_forces: The forces on each fictitious atom. Given in the
                same order as the order of QM-MM boundary pairs.
        
        Returns:
            Forces on all real atoms, redistributed according to the chain rule.
        """
        distributed = np.zeros(self.system.forces.shape)
        for i, pair in enumerate(self._direct_pairs):
            n = self.system.positions[pair[1]] - self.system.positions[pair[0]]
            g = self.distance / np.linalg.norm(n)
            n = n/np.linalg.norm(n)

            distributed[pair[1]] = -g * np.dot(fictitious_forces[i], n)*n + g*fictitious_forces[i]
            distributed[pair[0]] = g * np.dot(fictitious_forces[i], n)*n + (1-g)*fictitious_forces[i]
        return distributed

    def generate_fictitious(self):
        """Generate a list of fictitious (link) atoms to give to Psi4.

        Returns:
            A list of lists containing the position vector and
            atom type for each link atom.
        """
        fictitious = []
        for pair in self._direct_pairs:
            pos = self.system.positions[pair[1]] - self.system.positions[pair[0]]
            pos = self.system.positions[pair[0]] + self.distance * pos/np.linalg.norm(pos)
            fictitious.append([pos, "H"])
        return fictitious
    
    def get_atom_information(self, index, method):
        """Get an atom's name, residue, type, and class for use in the
        MM force field.

        Args:
            index: The atom's index.
            method: "MM" or "QM", the method used for the atom.
        
        Returns:
            A dictionary with all the atom's information
        """
        name = self.system.names[index]
        res_name = self.system.residue_names[index]
        atom_type = ""
        atom_class = ""
        for tree in self.ffs:
            root = tree.getroot()
            residues = root.find("Residues")
            types = root.find("AtomTypes")
            for residue in residues:
                for atom in residue.findall("Atom"):
                    if atom.attrib["name"] == name:
                        atom_type = atom.attrib["type"]
            for type in types:
                if type.attrib["name"] == atom_type:
                    atom_class = type.attrib["class"]
        if atom_type == "" or atom_class == "":
            raise ValueError("Bad force field: atoms not found")
        return {
            "index": index,
            "method": method,
            "name": name,
            "residue_name": res_name,
            "type": atom_type,
            "class": atom_class,
        }

