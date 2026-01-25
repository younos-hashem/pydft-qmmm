"""Plugin for computing region I-region II bonded interactions with the link-atom method
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import openmm

from pydft_qmmm.calculators import PotentialCalculator
from pydft_qmmm.interfaces import QMInterface, MMInterface
from pydft_qmmm.calculators.composite_calculator import CompositeCalculatorPlugin
from pydft_qmmm.calculators.calculator import Results

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
            boundary_atoms: list[tuple[tuple[int,int], tuple[int,...]]],
            distance: float,
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
        self.calculators = [calc for calc in calculator.calculators]
        self.system = calculator.system
        # Grab the QM potential so we can access it to change fictitious atoms
        for calc in calculator.calculators:
            if isinstance(calc, PotentialCalculator):
                if isinstance(calc.potential, QMInterface):
                    self.qm_potential = calc.potential
                elif isinstance(calc.potential, MMInterface):
                    self.mm_potential = calc.potential

        ## Create arrays of original and shifted charges
        # Get original charges
        original_charges = self.system.charges.base.copy()
        # Prepare array of "shifted charges" to use later
        shifted_charges = original_charges.copy()
        for i, b_pair in enumerate(self._boundary_atoms):
            q_0 = shifted_charges[b_pair[0][1]]
            shifted_charges[b_pair[0][1]] = 0 # zero out M1 charge
            n = len(b_pair[1])*1. # number connected to M1
            for j in range(int(n)):
                # redistribute charges
                shifted_charges[b_pair[1][j]] += q_0/n
        self.qm_potential.update_charges(shifted_charges) # set new charges

        # populate atom information dictionary
        for pair in self._boundary_atoms:
            self.atoms[pair[0][0]] = self.get_atom_information(pair[0][0])
            self.atoms[pair[0][1]] = self.get_atom_information(pair[0][1])
            for mm_atom in pair[1]:
                self.atoms[mm_atom] = self.get_atom_information(mm_atom)
        # add harmonic bonds and angles
        self._openmm_context = self.mm_potential.base_context
        self.add_harmonic_bonds()
        self.add_harmonic_angles()
        prior_state = self._openmm_context.getState(getPositions=True)
        prior_positions = prior_state.getPositions()
        self._openmm_context.reinitialize()
        self._openmm_context.setPositions(prior_positions)

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
            self.generate_fictitious()

            # Stripped from the stock calculate method in composite_calculator.py
            energy = 0.
            forces = np.zeros(self.system.forces.shape)
            components: Components = dict()
            for i, calculator in enumerate(self.calculators):
                # Calculate the energy, forces, and components.
                calc_results = self.get_results(calculator)
                energy += calc_results.energy
                forces += calc_results.forces
                # Determine a unique name for the calculator.
                name = calculator.name
                suffix = "0"
                while name in components:
                    suffix = str(int(suffix) + 1)
                    name = calculator.name + suffix
                # Assign the components appropriately.
                components[name] = calc_results.energy
                components["."*(i + 1)] = calc_results.components
            results = Results(energy, forces, components)
            return results
        return inner
    
    def get_results(self, calc: PotentialCalculator) -> Results:
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
        if isinstance(calc.potential, QMInterface):
            forces = results.forces[:len(self.system.positions), :]
            forces_fictitious = results.forces[len(self.system.positions):, :]
            forces += self.distribute_forces(forces_fictitious)
        else:
            forces = results.forces
        return Results(energy, forces, results.components)
    
    def distribute_forces(self, fictitious_forces: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def generate_fictitious(self) -> None:
        """Add all fictitious atoms to the Psi4 potential.
        """
        for pair in self._direct_pairs:
            pos = self.system.positions[pair[1]] - self.system.positions[pair[0]]
            pos = self.system.positions[pair[0]] + self.distance * pos/np.linalg.norm(pos)
            atom = {
                "position": pos,
                "element": "H",
                "label": "",
                "ghost": False,
            }
            self.qm_potential.add_fictitious_atom(atom)
    
    def get_atom_information(self, index: int) -> dict:
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
        element = self.system.elements[index]
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
            "name": name,
            "element": element,
            "residue_name": res_name,
            "type": atom_type,
            "class": atom_class,
        }

    def add_harmonic_bonds(self):
        """Add harmonic bond interactions across the QM-MM boundary
        """
        openmm_system = self._openmm_context.getSystem()
        harmonic_bond_forces = [
            force for force in openmm_system.getForces()
            if isinstance(force, openmm.HarmonicBondForce)
        ]
        hbondforce = harmonic_bond_forces[0] # just grab the first one
        for pair in self._direct_pairs:
            atom1 = self.atoms[pair[0]]
            atom2 = self.atoms[pair[1]]
            for tree in self.ffs:
                root = tree.getroot()
                bonds = root.find("HarmonicBondForce")
                for child in bonds:
                    add = False
                    # bonds can be specified between classes or types
                    if "class1" in child.attrib:
                        if (atom1["class"] in child.attrib.values()
                        and atom2["class"] in child.attrib.values()):
                            add = True
                    elif "type1" in child.attrib:
                        if (atom1["type"] in child.attrib.values()
                        and atom2["type"] in child.attrib.values()):
                            add = True
                    if add:
                        hbondforce.addBond(atom1["index"], atom2["index"],
                                            float(child.attrib["length"]),
                                            float(child.attrib["k"]))
                        
    def add_harmonic_angles(self):
        """Add harmonic angle interactions across the QM-MM boundary
        """
        openmm_system = self._openmm_context.getSystem()
        harmonic_angle_forces = [
            force for force in openmm_system.getForces()
            if isinstance(force, openmm.HarmonicAngleForce)
        ]
        hangleforce = harmonic_angle_forces[0] # just grab the first one
        triplets = []
        for crossing in self._boundary_atoms:
            for m2 in crossing[1]:
                triplets.append([crossing[0][0], crossing[0][1], m2])
        for triplet in triplets:
            atom1 = self.atoms[triplet[0]]
            atom2 = self.atoms[triplet[1]]
            atom3 = self.atoms[triplet[2]]
            for tree in self.ffs:
                root = tree.getroot()
                angles = root.find("HarmonicAngleForce")
                for child in angles:
                    add = False
                    # angles can be specified with classes or types
                    if "class1" in child.attrib:
                        if (atom1["class"] in child.attrib.values()
                        and atom2["class"] in child.attrib.values()
                        and atom3["class"] in child.attrib.values()):
                            add = True
                    elif "type1" in child.attrib:
                        if (atom1["type"] in child.attrib.values()
                        and atom2["type"] in child.attrib.values()
                        and atom3["type"] in child.attrib.values()):
                            add = True
                    if add:
                        hangleforce.addAngle(atom1["index"],
                                             atom2["index"],
                                             atom3["index"],
                                             float(child.attrib["angle"]),
                                             float(child.attrib["k"]))

