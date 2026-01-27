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
    ) -> None:
        self._boundary_atoms = boundary_atoms
        self.distance = distance
        self.fictitious = []
        self._direct_pairs = [pairs[0] for pairs in self._boundary_atoms]

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
        for calc in self.calculators:
            if isinstance(calc, PotentialCalculator):
                if isinstance(calc.potential, QMInterface):
                    self.qm_potential = calc.potential
                elif isinstance(calc.potential, MMInterface):
                    self.mm_potential = calc.potential
                    self.mm_calculator = calc

        # Set system and OpenMM system/context and get QM atom set
        self.system = calculator.system
        self.omm_context: openmm.Context = self.mm_potential.base_context
        self.omm_system: openmm.System = self.omm_context.getSystem()
        self.atoms = self.system.select("subsystem I")

        # Update bond exclusions
        self.exclude_harmonic_angles()
        self.exclude_torsion()
        base_state = self.omm_context.getState(positions=True)
        omm_pos = base_state.getPositions()
        self.omm_context.reinitialize()
        self.omm_context.setPositions(omm_pos)

        ## Create arrays of original and shifted charges
        # Get original charges
        original_charges = self.system.charges.base.copy()
        # Prepare array of "shifted charges" to use in Psi4
        shifted_charges = original_charges.copy()
        for i, b_pair in enumerate(self._boundary_atoms):
            q_0 = shifted_charges[b_pair[0][1]]
            shifted_charges[b_pair[0][1]] = 0 # zero out M1 charge
            n = len(b_pair[1])*1. # number connected to M1
            for j in range(int(n)):
                # redistribute charges
                shifted_charges[b_pair[1][j]] += q_0/n
        self.qm_potential.update_charges(shifted_charges) # set new charges


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
    
    def exclude_harmonic_angles(self) -> None:
        """Remove harmonic angle interactions in which the central
            atom is a QM atom.
        """
        harmonic_angle_forces = [
            force for force in self.omm_system.getForces()
            if isinstance(force, openmm.HarmonicAngleForce)
        ]
        for force in harmonic_angle_forces:
            for i in range(force.getNumAngles()):
                *p, a, k = force.getAngleParameters(i)
                if p[1] in self.atoms:
                    k *= 0
                    force.setAngleParameters(i, *p, a, k)
    
    def exclude_torsion(self):
        """Remove torsion interactions where necessary. For proper
            torsion, interactions where atom 2 or 3 is an MM atom are
            retained. For improper torsion, interactions in which the
            central atom (1) is MM are retained."""
        harmonic_bond_forces = []
        torsion_forces = []
        for force in self.omm_system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                harmonic_bond_forces.append(force)
            elif isinstance(force, (openmm.PeriodicTorsionForce,
                                    openmm.RBTorsionForce,)):
                torsion_forces.append(force)
        
        bonds = set()
        for hbf in harmonic_bond_forces:
            for i in range(hbf.getNumBonds()):
                *p, d, k = hbf.getBondParameters(i)
                bonds.add(tuple(sorted(p)))
        for force in torsion_forces:
            if isinstance(force, openmm.RBTorsionForce):
                rb = True
            else:
                rb = False
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, *o = force.getTorsionParameters(i)
                if set([
                    tuple(sorted((p1, p2))),
                    tuple(sorted((p2, p3))),
                    tuple(sorted((p3, p4))),
                ]) <= bonds: # proper
                    if set([p2, p3]) <= self.atoms:
                        if rb:
                            o = [0]*6
                        else:
                            o[-1] = 0.0
                elif set([
                    tuple(sorted((p3, p1))),
                    tuple(sorted((p3, p2))),
                    tuple(sorted((p3, p4))),
                ]) <= bonds: # improper
                    if p3 in self.atoms:
                        if rb:
                            o = [0]*6
                        else:
                            o[-1] = 0.0
                else:
                    raise ValueError(f"Could not resolve torsion type"
                        f" (proper or improper) of torsion {i} in"
                        f" force  {force}")
                    # Improper torsion implementation can vary by
                    # force field. In OpenMM, the central atom should
                    # be first in the XML but will be third when
                    # getting parameters. See:
                    # https://github.com/ParmEd/ParmEd/issues/881
                force.setTorsionParameters(i, p1, p2, p3, p4, *o)