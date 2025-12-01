"""The Psi4 software interface and potential.

This module contains the software interface for storing and manipulating
Psi4 data types and the potential using Psi4 to calculate energies
and forces.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
import psi4.core

from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import system_cache

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401
    from . import psi4_utils


@dataclass(frozen=True)
class Psi4Interface(QMInterface):
    r"""A mix-in for storing and manipulating Psi4 data types.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
        fictitious: A list of fictitious atoms present in the Psi4
            calculation but not the system at large.
        alt_charges: An alternative Subsystem II charge distribution in
            which to embed the QM subsystem.
    """
    functional: str
    charge: int
    multiplicity: int
    output_file: str
    output_interval: int
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )
    fictitious: list[dict] = field(default_factory=lambda: [])
    alt_charges: list[np.float64] = field(default_factory=lambda: [])

    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Add an electronic potential to apply before calculations.

        Args:
            potential: The electronic potential to incorporate into
                QM calculations.
        """
        self.potentials.append(potential)
        self.update_options(perturb_h=True, perturb_with="EMBPOT")
    
    def add_fictitious_atom(self, fict: dict) -> None:
        """Add a fictitious atom that exists only in Psi4.

        Args:
            fict: The fictitious atom to append, which should have four
                elements:
                position: The Cartesian coordinates of the fictitious
                    atom
                element: The atom's element
                label: An optional string label
                ghost: A boolean indicating whether to treat the atom
                    as a Psi4 ghost atom
        """
        self.fictitious.append(fict)
    
    def update_charges(self, dist: NDArray[np.float64]) -> None:
        """Update the alternative embedding charges.
        
        Args:
            dist: The new charge distribution to use.
        """
        if dist.shape != self.system.charges.shape:
            raise ValueError(
                f"Charge array of shape {dist.shape} is incompatible "
                f"with system with charge array of shape {self.system}.")
        while len(dist) > len(self.alt_charges):
            self.alt_charges.append(0.)
        for i in range(len(self.alt_charges)):
            self.alt_charges[i] = dist[i]

    @system_cache("positions", "charges", "elements", "subsystems")
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object.

        Returns:
            The Psi4 Wavefunction object, which contains the energy
            and coefficients determined through SCF.
        """
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file(self.output_file, True)
        molecule = self._generate_molecule()
        if self.potentials:
            basis_set = psi4.core.BasisSet.build(
                molecule,
                "BASIS",
                psi4.core.get_global_option("BASIS"),
            )
            grid = psi4.core.DFTGrid.build(molecule, basis_set)
            blocks = []
            for block in grid.blocks():
                x = block.x().np
                y = block.y().np
                z = block.z().np
                w = block.w().np
                blocks.append(np.stack((x, y, z, w), axis=-1))
            xyzw = np.concatenate(tuple(blocks), axis=0)
            v = np.zeros_like(xyzw[:, 0]).reshape(-1, 1)
            for potential in self.potentials:
                v += potential.compute_potential_and_derivs(
                    xyzw[:, :-1] / BOHR_PER_ANGSTROM,
                )[:,0].reshape(-1,1)
            data = np.concatenate((xyzw, v), axis=1)
            np.savetxt("EMBPOT", data, header=f"{len(data)}", comments="")
        _, wfn = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=molecule,
            external_potentials=self._generate_external_potential(),
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file("/dev/null", True)
        self.frame[0] += 1
        return wfn

    @system_cache("positions", "elements", "subsystems")
    def _generate_molecule(self) -> psi4.core.Molecule:
        """Generate the Psi4 Molecule object.

        Returns:
            The Psi4 Molecule object, which contains the positions,
            net charge, and net spin of atoms in the QM subsystem.
        """
        geometrystring = """\n"""
        atoms = sorted(self.system.select("subsystem I"))
        for atom in atoms:
            geometrystring = (
                geometrystring
                + str(self.system.elements[atom]) + str(atom) + " "
                + str(self.system.positions[atom][0]) + " "
                + str(self.system.positions[atom][1]) + " "
                + str(self.system.positions[atom][2]) + "\n"
            )
        if self.fictitious:
            ghost = [atom for atom in self.fictitious if atom["ghost"]]
            non_ghost = [atom for atom in self.fictitious if not atom["ghost"]]
            for atom in non_ghost: # put all ghosts at the end for organizational ease
                position = atom["position"]
                designation = atom["element"] + atom["label"]
                geometrystring = (
                    geometrystring
                    + designation + " "
                    + str(position[0]) + " "
                    + str(position[1]) + " "
                    + str(position[2]) + "\n"
                )
            for atom in ghost:
                position = atom["position"]
                designation = "@" + atom["element"] + atom["label"]
                geometrystring = (
                    geometrystring
                    + designation + " "
                    + str(position[0]) + " "
                    + str(position[1]) + " "
                    + str(position[2]) + "\n"
                )
        geometrystring += str(self.charge) + " "
        geometrystring += str(self.multiplicity) + "\n"
        # geometrystring += "symmetry c1\n"
        geometrystring += "noreorient\nnocom\n"
        return psi4.geometry(geometrystring)

    @system_cache("positions", "charges", "subsystems")
    def _generate_external_potential(self) -> NDArray[np.float64] | None:
        r"""Generate the data structure needed to perform embedding.

        Returns:
            The list of coordinates (:math:`\mathrm{a.u.}`) and charges
            (:math:`e`) that will be electrostatically embedded by Psi4
            during calculations.
        """
        external_potential = []
        embedding = sorted(self.system.select("subsystem II"))
        if len(self.alt_charges) > 0:
            charges = self.alt_charges
        else:
            charges = self.system.charges
        for i in embedding:
            external_potential.append(
                (
                    charges[i],
                    self.system.positions[i, 0] * BOHR_PER_ANGSTROM,
                    self.system.positions[i, 1] * BOHR_PER_ANGSTROM,
                    self.system.positions[i, 2] * BOHR_PER_ANGSTROM,
                ),
            )
        if not external_potential:
            return None
        return np.array(external_potential)
    

    def update_options(self, **kwargs: psi4_utils.Psi4Options) -> None:
        """Set additional options for Psi4.

        Args:
            kwargs: Additional options to provide to Psi4.  See
                `Psi4 options`_ for additional Psi4 options.
        """
        psi4.set_options(kwargs)
    


class Psi4Potential(Psi4Interface, AtomicPotential):
    """A potential wrapping Psi4 functionality.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.
        fictitious: Fictitious atoms, given as a list of lists, where
            each list contains each fictitious atom's position (as
            another list) and its Psi4 designation (e.g. element)

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using Psi4.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        wfn = self._generate_wavefunction()
        energy = wfn.energy() * KJMOL_PER_EH
        if self.fictitious:
            for potential in self.potentials:
                non_ghost = [fict for fict in self.fictitious if not fict["ghost"]] # ghost atoms do not contribute
                fict_coords = np.array([fict["position"] for fict in non_ghost])
                pot = potential.compute_potential_and_derivs(fict_coords)[:,0] # just the potential
                elements = list(ELEMENT_TO_MASS.keys())
                charges = [elements.index(fict["element"]) for fict in non_ghost] # get fictitious atoms' integer charges
                energy += np.dot(charges, pot * -KJMOL_PER_EH)
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Psi4.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        wfn = self._generate_wavefunction()
        forces = psi4.gradient(
            self.functional,
            ref_wfn=wfn,
        )

        # get coordinates and charges to compute potential
        n_real = len(sorted(self.system.select("subsystem I")))
        non_ghost = [fict for fict in self.fictitious if not fict["ghost"]]
        n_ghost = len(self.fictitious) - len(non_ghost)
        fict_coords = np.array([fict["position"] for fict in non_ghost])
        elements = list(ELEMENT_TO_MASS.keys())
        charges = [elements.index(fict["element"]) for fict in non_ghost]
        # get gradient from potential and add it to force
        if fict_coords.size > 0:
            for potential in self.potentials:
                zeros_pre = np.zeros((n_real, 3),dtype=np.float64)
                zeros_post= np.zeros((n_ghost, 3), dtype=np.float64)
                grad = potential.compute_potential_and_derivs(fict_coords)[:, 1:]
                pot_force = (grad.T * charges).T
                padded_force = np.concat((zeros_pre, pot_force, zeros_post)) # pad with zeros to ensure correct dimensions
                forces.add(psi4.core.Matrix.from_array(padded_force))
        forces = forces.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp = np.concatenate((np.zeros(self.system.positions.shape), np.zeros((len(self.fictitious), 3))))
        qm_indices = sorted(self.system.select("subsystem I"))
        forces_temp[qm_indices, :] = forces[:len(qm_indices), :]
        if self.fictitious:
            forces_temp[self.system.positions.shape[0]:, :] = forces[len(qm_indices):, :] # add fictitious forces at the end
        if self._generate_external_potential() is not None:
            embed_indices = sorted(self.system.select("subsystem II"))
            forces = (
                wfn.external_pot().gradient_on_charges().np
                * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            )
            forces_temp[embed_indices, :] = forces
        return forces_temp

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using Psi4.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components: dict[str, float] = {}
        wfn = self._generate_wavefunction()
        psi4_base = wfn.energy() * KJMOL_PER_EH
        external = 0.
        print("Number of fictitious atoms:", len(self.fictitious))
        if self.fictitious:
            print("fictitious check passed")
            for potential in self.potentials:
                non_ghost = [fict for fict in self.fictitious if not fict["ghost"]] # ghost atoms do not contribute
                fict_coords = np.array([fict["position"] for fict in non_ghost])
                pot = potential.compute_potential_and_derivs(fict_coords)[:,0] # just the potential
                elements = list(ELEMENT_TO_MASS.keys())
                charges = [elements.index(fict["element"]) for fict in non_ghost] # get fictitious atoms' integer charges
                external += np.dot(charges, pot * -KJMOL_PER_EH)
        components["Psi4 base energy"] = psi4_base
        components["External potential"] = external
        return components
