from __future__ import annotations

from pydft_qmmm import System, QMHamiltonian, QMMMHamiltonian, MMHamiltonian
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.utils.constants import Subsystem


system = System.load("./data/partition/partition.pdb")

qm = QMHamiltonian(
    basis="def2-SVP",
    functional="PBE",
    charge=0,
    multiplicity=2,
    reference="uks",
)

mm = MMHamiltonian(
    forcefield=["./data/partition/partition_ff.xml", "./data/partition/partition_residues.xml"],
    nonbonded_method="CutoffNonPeriodic",
    nonbonded_cutoff=14,
)

cutoff = 14.

qmmm = QMMMHamiltonian("electrostatic", "cutoff", cutoff=cutoff, partition=None)

qm_indices = (0,)
mm_indices = tuple(range(1, len(system)))
total = qm[qm_indices] + mm[mm_indices] + qmmm

calculator = total.build_calculator(system)

def test_centroid_partition():
    # reset to default state
    system.subsystems[list(qm_indices)] = Subsystem.I
    system.subsystems[list(mm_indices)] = Subsystem.II

    centroid_partition = CentroidPartition("all", cutoff=cutoff)
    calculator.register_plugin(centroid_partition)
    centroid_partition.generate_partition()
    
    # Manually create test case
    # Specific to the PDB file we use
    test_centroid = []
    test_centroid = test_centroid + [Subsystem.I]
    test_centroid = test_centroid + 6*[Subsystem.II]
    test_centroid = test_centroid + 2*[Subsystem.III]
    
    assert list(system.subsystems) == test_centroid
