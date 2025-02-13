from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import LINK

# based on case_0_api.py from pydft-qmmm/examples

# Load system first.
system = System.load("ethane.pdb")

# Generate velocities.
system.velocities = generate_velocities(
    system.masses,
    300,
    10101,
)

# Define QM Hamiltonian.
qm = QMHamiltonian(
    basis_set="def2-SVP",
    functional="PBE",
    charge=0,
    spin=1,
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    ["spce.xml", "spce_residues.xml"],
    nonbonded_method="NoCutoff",
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

# Define QM/MM Hamiltonian
total = qm[0,1,6,7] + mm[2,3,4,5] + qmmm

# Define the integrator to use.
integrator = VerletIntegrator(1)

# Define the logger.
logger = Logger("output_api/", system, decimal_places=6)

# Define plugins.
# settle = SETTLE()
# link = LINK() # not ready yet...

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    logger=logger,
    #plugins=[link],
)

simulation.run_dynamics(10)
