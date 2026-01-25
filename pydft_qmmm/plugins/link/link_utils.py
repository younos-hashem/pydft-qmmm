"""Extra functionality for the link atom plugin.
"""
from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.system import System
from pydft_qmmm.system.selection_utils import interpret
from pydft_qmmm.system.selection_utils import decompose

def auto_boundary(
        qm_indices: list[int],
        system: System,
        qm_bond_length:  np.float64,
        mm_bond_length: np.float64 = None,
) -> tuple[Any, ...]:
    r"""Automatically selects the boundary pairs and MM atoms to shift
        MM_1 charges onto.
    
    Args:
        qm_indices: Indices of the atoms in the QM system.
        system: The system to find the atoms in.
        qm_bond_length: The bond length from QM_1 to MM_1.
        mm_bond_length: Optionally, the bond length from MM_1 to
            the MM_2 atoms. These bond lengths are assumed all the
            same; for a more complicated system, enter indices
            manually. Defaults to qm_bond_length.
    """
    qm_indices = frozenset(qm_indices)
    if mm_bond_length == None:
        mm_bond_length = qm_bond_length
    
    def query(length: np.float64, index: int) -> str:
        return ("within " + str(length) + " of atom " + str(index))
    
    boundary_atoms = []
    
    for qm_atom in qm_indices:
        connected = interpret(decompose(query(qm_bond_length, qm_atom)), system) - qm_indices
        for mm_atom in connected:
            boundary_pair = [qm_atom, mm_atom]
            mm_2s = interpret(decompose(query(mm_bond_length, mm_atom)), system) - qm_indices - {mm_atom}
            boundary_atoms.append([boundary_pair, list(mm_2s)])
    
    return boundary_atoms
