"""Shared operations for normal mode augmentation scripts"""

from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


def moving_fragment_after_bond_cut(
    atoms: Atoms, stationary_atom: int, moving_atom: int
) -> list[int]:
    """
    Return the fragment on one side of a cut bond
    So it can be moved to restore the preserved bond length
    """
    neighbors = NeighborList(
        natural_cutoffs(atoms, mult=1.2), skin=0.0,
        self_interaction=False, bothways=True,
    )
    neighbors.update(atoms)
    adjacency = [set(neighbors.get_neighbors(i)[0]) for i in range(len(atoms))]
    fragment = {moving_atom}
    pending = [moving_atom]
    while pending:
        atom = pending.pop()
        for neighbor in adjacency[atom]:
            if {atom, neighbor} == {stationary_atom, moving_atom}:
                continue
            if neighbor not in fragment:
                fragment.add(neighbor)
                pending.append(neighbor)
    if stationary_atom in fragment:
        raise ValueError("the preserved bond is in a ring and cannot be restored rigidly")
    return sorted(fragment)


def remove_reference_properties(atoms: Atoms) -> None:
    """Remove labels that no longer describe an augmented geometry"""
    for key in list(atoms.info):
        if key.startswith("REF_"):
            del atoms.info[key]
    for key in list(atoms.arrays):
        if key.startswith("REF_"):
            del atoms.arrays[key]
