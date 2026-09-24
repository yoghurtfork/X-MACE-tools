"""Generate a geometry from bond length, dihedral, angle, twist"""

from ase.neighborlist import NeighborList, natural_cutoffs


def fragment_after_cut(adjacency, start, cut):
    """
    Separates a molecule into two by returning 
    the fragment reachable from start after cutting a bond
    """
    first, second = cut
    fragment = {start}
    pending = [start]
    while pending:
        atom = pending.pop()
        for neighbor in adjacency[atom]:
            if {atom, neighbor} == {first, second} or neighbor in fragment:
                continue
            fragment.add(neighbor)
            pending.append(neighbor)
    return sorted(fragment)


def make_geometry(
    reference_atoms,
    atom_indices,
    bond_length,
    dihedral,
    angle=None,
    twist=None,
    angle_symmetric=False,
    twist_symmetric=False,
):
    """Create a geometry from a reference structure and internal coordinates"""
    geometry = reference_atoms.copy()
    atom_a, atom_b, atom_c, atom_d = atom_indices

    # find neighbours
    neighbors = NeighborList(
        natural_cutoffs(reference_atoms, mult=1.2),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(reference_atoms)
    adjacency = [
        set(neighbors.get_neighbors(index)[0]) for index in range(len(reference_atoms))
    ]
    consecutive_pairs = tuple(zip(atom_indices, atom_indices[1:]))
    if any(second not in adjacency[first] for first, second in consecutive_pairs):
        raise ValueError("atom indices must identify three consecutive bonds")

    # find the stationary and moving fragments (the two sides of bond B-C)
    stationary_fragment = fragment_after_cut(adjacency, atom_b, (atom_b, atom_c))
    moving_fragment = fragment_after_cut(adjacency, atom_c, (atom_b, atom_c))
    if atom_c in stationary_fragment or atom_b in moving_fragment:
        raise ValueError("the central bond is in a ring and cannot be changed rigidly")

    # find the terminal twisting fragments (A group and D group)
    left_twist_fragment = fragment_after_cut(adjacency, atom_a, (atom_a, atom_b))
    right_twist_fragment = fragment_after_cut(adjacency, atom_d, (atom_c, atom_d))
    if atom_b in left_twist_fragment or atom_c in right_twist_fragment:
        raise ValueError("a terminal twist bond is in a ring and cannot be rotated rigidly")

    # set bond length
    geometry.set_distance(atom_b, atom_c, float(bond_length), fix=0.0, indices=moving_fragment)

    # set ABC and BCD angles
    if angle is not None:
        if angle_symmetric:
            geometry.set_angle(atom_c, atom_b, atom_a, float(angle), indices=stationary_fragment)
        geometry.set_angle(atom_b, atom_c, atom_d, float(angle), indices=moving_fragment)
    
    # set dihedral
    geometry.set_dihedral(*atom_indices, float(dihedral), indices=moving_fragment)

    # set A group and D group twists
    if twist is not None:
        if twist_symmetric:
            left_fragment = geometry[left_twist_fragment]
            left_fragment.rotate(
                float(twist),
                v=geometry.positions[atom_a] - geometry.positions[atom_b],
                center=geometry.positions[atom_b],
            )
            geometry.positions[left_twist_fragment] = left_fragment.positions
        right_fragment = geometry[right_twist_fragment]
        right_fragment.rotate(
            float(twist),
            v=geometry.positions[atom_d] - geometry.positions[atom_c],
            center=geometry.positions[atom_c],
        )
        geometry.positions[right_twist_fragment] = right_fragment.positions

    return geometry
