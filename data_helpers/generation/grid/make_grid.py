"""
Generate a grid from a reference geometry
The grid iterates over bond length, dihedral, 
(optional, useful for azobenzene) side group angles, side group twists
"""

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs


def fragment_after_cut(
    adjacency: list[set[int]], start: int, cut: tuple[int, int]
) -> list[int]:
    """After cutting a bond, find the two fragments"""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an XYZ grid over a bond length, dihedral, (optional) side group "
            "angles, and (optional) side group twists."
        )
    )
    parser.add_argument("reference", type=Path, help="Reference geometry XYZ file")
    parser.add_argument(
        "--atom-indices",
        nargs=4,
        type=int,
        required=True,
        metavar=("A", "B", "C", "D"),
        help=(
            "One-based A-B-C-D indices; B-C is the central bond, A-B-C-D is the dihedral"
        ),
    )
    parser.add_argument(
        "--bond-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="B-C bond-length range in angstrom",
    )
    parser.add_argument(
        "--bond-interval",
        type=float,
        required=True,
        help="Bond length interval in angstrom",
    )
    parser.add_argument(
        "--dihedral-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="A-B-C-D dihedral range in degrees",
    )
    parser.add_argument(
        "--dihedral-interval",
        type=float,
        required=True,
        help="Dihedral interval in degrees",
    )
    parser.add_argument(
        "--angle-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="B-C-D side group angle range in degrees",
    )
    parser.add_argument(
        "--angle-interval",
        type=float,
        required=True,
        help="Side group angle interval in degrees",
    )
    parser.add_argument(
        "--angle-symmetric",
        type=str.lower,
        choices=("true", "false"),
        required=True,
        help="Set both A-B-C and B-C-D angles when true; only B-C-D when false",
    )
    parser.add_argument(
        "--twist-range",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="D side group twist range in degrees",
    )
    parser.add_argument(
        "--twist-interval",
        type=float,
        required=True,
        help="Side group twist interval in degrees",
    )
    parser.add_argument(
        "--twist-symmetric",
        type=str.lower,
        choices=("true", "false"),
        required=True,
        help="Twist both terminal fragments when true; only the D side when false",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output XYZ file name"
    )
    arguments = parser.parse_args()

    intervals = (
        arguments.bond_interval,
        arguments.dihedral_interval,
        arguments.angle_interval,
        arguments.twist_interval,
    )
    ranges = (
        arguments.bond_range,
        arguments.dihedral_range,
        arguments.angle_range,
        arguments.twist_range,
    )

    reference_frames = read(arguments.reference, index=":")
    reference = reference_frames[0]
    natoms = len(reference)

    atom_indices = tuple(index - 1 for index in arguments.atom_indices)
    atom_a, atom_b, atom_c, atom_d = atom_indices

    # find neighbors
    neighbors = NeighborList(
        natural_cutoffs(reference, mult=1.2),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(reference)
    adjacency = [set(neighbors.get_neighbors(index)[0]) for index in range(natoms)]
    consecutive_pairs = tuple(zip(atom_indices, atom_indices[1:]))
    if any(second not in adjacency[first] for first, second in consecutive_pairs):
        parser.error("atom indices must identify three consecutive bonds")

    # cut down the central bond, identify the stationary fragment and moving/rotating fragment
    stationary_fragment = fragment_after_cut(
        adjacency, atom_b, (atom_b, atom_c)
    )
    moving_fragment = fragment_after_cut(adjacency, atom_c, (atom_b, atom_c))
    rotating_fragment = fragment_after_cut(adjacency, atom_d, (atom_b, atom_c))
    if atom_c in stationary_fragment or atom_b in moving_fragment:
        parser.error("the central bond is in a ring and cannot be changed rigidly")
    if set(moving_fragment) != set(rotating_fragment):
        parser.error(
            "the moving and dihedral-rotating fragments must be the same"
        )

    # identify the side group fragments for twisting
    left_twist_fragment = fragment_after_cut(
        adjacency, atom_a, (atom_a, atom_b)
    )
    right_twist_fragment = fragment_after_cut(
        adjacency, atom_d, (atom_c, atom_d)
    )
    if atom_b in left_twist_fragment or atom_c in right_twist_fragment:
        parser.error("a terminal twist bond is in a ring and cannot be rotated rigidly")

    # construct the grid, both bounds are inclusive
    bond_values = np.arange(
        arguments.bond_range[0],
        arguments.bond_range[1] + arguments.bond_interval / 2,
        arguments.bond_interval,
    )
    dihedral_values = np.arange(
        arguments.dihedral_range[0],
        arguments.dihedral_range[1] + arguments.dihedral_interval / 2,
        arguments.dihedral_interval,
    )
    angle_values = np.arange(
        arguments.angle_range[0],
        arguments.angle_range[1] + arguments.angle_interval / 2,
        arguments.angle_interval,
    )
    twist_values = np.arange(
        arguments.twist_range[0],
        arguments.twist_range[1] + arguments.twist_interval / 2,
        arguments.twist_interval,
    )

    angle_symmetric = arguments.angle_symmetric == "true"
    twist_symmetric = arguments.twist_symmetric == "true"

    # create geometries from reference geom
    geometries = []
    for bond_length in bond_values:
        for angle in angle_values:
            for twist in twist_values:
                for dihedral in dihedral_values:
                    geometry = Atoms(
                        symbols=reference.get_chemical_symbols(),
                        positions=reference.positions.copy(),
                        cell=reference.cell.copy(),
                        pbc=reference.pbc.copy(),
                    )
                    geometry.set_distance(
                        atom_b,
                        atom_c,
                        float(bond_length),
                        fix=0.0,
                        indices=moving_fragment,
                    )
                    if angle_symmetric:
                        geometry.set_angle(
                            atom_c,
                            atom_b,
                            atom_a,
                            float(angle),
                            indices=stationary_fragment,
                        )
                    geometry.set_angle(
                        atom_b,
                        atom_c,
                        atom_d,
                        float(angle),
                        indices=moving_fragment,
                    )
                    geometry.set_dihedral(
                        *atom_indices,
                        float(dihedral),
                        indices=moving_fragment,
                    )

                    if twist_symmetric:
                        left_fragment = geometry[left_twist_fragment]
                        left_fragment.rotate(
                            float(twist),
                            v=geometry.positions[atom_a]
                            - geometry.positions[atom_b],
                            center=geometry.positions[atom_b],
                        )
                        geometry.positions[left_twist_fragment] = (
                            left_fragment.positions
                        )
                    right_fragment = geometry[right_twist_fragment]
                    right_fragment.rotate(
                        float(twist),
                        v=geometry.positions[atom_d] - geometry.positions[atom_c],
                        center=geometry.positions[atom_c],
                    )
                    geometry.positions[right_twist_fragment] = (
                        right_fragment.positions
                    )

                    geometry.info = {
                        "bond_length": float(bond_length),
                        "angle": float(angle),
                        "dihedral": float(dihedral),
                        "twist": float(twist),
                    }
                    geometries.append(geometry)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.output, geometries, format="extxyz")
    print(f"Read reference geometry with {natoms} atoms from {arguments.reference}")
    print(f"Bond-length values: {len(bond_values)}")
    print(f"Angle values: {len(angle_values)}")
    print(f"Twist values: {len(twist_values)}")
    print(f"Dihedral values: {len(dihedral_values)}")
    print(f"Generated {len(geometries)} geometries")
    print(f"Wrote grid to {arguments.output}")


if __name__ == "__main__":
    main()
