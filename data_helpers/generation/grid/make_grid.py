"""
Generate a grid from a reference geometry
The grid iterates over bond length, dihedral, 
(optional, useful for azobenzene) side group angles, side group twists
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from data_helpers.generation.grid.make_geometry import make_geometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an XYZ grid over a bond length, dihedral, (optional) side group "
            "angles, and (optional) side group twists."
        )
    )
    parser.add_argument("reference", type=Path, help="Reference geometry XYZ file")
    parser.add_argument(
        "--atom-indices", nargs=4, type=int, required=True, metavar=("A", "B", "C", "D"),
        help="One-based A-B-C-D indices; B-C is the central bond, A-B-C-D is the dihedral",
    )
    parser.add_argument("--bond-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"), help="B-C bond length range in angstrom")
    parser.add_argument("--bond-interval", type=float, required=True, help="Bond length interval in angstrom")
    parser.add_argument("--dihedral-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"), help="A-B-C-D dihedral range in degrees")
    parser.add_argument("--dihedral-interval", type=float, required=True, help="Dihedral interval in degrees")
    parser.add_argument("--angle-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="B-C-D angle range in degrees")
    parser.add_argument("--angle-interval", type=float, help="Angle interval in degrees")
    parser.add_argument(
        "--angle-symmetric", type=str.lower, choices=("true", "false"), default="false",
        help="Set both A-B-C and B-C-D angles when true; only B-C-D when false",
    )
    parser.add_argument("--twist-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="D group twist range in degrees")
    parser.add_argument("--twist-interval", type=float, help="Twist interval in degrees")
    parser.add_argument(
        "--twist-symmetric", type=str.lower, choices=("true", "false"), default="false",
        help="Twist both A and D groups when true; only the D group when false",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output XYZ file name")
    arguments = parser.parse_args()

    if (arguments.angle_range is None) != (arguments.angle_interval is None):
        parser.error("--angle-range and --angle-interval must be specified together")
    if (arguments.twist_range is None) != (arguments.twist_interval is None):
        parser.error("--twist-range and --twist-interval must be specified together")

    reference = read(arguments.reference, index=":")[0]
    natoms = len(reference)
    atom_indices = tuple(index - 1 for index in arguments.atom_indices) # convert to 0-based indices

    # construct the grid, both bounds are inclusive
    bond_values = np.arange(arguments.bond_range[0], arguments.bond_range[1] + arguments.bond_interval / 2, arguments.bond_interval)
    dihedral_values = np.arange(arguments.dihedral_range[0], arguments.dihedral_range[1] + arguments.dihedral_interval / 2, arguments.dihedral_interval)
    angle_values = [None] if arguments.angle_range is None else np.arange(arguments.angle_range[0], arguments.angle_range[1] + arguments.angle_interval / 2, arguments.angle_interval)
    twist_values = [None] if arguments.twist_range is None else np.arange(arguments.twist_range[0], arguments.twist_range[1] + arguments.twist_interval / 2, arguments.twist_interval)
    angle_symmetric = arguments.angle_symmetric == "true"
    twist_symmetric = arguments.twist_symmetric == "true"

    # create geometries from reference geom
    geometries = []
    for bond_length in bond_values:
        for angle in angle_values:
            for twist in twist_values:
                for dihedral in dihedral_values:
                    try:
                        geometry = make_geometry(
                            reference,
                            atom_indices,
                            bond_length,
                            dihedral,
                            angle,
                            twist,
                            angle_symmetric,
                            twist_symmetric,
                        )
                    except ValueError as error:
                        parser.error(str(error))
                    geometry.info = {"bond_length": float(bond_length)}
                    if angle is not None:
                        geometry.info["angle"] = float(angle)
                    geometry.info["dihedral"] = float(dihedral)
                    if twist is not None:
                        geometry.info["twist"] = float(twist)
                    geometries.append(geometry)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.output, geometries, format="extxyz")
    print(f"Read reference geometry with {natoms} atoms from {arguments.reference}")
    print(f"Bond length values: {len(bond_values)}")
    print(f"Dihedral values: {len(dihedral_values)}")
    print(f"Angle values: {len(angle_values)}")
    print(f"Twist values: {len(twist_values)}")
    print(f"Wrote grid with {len(geometries)} geometries to {arguments.output}")


if __name__ == "__main__":
    main()
