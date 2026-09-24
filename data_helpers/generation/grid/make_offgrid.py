"""
Generate offgrid geometries from a reference geometry
Uses Sobol quasi-random sampling to generate geometries that cover the dimensions of
bond length, dihedral, (optional) side group angles, side group twists
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.stats import qmc

from data_helpers.generation.grid.make_geometry import make_geometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Sobol geometries over a bond length, dihedral, "
        "(optional) side group angles, and (optional) side group twists."
    )
    parser.add_argument("reference", type=Path, help="Reference geometry XYZ file")
    parser.add_argument(
        "--atom-indices", nargs=4, type=int, required=True, metavar=("A", "B", "C", "D"),
        help="One-based A-B-C-D indices; B-C is the central bond, A-B-C-D is the dihedral",
    )
    parser.add_argument("--bond-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"), help="B-C bond length range in angstrom")
    parser.add_argument("--dihedral-range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"), help="A-B-C-D dihedral range in degrees")
    parser.add_argument("--angle-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="B-C-D angle range in degrees")
    parser.add_argument(
        "--angle-symmetric", type=str.lower, choices=("true", "false"), default="false",
        help="Set both A-B-C and B-C-D angles when true; only B-C-D when false",
    )
    parser.add_argument("--twist-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="D group twist range in degrees")
    parser.add_argument(
        "--twist-symmetric", type=str.lower, choices=("true", "false"), default="false",
        help="Twist both A and D groups when true; only the D group when false",
    )
    parser.add_argument("--num-geometries", type=int, required=True, metavar="N", help="Number of geometries to generate")
    parser.add_argument("--seed", type=int, default=42, help="Sobol scramble seed (default: 42)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output XYZ file name")
    arguments = parser.parse_args()

    reference = read(arguments.reference, index=":")[0]
    natoms = len(reference)
    atom_indices = tuple(index - 1 for index in arguments.atom_indices) # convert to 0-based indices

    # create ranges for Sobol
    ranges = [arguments.bond_range, arguments.dihedral_range]
    names = ["bond", "dihedral"]
    if arguments.angle_range is not None:
        ranges.append(arguments.angle_range)
        names.append("angle")
    if arguments.twist_range is not None:
        ranges.append(arguments.twist_range)
        names.append("twist")

    # do Sobol sampling
    active_indices = [index for index, limits in enumerate(ranges) if limits[0] != limits[1]] # if MIN and MAX are equal, the value of that dimension is fixed in all generated geometries
    values = np.tile(np.array([limits[0] for limits in ranges]), (arguments.num_geometries, 1))
    if active_indices:
        power = (arguments.num_geometries - 1).bit_length()
        samples = qmc.Sobol(d=len(active_indices), scramble=True, seed=arguments.seed).random_base2(power)[: arguments.num_geometries]
        for sample_index, range_index in enumerate(active_indices):
            lower, upper = ranges[range_index]
            values[:, range_index] = lower + samples[:, sample_index] * (upper - lower)

    # create geometries from reference geom
    geometries = []
    for row in values:
        coordinates = dict(zip(names, row))
        try:
            geometry = make_geometry(
                reference,
                atom_indices,
                coordinates["bond"],
                coordinates["dihedral"],
                coordinates.get("angle"),
                coordinates.get("twist"),
                arguments.angle_symmetric == "true",
                arguments.twist_symmetric == "true",
            )
        except ValueError as error:
            parser.error(str(error))
        geometry.info = {}
        geometries.append(geometry)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.output, geometries, format="extxyz")
    print(f"Read reference geometry with {natoms} atoms from {arguments.reference}")
    print(f"Seed: {arguments.seed}")
    print(f"Bond length range: {tuple(arguments.bond_range)}")
    print(f"Dihedral range: {tuple(arguments.dihedral_range)}")
    print(f"Angle range: {tuple(arguments.angle_range) if arguments.angle_range is not None else None}")
    print(f"Twist range: {tuple(arguments.twist_range) if arguments.twist_range is not None else None}")
    print(f"Wrote {len(geometries)} geometries to {arguments.output}")

if __name__ == "__main__":
    main()
