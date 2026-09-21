"""
Augment geometries using normal mode sampling
Specify a range for each mode in angstrom
The magnitude of the augmentations applied will be uniformly selected from within the ranges
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from augmentation_utils import moving_fragment_after_bond_cut, remove_reference_properties


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment geometries using normal mode sampling"
    )
    parser.add_argument("input", type=Path, help="Input XYZ containing geometries")
    parser.add_argument("normal_modes", type=Path, help="XYZ file containing normal modes")
    parser.add_argument(
        "--augmentations-per-geometry", type=int, required=True, metavar="N"
    )
    parser.add_argument(
        "--mode-range", type=float, nargs=3, action="append", required=True,
        metavar=("MODE", "MIN", "MAX"),
        help="For each normal vibration mode, the zero-based index and range in angstrom",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-bond", type=int, nargs=2, metavar=("I", "J"),
        help="One-based atom indices; the bond length between these atoms will be preserved after augmentation",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args()

    # read input XYZ and normal mode XYZ, check that they match
    input_frames = read(arguments.input, index=":")
    reference = read(arguments.normal_modes)
    reference_symbols = reference.get_chemical_symbols()
    for frame_index, atoms in enumerate(input_frames):
        if atoms.get_chemical_symbols() != reference_symbols:
            parser.error(
                f"input frame {frame_index} does not match the atom order in normal mode XYZ"
            )

    # find the normal vibration modes and their ranges
    mode_ranges = []
    mode_vectors = {}
    for mode_value, lower, upper in arguments.mode_range:
        mode = int(mode_value)
        key = f"xtb2_normal_mode_displacements_{mode}"
        if key not in reference.arrays:
            parser.error(f"mode {mode} is not present in {arguments.normal_modes}")
        vector = np.asarray(reference.arrays[key], dtype=float)
        # normalise the normal mode vectors
        mode_vectors[mode] = vector / np.linalg.norm(vector)
        mode_ranges.append((mode, lower, upper))

    # find the bond to be preserved
    preserved_bond = None
    moving_fragment = None
    if arguments.preserve_bond:
        preserved_bond = tuple(index - 1 for index in arguments.preserve_bond)
        moving_fragment = moving_fragment_after_bond_cut(reference, *preserved_bond)

    # do augmentation
    rng = np.random.default_rng(arguments.seed)
    augmented_frames = []
    for source_frame_index, base_atoms in enumerate(input_frames):
        target_bond_length = (
            base_atoms.get_distance(*preserved_bond) if preserved_bond else None
        )
        for augmentation_index in range(arguments.augmentations_per_geometry):
            magnitudes = {
                mode: float(rng.uniform(lower, upper))
                for mode, lower, upper in mode_ranges
            }
            displacement = sum(
                (magnitudes[mode] * mode_vectors[mode] for mode, _, _ in mode_ranges),
                start=np.zeros_like(base_atoms.positions),
            )
            atoms = base_atoms.copy()
            atoms.positions = base_atoms.positions + displacement
            # remove properties like energy and forces which no longer describe the augmented geometries
            remove_reference_properties(atoms)
            # restore the preserved bond length
            if preserved_bond:
                atoms.set_distance(
                    *preserved_bond, target_bond_length,
                    fix=0.0, indices=moving_fragment,
                )
            atoms.info.update(
                {
                    "source_frame_index": source_frame_index,
                    "augmentation_index": augmentation_index,
                    "augmentation_method": "uniform_normal_modes",
                    **{
                        f"mode_{mode}_magnitude_angstrom": magnitude
                        for mode, magnitude in magnitudes.items()
                    },
                }
            )
            augmented_frames.append(atoms)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.output, augmented_frames, format="extxyz")
    print(f"Read {len(input_frames)} geometries from {arguments.input}")
    print(f"Used modes: {' '.join(str(mode) for mode, _, _ in mode_ranges)}")
    print(f"Wrote {len(augmented_frames)} uniform augmentations to {arguments.output}")


if __name__ == "__main__":
    main()
