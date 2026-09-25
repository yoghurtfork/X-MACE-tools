"""
Augment geometries using thermal normal mode sampling
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from wfl.generate import normal_modes as nm

from augmentation_utils import moving_fragment_after_bond_cut, remove_reference_properties


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment geometries using thermal normal mode sampling"
    )
    parser.add_argument("input", type=Path, help="Input XYZ containing geometries")
    parser.add_argument("normal_modes", type=Path, help="XYZ file containing normal modes")
    parser.add_argument(
        "--augmentations-per-geometry", type=int, required=True, metavar="N"
    )
    parser.add_argument("--temperature", type=float, required=True, metavar="K")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", type=int, nargs="+", required=True,
        help="Zero-based indices of the normal modes you want to use",
    )
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

    # find the bond to be preserved
    preserved_bond = None
    moving_fragment = None
    if arguments.preserve_bond:
        preserved_bond = tuple(index - 1 for index in arguments.preserve_bond)
        moving_fragment = moving_fragment_after_bond_cut(reference, *preserved_bond)

    # do thermal augmentation using wfl
    np.random.seed(arguments.seed)
    augmented_frames = []
    for source_frame_index, base_atoms in enumerate(input_frames):
        normal_mode_atoms = reference.copy()
        normal_mode_atoms.positions = base_atoms.positions.copy()
        normal_mode_atoms.info["source_frame_index"] = source_frame_index
        normal_modes = nm.NormalModes(normal_mode_atoms, "xtb2_")
        samples = normal_modes.sample_normal_modes(
            sample_size=arguments.augmentations_per_geometry,
            temp=arguments.temperature,
            normal_mode_numbers=arguments.modes,
            info_to_keep=["source_frame_index"],
        )

        target_bond_length = (
            base_atoms.get_distance(*preserved_bond) if preserved_bond else None
        )
        for augmentation_index, atoms in enumerate(samples):
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
                    "augmentation_index": augmentation_index,
                    "augmentation_method": "thermal_normal_modes",
                    "temperature_kelvin": arguments.temperature,
                }
            )
            augmented_frames.append(atoms)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.output, augmented_frames, format="extxyz")
    print(f"Read {len(input_frames)} geometries from {arguments.input}")
    print(f"Used modes: {' '.join(map(str, arguments.modes))}")
    print(
        f"Wrote {len(augmented_frames)} thermal augmentations at "
        f"{arguments.temperature:g} K to {arguments.output}"
    )


if __name__ == "__main__":
    main()
