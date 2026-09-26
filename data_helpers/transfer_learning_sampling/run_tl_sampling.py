"""CLI script for transfer learning sampling"""

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    script_directory = Path(__file__).resolve().parent
    sys.path = [path for path in sys.path if Path(path or ".").resolve() != script_directory]
    sys.path.insert(0, str(script_directory.parents[1]))

from ase.io import read, write

from data_helpers.transfer_learning_sampling.descriptors import (
    DESCRIPTOR_REGISTRY,
    calculate_descriptors,
)
from data_helpers.transfer_learning_sampling.selectors import (
    SELECTOR_REGISTRY,
    select_atoms,
)


def _bond_pair(value: str) -> tuple[int, int]:
    '''Validate bond_pair kwarg for bond length descriptor'''
    try:
        first, second = value.split("-", maxsplit=1)
        return int(first), int(second)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid bond pair '{value}'; expected one-based indices such as 1-2."
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select transfer learning geometries from an extended XYZ file."
    )
    parser.add_argument("input", type=Path, help="Input XYZ file")
    parser.add_argument("output", type=Path, help="Output XYZ file")
    parser.add_argument(
        "--descriptor", required=True, choices=DESCRIPTOR_REGISTRY.keys()
    )
    parser.add_argument("--selector", required=True, choices=SELECTOR_REGISTRY.keys())
    parser.add_argument("--n-samples", required=True, type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument(
        "--bond-pairs",
        nargs="+",
        type=_bond_pair,
        help="For bond_lengths descriptor: one-based atom pairs, for example 1-2 2-3",
    )
    parser.add_argument(
        "--initialize",
        type=int,
        default=0,
        help="For FPS selector: initial geometry (default: 0)",
    )
    parser.add_argument("--n-clusters", type=int, help="For K-means selector: number of clusters")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="For K-means selector: if True, weigh sample allocation by cluster size",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        help="For stratified selector: stratum fractions",
    )
    parser.add_argument(
        "--allocations",
        nargs="+",
        type=int,
        help="For stratified selector: number of samples to draw from each stratum",
    )
    return parser


def main() -> None:
    # parse arguments
    arguments = build_parser().parse_args()
    atoms_list = read(arguments.input, index=":")

    # assign descriptor kwargs
    descriptor_kwargs: dict[str, Any] = {}
    if arguments.descriptor == "bond_lengths":
        descriptor_kwargs["bond_pairs"] = arguments.bond_pairs

    # assign selector kwargs
    selector_kwargs: dict[str, Any] = {}
    if arguments.selector == "random":
        selector_kwargs["seed"] = arguments.seed
    elif arguments.selector == "fps":
        selector_kwargs["initialize"] = arguments.initialize
    elif arguments.selector == "kmeans":
        selector_kwargs.update(
            seed=arguments.seed,
            n_clusters=arguments.n_clusters,
            weighted=arguments.weighted,
        )
    elif arguments.selector == "stratified":
        selector_kwargs.update(
            fractions=arguments.fractions,
            allocations=arguments.allocations,
            seed=arguments.seed,
        )

    # get descriptor vectors
    described_atoms = calculate_descriptors(
        atoms_list, arguments.descriptor, **descriptor_kwargs
    )

    # select geometries
    selected_atoms = select_atoms(
        described_atoms,
        arguments.selector,
        arguments.n_samples,
        **selector_kwargs,
    )
    write(arguments.output, selected_atoms, format="extxyz")

    print(f"Descriptor: {arguments.descriptor}")
    print(f"Selector: {arguments.selector}")
    print(f"Wrote {arguments.n_samples} geometries to {arguments.output}")


if __name__ == "__main__":
    main()
