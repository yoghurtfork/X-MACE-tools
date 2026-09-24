"""
Get descriptor vectors for each geometry
calculate_descriptors(list of geometries, descriptor, kwargs) is the main function
that returns list[tuple[Atoms, np.ndarray]]
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from dscribe.core import system as dscribe_system
from dscribe.descriptors import ACSF, MBTR, SOAP


SOAP_CONFIG = {
    "periodic": False,
    "r_cut": 5.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.3,
}

ACSF_CONFIG = {
    "periodic": False,
    "r_cut": 5.0,
    "g2_params": [[1.0, shift] for shift in (0.0, 1.0, 2.0, 3.0, 4.0)],
    "g4_params": [
        [1.0, eta, zeta]
        for eta in (1.0, 2.0, 4.0)
        for zeta in (1.0, -1.0)
    ],
}

MBTR_DISTANCE_CONFIG = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0.0, "max": 1.1, "sigma": 0.02, "n": 75},
    "weighting": {"function": "unity"},
    "normalization": "l2",
    "periodic": False,
}

MBTR_ANGLE_CONFIG = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0.0, "max": 180.0, "sigma": 3.0, "n": 90},
    "weighting": {"function": "unity"},
    "normalization": "l2",
    "periodic": False,
}


def _patch_dscribe_system_init() -> None:
    """Patch to keep DScribe compatible with ASE versions that added velocities"""
    if getattr(dscribe_system.System, "_xmace_keyword_init", False):
        return

    def patched_init(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        wyckoff_positions=None,
        equivalent_atoms=None,
    ):
        super(dscribe_system.System, self).__init__(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
        )
        self.wyckoff_positions = wyckoff_positions
        self.equivalent_atoms = equivalent_atoms
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None

    dscribe_system.System.__init__ = patched_init
    dscribe_system.System._xmace_keyword_init = True


_patch_dscribe_system_init()


def _species(atoms_list: Sequence[Atoms]) -> list[str]:
    """Returns all the elements present in the dataset (needed for SOAP, ACSF, MBTR)"""
    return sorted(
        {
            symbol
            for atoms in atoms_list
            for symbol in atoms.get_chemical_symbols()
        }
    )


def bond_lengths(
    atoms_list: Sequence[Atoms], bond_pairs: Sequence[tuple[int, int]]
) -> list[np.ndarray]:
    """Return lengths of all bonds in bond_pairs (one-based atom indices)"""
    zero_based_pairs = [(first - 1, second - 1) for first, second in bond_pairs]
    return [
        np.asarray(
            [atoms.get_distance(first, second) for first, second in zero_based_pairs],
            dtype=float,
        )
        for atoms in atoms_list
    ]


def pairwise_distances(atoms_list: Sequence[Atoms]) -> list[np.ndarray]:
    """Return distances between all atoms in the geometry, in upper-triangle order"""
    vectors = []
    for atoms in atoms_list:
        upper_triangle = np.triu_indices(len(atoms), k=1)
        vectors.append(
            np.asarray(atoms.get_all_distances(mic=False)[upper_triangle], dtype=float)
        )
    return vectors


def energies(atoms_list: Sequence[Atoms]) -> list[np.ndarray]:
    """Return each geometry's ``REF_energy`` vector"""
    return [
        np.asarray(atoms.info["REF_energy"], dtype=float).reshape(-1)
        for atoms in atoms_list
    ]


def soap(atoms_list: Sequence[Atoms]) -> list[np.ndarray]:
    """Return per-atom SOAP descriptors concatenated into one vector"""
    descriptor = SOAP(species=_species(atoms_list), **SOAP_CONFIG)
    return [
        np.asarray(descriptor.create(atoms), dtype=float).reshape(-1)
        for atoms in atoms_list
    ]


def acsf(atoms_list: Sequence[Atoms]) -> list[np.ndarray]:
    """Return per-atom ACSF descriptors concatenated into one vector"""
    descriptor = ACSF(species=_species(atoms_list), **ACSF_CONFIG)
    return [
        np.asarray(descriptor.create(atoms), dtype=float).reshape(-1)
        for atoms in atoms_list
    ]


def mbtr(atoms_list: Sequence[Atoms]) -> list[np.ndarray]:
    """Return distance and angle MBTR descriptors concatenated into one vector"""
    species = _species(atoms_list)
    distance_descriptor = MBTR(species=species, **MBTR_DISTANCE_CONFIG)
    angle_descriptor = MBTR(species=species, **MBTR_ANGLE_CONFIG)
    return [
        np.concatenate(
            (
                np.asarray(distance_descriptor.create(atoms), dtype=float).reshape(-1),
                np.asarray(angle_descriptor.create(atoms), dtype=float).reshape(-1),
            )
        )
        for atoms in atoms_list
    ]

DescriptorFunction = Callable[..., list[np.ndarray]]
DESCRIPTOR_REGISTRY: dict[str, DescriptorFunction] = {
    "bond_lengths": bond_lengths,
    "pairwise_distances": pairwise_distances,
    "energies": energies,
    "soap": soap,
    "acsf": acsf,
    "mbtr": mbtr,
}

def calculate_descriptors(
    atoms_list: Sequence[Atoms], descriptor: str, **kwargs: Any
) -> list[tuple[Atoms, np.ndarray]]:
    """Associate every input geometry with its descriptor vector"""
    vectors = DESCRIPTOR_REGISTRY[descriptor](atoms_list, **kwargs)
    return [
        (atoms, np.asarray(vector, dtype=float).reshape(-1))
        for atoms, vector in zip(atoms_list, vectors, strict=True)
    ]
