"""Filtering functions for lists of ASE Atoms objects"""

import json
from pathlib import Path

import numpy as np
import shnitsel as st
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from rdkit import Chem
from shnitsel.clean import sanity_check
from shnitsel.clean.filter_energy import EnergyFiltrationThresholds
from shnitsel.clean.filter_geo import GeometryFiltrationThresholds


def filter_clashes(
    geometries: list[Atoms],
    bonds: list[tuple[int, int]],
    clash_threshold: float = 0.75,
) -> list[Atoms]:
    """Filter geometries where unbonded atoms are too close together"""
    bonds = [(first - 1, second - 1) for first, second in bonds] # convert to 0-based indices
    bonded_pairs = {frozenset(bond) for bond in bonds}
    filtered_geometries = []

    for atoms in geometries:
        has_clash = False
        for first in range(len(atoms)):
            for second in range(first + 1, len(atoms)):
                if frozenset((first, second)) in bonded_pairs:
                    continue
                minimum_distance = clash_threshold * (
                    covalent_radii[atoms[first].number]
                    + covalent_radii[atoms[second].number]
                )
                if atoms.get_distance(first, second, mic=True) < minimum_distance:
                    has_clash = True
                    break
            if has_clash:
                break

        if not has_clash:
            filtered_geometries.append(atoms)

    return filtered_geometries


def filter_static(
    geometries: list[Atoms],
    filter_settings_path: str | Path,
    bonds: list[tuple[int, int]],
) -> list[Atoms]:
    """Apply only the geometry filters in filter settings"""
    bonds = [(first - 1, second - 1) for first, second in bonds] # convert to 0-based indices
    with Path(filter_settings_path).open(encoding="utf-8") as handle:
        settings = json.load(handle)
    bond_limits = settings["bond_length_limits_angstrom"]
    overrides = bond_limits["overrides"]
    filtered_geometries = []

    for atoms in geometries:
        passes_filter = True
        for first, second in bonds:
            first_symbol = atoms[first].symbol
            second_symbol = atoms[second].symbol
            limit = overrides.get(
                f"{first_symbol}-{second_symbol}",
                overrides.get(
                    f"{second_symbol}-{first_symbol}", # check for both C-H and H-C
                    bond_limits["all_bonds"],
                ),
            )
            if atoms.get_distance(first, second, mic=True) > limit:
                passes_filter = False
                break

        if passes_filter:
            filtered_geometries.append(atoms)

    return filtered_geometries


def filter_dynamic(
    geometries: list[Atoms],
    netcdf_path: str | Path,
    filter_settings_path: str | Path,
    bonds: list[tuple[int, int]],
    filter_method: str | float = "truncate",
) -> list[Atoms]:
    """
    Some trajectory information is stored in the original SHNITSEL NetCDF files, but not the extxyz data we are using
    Apply the filters (from filter settings file) to the NetCDF, then map the retained geometries back onto the list of Atoms objects
    """
    bonds = [(first - 1, second - 1) for first, second in bonds] # convert to 0-based indices
    with Path(filter_settings_path).open(encoding="utf-8") as handle:
        settings = json.load(handle)

    # read energy filters from filter settings file
    energy_settings = settings["energy_limits_ev"]
    energy_thresholds = EnergyFiltrationThresholds(
        {
            "epot_active_step": energy_settings["active_potential_step"],
            "epot_hop_step": energy_settings["hop_potential_step"],
            "etot_step": energy_settings["total_energy_step"],
            "etot_drift": energy_settings["total_energy_drift"],
            "ekin_step": energy_settings["kinetic_energy_step"],
        }
    )

    # read geometry filters from filter settings file
    bond_settings = settings["bond_length_limits_angstrom"]
    smarts_limits = {}
    for element_pair, limit in bond_settings["overrides"].items():
        first_symbol, second_symbol = element_pair.split("-")
        smarts_limits[
            f"[#{atomic_numbers[first_symbol]}]~[#{atomic_numbers[second_symbol]}]"
        ] = limit
    geometry_thresholds = GeometryFiltrationThresholds(smarts_limits)
    geometry_thresholds.all_bonds_threshold = bond_settings["all_bonds"]
    # disable SHNITSEL's default rule that CH and NH bonds are limited to 2 Ang
    geometry_thresholds.all_h_to_C_or_N_bonds_threshold = np.inf

    # construct an RDKit molecule for SHNITSEL tools' sanity_check()
    molecule = Chem.RWMol()
    for atom in geometries[0]:
        molecule.AddAtom(Chem.Atom(atom.symbol))
    for first, second in bonds:
        molecule.AddBond(first, second, Chem.BondType.SINGLE)

    frames = st.io.read(netcdf_path)
    frames.dataset["atXYZ"].attrs["units"] = "angstrom"

    # filter NetCDF frames
    filtered_frames = sanity_check(
        frames,
        filter_method,
        energy_thresholds=energy_thresholds,
        geometry_thresholds=geometry_thresholds,
        mol=molecule.GetMol(),
    )
    if filtered_frames is None:
        return []

    # map filter results onto the ASE Atoms list
    source_index = frames.dataset.indexes["frame"]
    retained_index = filtered_frames.dataset.indexes["frame"]
    retained_positions = source_index.get_indexer(retained_index)
    return [geometries[position] for position in retained_positions]
