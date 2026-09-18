#!/usr/bin/env python3
"""
Creates a SHARC initial conditions file from an extended XYZ file (must contain REF_energy)
Reads in input JSON and defaults
Outputs a initconds file
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import units
from ase.data import atomic_masses
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


def merge_config(defaults, overrides):
    """Merges settings in input JSON and defaults recursively"""
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def write_geom_coords(output, atoms, distance_unit: str) -> None:
    """Write the atom coordinates block which includes positions and velocities"""
    positions = atoms.get_positions()
    if distance_unit == "Ang":
        positions = positions / units.Bohr
    velocities = atoms.get_velocities() * 0.00448998 # convert ASE velocities to SHARC atomic units
    for atom, position, velocity in zip(atoms, positions, velocities):
        output.write(
            f"{atom.symbol} {atom.number:.1f} "
            f"{position[0]:.8f} {position[1]:.8f} {position[2]:.8f} "
            f"{atomic_masses[atom.number]:.8f} "
            f"{velocity[0]:.8f} {velocity[1]:.8f} {velocity[2]:.8f}\n"
        )


def write_geom_info(
    output,
    atoms,
    energies,
    eref: float,
    excited_state: int,
    energy_to_hartree: float,
) -> None:
    """Write the information block which includes states and energy information"""
    energies = np.asarray(energies, dtype=float).ravel() * energy_to_hartree
    ground_energy = energies[0]
    output.write("States\n")
    for state, energy in enumerate(energies, start=1):
        gap_ev = (energy - ground_energy) * units.Hartree
        output.write(
            f"{state:03d} {energy: 18.10f} {ground_energy: 18.10f} "
            # transition dipole components can be written as zero
            f"{0.0: 12.8f} {0.0: 12.8f} {0.0: 12.8f} {0.0: 12.8f} {0.0: 12.8f} {0.0: 12.8f} " 
            f"{gap_ev: 12.8f} "
            # oscillator strength can be written as zero
            f"{0.0: 12.8f} " 
            # writes True if this is the state we want to excite to, else False
            f"{state - 1 == excited_state}\n" 
        )
    ekin = atoms.get_kinetic_energy() / units.Hartree
    epot_harm = 0.0
    epot = ground_energy - eref
    output.write(f"Ekin      {ekin: 16.12f} a.u.\n")
    output.write(f"Epot_harm {epot_harm: 16.12f} a.u.\n")
    output.write(f"Epot      {epot: 16.12f} a.u.\n")
    output.write(f"Etot_harm {ekin: 16.12f} a.u.\n")
    output.write(f"Etot      {ekin + epot: 16.12f} a.u.\n\n\n")


def main():
    # Read the settings and defaults, merge their configs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON configuration file")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    defaults_path = Path(__file__).resolve().parent / "input" / "defaults.json"
    config = merge_config(
        json.loads(defaults_path.read_text(encoding="utf-8")),
        json.loads(input_path.read_text(encoding="utf-8")),
    )

    xyz_path = (input_path.parent / Path(config["input"]["xyz"]).expanduser()).resolve()
    frames = list(read(xyz_path, index=":"))

    # Set velocities to zeros or use maxwell distribution
    initialise = config["initialise"]
    if initialise["velocity_mode"] == "zero":
        for atoms in frames:
            atoms.set_velocities(np.zeros((len(atoms), 3)))
    elif initialise["velocity_mode"] == "maxwell_boltzmann":
        rng = np.random.default_rng(initialise["seed"])
        for atoms in frames:
            MaxwellBoltzmannDistribution(atoms, temperature_K=initialise["temp_K"], rng=rng)
            Stationary(atoms)
            ZeroRotation(atoms)

    input_config = config["input"]
    equilibrium = frames[input_config["ref_geom"]]
    energy_to_hartree = 1 / units.Hartree if input_config["energy_unit"] == "eV" else 1.0
    eref = float(np.asarray(equilibrium.info["REF_energy"]).ravel()[0]) * energy_to_hartree
    excited_state = int(initialise["excite_to"][1:])

    # Remove the ref geom if remove_ref_geom is true
    initial_conditions = [
        atoms for index, atoms in enumerate(frames)
        if not input_config["remove_ref_geom"] or index != input_config["ref_geom"]
    ]

    output_path = config["output"] or "initconds.excited"
    output_path = (input_path.parent / Path(output_path).expanduser()).resolve()

    # Write initconds file
    with output_path.open("w", encoding="utf-8") as output:
        output.write("SHARC Initial conditions file, version 4.0   <Excited>\n")
        output.write(f"Ninit     {len(initial_conditions)}\n")
        output.write(f"Natom     {len(equilibrium)}\n")
        output.write("Repr      MCH\n")
        output.write(f"Eref      {eref: 18.10f}\n")
        output.write("Eharm     0.0000000000\n")
        output.write(f"States    {len(np.asarray(equilibrium.info['REF_energy']).ravel())} 0 0\n\n\n")
        output.write("Equilibrium\n")
        write_geom_coords(output, equilibrium, input_config["distance_unit"]) # writes the equilibrium (ref) geometry
        output.write("\n\n")
        for index, atoms in enumerate(initial_conditions, start=1): # writes all the initial geometries for SHARC
            output.write(f"Index     {index}\nAtoms\n")
            write_geom_coords(output, atoms, input_config["distance_unit"])
            write_geom_info(
                output, atoms, atoms.info["REF_energy"], eref, excited_state, energy_to_hartree
            )

    print(output_path)


if __name__ == "__main__":
    main()
