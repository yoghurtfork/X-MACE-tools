#!/usr/bin/env python3
"""
Runs one SHARC MACE ensemble 
Reads in input JSON and defaults 
Outputs a folder of trajectories

Input JSON requires initconds file, so before running this,
prepare initconds using initconds_scripts/
"""

import argparse
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

# This is used to initialise the rng that generates rngseed values for each trajectory
RNG_SEED = 42

def merge_config(defaults, overrides):
    """Merges settings in input JSON and defaults recursively"""
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def parse_initconds(path: Path) -> tuple[float, list[tuple[list[str], list[int]]]]:
    """
    Parses the initconds file
    eref -> used as SHARC ezero
    blocks -> coords, velocs, and initial states for each geometry
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    natom = int(next(line.split()[1] for line in lines if line.startswith("Natom")))
    eref = float(next(line.split()[1] for line in lines if line.startswith("Eref")))

    blocks = []
    cursor = 0

    while cursor < len(lines):
        if not lines[cursor].startswith("Index"):
            cursor += 1
            continue
        if cursor + 1 >= len(lines) or lines[cursor + 1].strip() != "Atoms":
            raise ValueError(f"{lines[cursor]}: expected an Atoms line")
        atoms = lines[cursor + 2 : cursor + 2 + natom]
        if len(atoms) != natom:
            raise ValueError(f"{lines[cursor]}: expected {natom} atom lines")
        state_header = cursor + 2 + natom
        if state_header >= len(lines) or lines[state_header].strip() != "States":
            raise ValueError(f"{lines[cursor]}: expected a States line")
        selected_states = []
        cursor = state_header + 1
        while cursor < len(lines) and not lines[cursor].startswith("Ekin"):
            fields = lines[cursor].split()
            if fields and fields[-1] == "True":
                selected_states.append(int(fields[0]))
            cursor += 1
        blocks.append((atoms, selected_states))
    return eref, blocks


def write_inputs(trajectory, atoms, state, rngseed, eref, molecule, dynamics, hopping, output):
    """Writes geom, veloc, input files for one trajectory"""
    geom, veloc = [], []
    for atom_line in atoms:
        fields = atom_line.split()
        if len(fields) < 9:
            raise ValueError(f"malformed atom record: {atom_line}")
        geom.append("%2s %5.1f %12.8f %12.8f %12.8f %12.8f" %
                    (fields[0], *(float(value) for value in fields[1:6])))
        veloc.append("% 12.8f % 12.8f % 12.8f" % tuple(float(v) for v in fields[-3:]))
    (trajectory / "geom").write_text("\n".join(geom) + "\n", encoding="utf-8")
    (trajectory / "veloc").write_text("\n".join(veloc) + "\n", encoding="utf-8")

    nstates = " ".join(map(str, molecule["n_states"]))
    actstates = " ".join(map(str, molecule["active_states"]))
    charge = " ".join(map(str, molecule["charge"]))
    input_lines = [
        "printlevel 2", 'geomfile "geom"', "veloc external", 'velocfile "veloc"',
        f"nstates {nstates}", f"actstates {actstates}", f"state {state} mch",
        f"coeff {molecule['wavefunc_coeffs']}", f"rngseed {rngseed}", f"charge {charge}",
        f"ezero {eref:.10f}", f"tmax {dynamics['duration_fs']}",
        f"stepsize {dynamics['stepsize_fs']}", f"nsubsteps {dynamics['n_substeps']}",
        f"surf {hopping['surface_representation']}", f"coupling {hopping['coupling']}",
        f"ekincorrect {hopping['kinetic_energy_correction']}",
        f"reflect_frustrated {hopping['reflect_frustrated_hops']}",
        f"decoherence_scheme {hopping['decoherence_scheme']}",
        f"decoherence_param {hopping['decoherence_param']}",
        f"hopping_procedure {hopping['hopping_procedure']}",
        f"output_format {output['format']}", f"output_dat_steps {output['write_every_n_steps']}",
    ]
    if hopping["gradient_correction"]:
        input_lines.append("gradcorrect")
    if hopping["calc_grads_for_all_states"]:
        input_lines.append("grad_all")
    if hopping["calc_nacs_for_all_states"]:
        input_lines.append("nac_all")
    if not hopping["spin_orbit_coupling"]:
        input_lines.append("nospinorbit")
    if output["write_grads"]:
        input_lines.append("write_grad")
    if output["write_nacs"]:
        input_lines.append("write_nacdr")
    (trajectory / "input").write_text("\n".join(input_lines) + "\n", encoding="utf-8")


def main():
    # Read the settings and defaults, merge their configs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON configuration file")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    defaults_path = Path(__file__).resolve().parent / "input" / "defaults.json"
    if not input_path.is_file():
        parser.error(f"configuration file does not exist: {input_path}")
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    overrides = json.loads(input_path.read_text(encoding="utf-8"))
    config = merge_config(defaults, overrides)

    # Resolve experiment paths relative to the input JSON
    initconds_path = (input_path.parent / Path(config["initconds_file"]).expanduser()).resolve()
    output_dir = (input_path.parent / Path(config["output_dir"]).expanduser()).resolve()
    mace = config["mace"]
    model_file = (input_path.parent / Path(mace["model_file"]).expanduser()).resolve()
    if not initconds_path.is_file():
        parser.error(f"initconds_file does not exist: {initconds_path}")
    if not model_file.is_file():
        parser.error(f"mace.model_file does not exist: {model_file}")

    # User should have set environment variable SHARC to the SHARC bin directory
    sharc_bin = Path(os.environ["SHARC"]).expanduser().resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(sharc_bin.parent / "lib") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment.setdefault("NUMBA_DISABLE_JIT", "1")

    # Read initconds and create MACE.template and MACE.resources, which are shared by all trajectories
    try:
        eref, blocks = parse_initconds(initconds_path)
    except (StopIteration, ValueError) as error:
        parser.error(f"invalid initconds_file: {error}")
    output_dir.mkdir(parents=True, exist_ok=True)
    qm_shared = output_dir / "QM_shared"
    qm_shared.mkdir(parents=True, exist_ok=True)
    (qm_shared / "MACE.template").write_text(
        f"model_file {model_file}\ncutoff {mace['cutoff']}\n"
        f"properties {' '.join(mace['properties'])}\n"
        f"energy_unit {mace['energy_unit']}\ndistance_unit {mace['distance_unit']}\n"
        f"device {mace['device']}\npaddingstates {mace['paddingstates']}\n"
        + (f"head {mace['head']}\n" if mace["head"] is not None else ""),
        encoding="utf-8",
    )
    (qm_shared / "MACE.resources").write_text("", encoding="utf-8")

    # Prepare one trajectory folder (with geom, veloc, input files) per selected excited singlet state
    molecule, dynamics, hopping, output = config["molecule"], config["dynamics"], config["surface_hopping"], config["output"]
    rng, counters, trajectories = random.Random(RNG_SEED), {}, []
    for atoms, selected_states in blocks:
        for state in selected_states:
            if state < 2:
                continue
            singlet = state - 1
            counters[singlet] = counters.get(singlet, 0) + 1
            trajectory = output_dir / f"Singlet_{singlet}" / f"TRAJ_{counters[singlet]:05d}"
            trajectory.mkdir(parents=True, exist_ok=True)
            rngseed = rng.randint(-999999, 999999) or 1
            write_inputs(trajectory, atoms, state, rngseed, eref, molecule, dynamics, hopping, output)
            (trajectory / "QM").symlink_to("../../QM_shared", target_is_directory=True)
            trajectories.append(trajectory)
    if not trajectories:
        parser.error("no selected excited singlet states found in initconds_file")

    # Run trajectories sequentially
    try:
        for index, trajectory in enumerate(trajectories, start=1):
            print(f"[{index}/{len(trajectories)}] Running {trajectory.relative_to(output_dir)}")
            with (trajectory / "driver.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, str(sharc_bin / "driver.py"), "-i", "mace", "input"],
                    cwd=trajectory, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f"SHARC failed for {trajectory}\nSee {trajectory / 'driver.log'}", file=sys.stderr)
                return result.returncode
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    print(f"Completed {len(trajectories)} trajectories in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
