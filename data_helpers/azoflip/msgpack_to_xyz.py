"""
Read an azoflip MessagePack, 
pick out one species (by InChIKey) and write S0/S1 data to xyz
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import msgpack
from ase.data import chemical_symbols


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
DEBYE_TO_E_ANGSTROM = 0.2081943344


def get_pair_property(
    first_state: dict,
    second_state: dict,
    key: str,
    *,
    antisymmetric: bool = False,
) -> list | None:
    """Find a pair property (NACs, transition dipoles) by checking the dictionaries of both the states"""
    # absolutestate is the index of the state when all the states are ordered by energy
    # starts at 0
    first_index = first_state.get("absolutestate")
    second_index = second_state.get("absolutestate")
    if first_index is None or second_index is None:
        return None

    # try to find the pair property in the dictionary of the first state
    first_values = first_state.get(key) or {}
    value = first_values.get(str(second_index), first_values.get(second_index))
    if value is not None:
        return value

    # try to find the pair property in the dictionary of the second state
    second_values = second_state.get(key) or {}
    value = second_values.get(str(first_index), second_values.get(first_index))
    if value is None:
        return None
    if antisymmetric:
        return [[-component for component in vector] for vector in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete S0/S1 energies, forces, NACs, permanent dipoles, "
            "and transition dipoles for one AzoFlip species."
        )
    )
    parser.add_argument("input_path", type=Path, help="AzoFlip MessagePack file")
    parser.add_argument(
        "--inchikey", required=True, help="InChIKey of the species to export"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output xyz file name"
    )
    arguments = parser.parse_args()

    if not arguments.input_path.is_file():
        parser.error(f"input file does not exist: {arguments.input_path}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    force_conversion = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    nac_conversion = 1.0 / BOHR_TO_ANGSTROM

    with arguments.input_path.open("rb") as packed_file, arguments.output.open(
        "w", encoding="utf-8"
    ) as xyz_file:
        for geometries in msgpack.Unpacker(packed_file, strict_map_key=False):
            for geometry_id, geometry in geometries.items():
                counts["read"] += 1
                if geometry.get("species", {}).get("inchikey") != arguments.inchikey:
                    continue
                counts["matched"] += 1

                coordinates = geometry.get("xyz")
                if not coordinates:
                    counts["missing or malformed coordinates"] += 1
                    continue
                try:
                    atoms = [
                        (
                            chemical_symbols[int(row[0])],
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        )
                        for row in coordinates
                    ]
                except (IndexError, TypeError, ValueError):
                    counts["missing or malformed coordinates"] += 1
                    continue
                if any(not symbol for symbol, *_ in atoms):
                    counts["missing or malformed coordinates"] += 1
                    continue

                properties = geometry.get("props") or {}
                excited_states = properties.get("excitedstates") or []
                if not excited_states:
                    counts["missing S1"] += 1
                    continue
                # azoflip stores the S0 data directly in props
                # and other states data under excitedstates in props
                s0, s1 = properties, excited_states[0]

                # get energy
                s0_energy, s1_energy = s0.get("totalenergy"), s1.get("energy")
                if s0_energy is None or s1_energy is None:
                    counts["missing energy"] += 1
                    continue

                # get forces
                s0_forces, s1_forces = s0.get("forces"), s1.get("forces")
                if (
                    not isinstance(s0_forces, (list, tuple))
                    or not isinstance(s1_forces, (list, tuple))
                    or len(s0_forces) != len(atoms)
                    or len(s1_forces) != len(atoms)
                    or any(
                        not isinstance(vector, (list, tuple))
                        or len(vector) != 3
                        or any(component is None for component in vector)
                        for vector in [*s0_forces, *s1_forces]
                    )
                ):
                    counts["missing or malformed forces"] += 1
                    continue

                # get nacs
                # deriv_nacv_etf is the derivative form of the NACV computed with electronic translation factors
                nac = get_pair_property(
                    s0, s1, "deriv_nacv_etf", antisymmetric=True
                )
                if (
                    not isinstance(nac, (list, tuple))
                    or len(nac) != len(atoms)
                    or any(
                        not isinstance(vector, (list, tuple))
                        or len(vector) != 3
                        or any(component is None for component in vector)
                        for vector in nac
                    )
                ):
                    counts["missing or malformed NAC"] += 1
                    continue

                # get dip_perm
                permanent_dipoles = []
                for state in (s0, s1):
                    multipoles = state.get("multipoles") or []
                    dipole = multipoles[0] if multipoles else None
                    if not isinstance(dipole, dict) or any(
                        dipole.get(axis) is None for axis in ("X", "Y", "Z")
                    ):
                        break
                    permanent_dipoles.append(
                        [float(dipole[axis]) for axis in ("X", "Y", "Z")]
                    )
                if len(permanent_dipoles) != 2:
                    counts["missing or malformed permanent dipole"] += 1
                    continue

                # get dip_trans
                transition_dipole = get_pair_property(s0, s1, "trans_dipole")
                if (
                    not isinstance(transition_dipole, (list, tuple))
                    or len(transition_dipole) != 3
                    or any(component is None for component in transition_dipole)
                ):
                    counts["missing or malformed transition dipole"] += 1
                    continue

                # unit conversions
                energies = [[float(s0_energy) * HARTREE_TO_EV, float(s1_energy) * HARTREE_TO_EV]]
                forces = [
                    [
                        [float(component) * force_conversion for component in f0],
                        [float(component) * force_conversion for component in f1],
                    ]
                    for f0, f1 in zip(s0_forces, s1_forces)
                ]
                nacs = [
                    [[float(component) * nac_conversion for component in vector]]
                    for vector in nac
                ]
                dip_perm = [
                    [component * DEBYE_TO_E_ANGSTROM for component in vector]
                    for vector in permanent_dipoles
                ]
                dip_trans = [[float(component) * DEBYE_TO_E_ANGSTROM for component in transition_dipole]]

                # write extended xyz
                metadata = [
                    "Properties=species:S:1:pos:R:3",
                    f"geometry_id={geometry_id}",
                    f"inchikey={arguments.inchikey}",
                ]
                for name, value in (
                    ("REF_energy", energies),
                    ("REF_forces", forces),
                    ("REF_nacs", nacs),
                    ("REF_dip_perm", dip_perm),
                    ("REF_dip_trans", dip_trans),
                ):
                    payload = json.dumps(value, separators=(",", ":"), allow_nan=False)
                    metadata.append(f'{name}="_JSON {payload}"')

                xyz_file.write(f"{len(atoms)}\n")
                xyz_file.write(" ".join(metadata) + "\n")
                for symbol, x, y, z in atoms:
                    xyz_file.write(f"{symbol:<2} {x: .8f} {y: .8f} {z: .8f}\n")
                counts["written"] += 1

    print(f"Read {counts['read']} geometries from {arguments.input_path}")
    print(f"Matched InChIKey {arguments.inchikey}: {counts['matched']}")
    print(f"Wrote {counts['written']} geometries to {arguments.output}")
    print(f"Skipped incomplete geometries: {counts['matched'] - counts['written']}")
    for reason in (
        "missing or malformed coordinates",
        "missing S1",
        "missing energy",
        "missing or malformed forces",
        "missing or malformed NAC",
        "missing or malformed permanent dipole",
        "missing or malformed transition dipole",
    ):
        if counts[reason]:
            print(f"  {reason}: {counts[reason]}")


if __name__ == "__main__":
    main()
