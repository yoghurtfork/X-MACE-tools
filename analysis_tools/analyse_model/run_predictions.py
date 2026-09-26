"""Run X-MACE predictions on an xyz file and write PRED_energy, PRED_forces, PRED_nacs to the file in-place"""

import argparse
from pathlib import Path

import ase.io
import torch
from mace.data.atom_data_loader import AtomDataLoaderBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="XYZ input file")
    parser.add_argument("model", type=Path, help="X-MACE model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compute-force", action='store_true', default=False)
    parser.add_argument("--compute-nacs", action='store_true', default=False)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    atoms_list = ase.io.read(input_path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    # load model
    model = torch.load(model_path, map_location=args.device, weights_only=False)
    model = model.to(args.device)
    model.eval()
    if args.compute_nacs and not getattr(model, "compute_nacs", False):
        raise ValueError("The loaded model is unable to compute NACs.")

    # build data loader
    loader_builder = AtomDataLoaderBuilder(cutoff=float(model.r_max))
    data_loader = loader_builder.load(atoms_list, batch_size=args.batch_size, shuffle=False)

    # run predictions
    energies = []
    forces = []
    nacs = []
    for batch in data_loader:
        batch = batch.to(args.device)
        output = model(
            batch.to_dict(), 
            training=False, 
            compute_force=args.compute_force
            )
        energies.extend(output["energy"].detach().cpu().numpy())
        # pointer helps to split the forces/nacs by geometry
        pointers = batch.ptr.detach().cpu().numpy() 
        if args.compute_force:
            batch_forces = output["forces"].detach().cpu().numpy()
            forces.extend(batch_forces[pointers[i]:pointers[i + 1]] for i in range(len(pointers) - 1))
        if args.compute_nacs:
            batch_nacs = output["nacs"].detach().cpu().numpy()
            nacs.extend(batch_nacs[pointers[i]:pointers[i + 1]] for i in range(len(pointers) - 1))

    # write to file
    for index, atoms in enumerate(atoms_list):
        atoms.info["PRED_energy"] = energies[index]
        if args.compute_force:
            atoms.info["PRED_forces"] = forces[index]
        if args.compute_nacs:
            atoms.info["PRED_nacs"] = nacs[index]

    temporary_path = input_path.with_suffix(input_path.suffix + ".tmp")
    ase.io.write(temporary_path, atoms_list, format="extxyz")
    temporary_path.replace(input_path)

    written = ["PRED_energy"]
    if args.compute_force:
        written.append("PRED_forces")
    if args.compute_nacs:
        written.append("PRED_nacs")
    print(
        f"Predicted {len(atoms_list)} geometries with {model_path} on {args.device}. "
        f"Wrote {', '.join(written)} to {input_path}."
    )


if __name__ == "__main__":
    main()
