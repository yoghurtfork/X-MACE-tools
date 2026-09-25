"""Relax a molecule and generate its GFN2-xTB normal modes"""

import argparse
from pathlib import Path

from ase.io import read
from ase.optimize import BFGS
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm
from xtb.ase.calculator import XTB


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relax a reference geometry and generate its normal modes"
    )
    parser.add_argument("reference_xyz", type=Path, help="Reference geometry XYZ")
    arguments = parser.parse_args()

    molecule_name = arguments.reference_xyz.stem
    output_directory = Path(__file__).resolve().parent / "normal_vibration_modes"
    normal_modes_xyz = output_directory / f"{molecule_name}_normal_modes.xyz"
    trajectory_directory = output_directory / f"{molecule_name}_normal_modes"
    output_directory.mkdir(parents=True, exist_ok=True)

    atoms = read(arguments.reference_xyz)
    atoms.calc = XTB(method="GFN2-xTB")
    optimizer = BFGS(atoms)
    optimizer.run(fmax=1.0e-4, steps=500)

    relaxed_atoms = atoms.copy()
    relaxed_atoms.calc = None
    calculator = (XTB, [], {"method": "GFN2-xTB"})
    nm.generate_normal_modes_parallel_hessian(
        inputs=ConfigSet([relaxed_atoms]),
        outputs=OutputSpec(normal_modes_xyz, overwrite=True),
        calculator=calculator,
        prop_prefix="xtb2_",
    )

    normal_modes = nm.NormalModes(read(normal_modes_xyz), "xtb2_")
    normal_modes.view(
        prefix=f"{molecule_name}_mode",
        output_dir=trajectory_directory,
        normal_mode_numbers="all",
        temp=300,
        nimages=16,
    )
    normal_modes.summary()

    print(f"Wrote normal modes to {normal_modes_xyz}")
    print(f"Wrote {normal_modes.num_nm} normal-mode trajectories to {trajectory_directory}")



if __name__ == "__main__":
    main()
