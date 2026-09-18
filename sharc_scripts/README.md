# SHARC SCRIPTS

SHARC itself runs from a separate program, but the scripts here help to prepare the folders required by SHARC and launch ensembles from configuration files (input JSONs)

Using input JSONs makes it easier to prep and run many reproducible SHARC jobs

### Notes on the input JSONs

- All file paths are resolved relative to the input JSONs

- All the configurable keys are listed in the respective `defaults.json`

- If a key is excluded from the input JSON, the script will just use the default

## Initconds scripts

`initconds_from_xyz.py` converts an extended xyz file into a SHARC initial conditions file

### Usage

1. Prepare an input JSON file

    Minimally, the input JSON must specify `xyz`

    If `output` is omitted, the output filename defaults to `initconds.excited`

2. Run

    ```bash
    python initconds_from_xyz.py input.json
    ```

### Notes on the input keys

- `xyz`: the xyz file of initial geometries, must contain minimally atom coordinates and `REF_energy`

- `ref_geom`: 0-based index of the geometry SHARC takes reference energy, `Eref`, from

- `remove_ref_geom`: if `true`, the ref geom is not included among the trajectory initial conditions

    SHARC subtracts `Eref` from all energies
    
    If you want the energy outputs to be physically meaningful/comparable across runs, use a relaxed geometry as `ref_geom` and set `remove_ref_geom` to `true`

- `excite_to`: `"S1"` or `"S2"`

- `velocity_mode`: decides how initial velocities are set
    
    Can be `zero` (all velocities set to zero) or `maxwell_boltzmann` (needs `temp_K` and `seed`)

## SHARC MACE scripts

`run_sharc_mace.py` preps and runs one ensemble of SHARC MACE trajectories

### Usage

1. Install SHARC and set the environment variable `SHARC` to the installation `bin`

2. Copy `sharc_mace_scripts/SHARC_MACE.py` into `bin`

    We've edited this, this is not the same as `SHARC_MACE.py` in the original X-MACE repo

3. Prepare an input JSON file

    Minimally specify `output_dir`, `initconds_file`, `model_file`

4. Run

    ```bash
    python run_sharc_mace.py input.json
    ```

### Notes on the input keys
- `head`: which MACE model head SHARC uses

    If your model was trained with a single head, leave this alone

    If your model was trained with multiheadeded training, SHARC will use the highest fidelity (highest index) head by default

    This is useful if you want to use a single multiheaded model to run low fidelity and high fidelity SHARC

- `gpus`: GPU indices available to the ensemble

    The script will assign different trajectories to different GPUs, one trajectory per GPU

## SHARC Molcas scripts

`run_sharc_molcas.py` preps and runs one ensemble of SHARC OpenMolcas trajectories

### Usage

1. Install SHARC and set the environment variable `SHARC` to the installation `bin`

2. Install OpenMolcas and set the environment variable `MOLCAS` to the installation `bin`, which should contain `pymolcas`

3. Prepare an input JSON file

    Minimally specify `output_dir`, `initconds_file`, `initial_orbitals_file`

4. Run

    ```bash
    python run_sharc_molcas.py input.json
    ```

### Notes on the input keys
- `initial_orbitals_file`: the initial RasOrb wavefunction file

    Examples can be found in `input/initial_orbitals/`

    To create a new one, run OpenMolcas single-point calculation on the relaxed geometry using OpenMolcas's `molcas_input.py`

- `total_cpus`: CPU budget for concurrently running trajectories

- `total_memory_mb`: memory budget for concurrently running trajectories

- `n_cpu_per_traj`: number of CPUs allocated to one trajectory

- `memory_mb_per_traj`: memory allocated to one trajectory

    The number of simultaneously-running trajectories is 

    ```python
    max_concurrent = min(
        execution["total_cpus"] // execution["n_cpu_per_traj"],
        execution["total_memory_mb"] // execution["memory_mb_per_traj"],
    )
    ```

    If an ensemble uses less than `total_memory_mb`, the unused memory remains available to the system

## Queue manager

`run_sharc_queue.py` runs SHARC MACE and Molcas ensembles sequentially in CLI argument order

Each runner parallelizes its own trajectories, the queue manager doesn't run multiple ensembles concurrently

### Usage

1. Prepare input JSONs for SHARC MACE and SHARC molcas runs

2. Run

    ```bash
    python run_sharc_queue.py \
            mace job1.json job2.json \
            molcas job3.json \
            mace job4.json
    ```

This runs

1. MACE `job1.json`
2. MACE `job2.json`
3. Molcas `job3.json`
4. MACE `job4.json`

The `mace`/`molcas` keywords act as mode switches, ensure they are placed before the correct input JSONs

If a job fails, the queue manager continues with the remaining jobs