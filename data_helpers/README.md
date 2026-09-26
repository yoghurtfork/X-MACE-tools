# DATA HELPERS

This directory contains modules/scripts that prepare molecular datasets for X-MACE workflows

## Generation

These are helpers to generate the xyz geometries used in training

### General notes

- All helpers require a reference geometry xyz; those relevant to this project are stored in `ref_geoms/`

- Definitions:

    `--atom-indices` are one-based indices that refer to the relevant atoms (eg C-N=N-C for azobenzene)

    `angle` is the side group angle (eg C-N=N angle for azobenzene)

    `angle` is the side group twist (eg the rotation of the phenyl group about the C-N bond)

    `--angle-symmetric` and `--twist-symmetric` make both left and right angles/twists the same

### `make_geometry.py` (module)

Creates a geometry from a specified bond length, dihedral, (optional) angle, and (optional) twist

### `make_grid.py` (CLI script)

Generates a grid of geometries by iterating over bond length, dihedral, (optional) angle, (optional) twist

- Example command:

    ```bash
    python make_grid.py \
    ref_geoms/butene.xyz \
    --atom-indices 1 2 5 10 \
    --bond-range 1.45 1.65 \
    --bond-interval 0.05 \
    --dihedral-range 0 180 \
    --dihedral-interval 10 \
    --angle-range 110 120 \
    --angle-interval 5 \
    --angle-symmetric true \
    --twist-range 0 90 \
    --twist-interval 5 \
    --twist-symmetric true \
    --output butene_grid.xyz
    ```

### `make_offgrid.py` (CLI script)

Uses Sobol sampling to generate 'offgrid' geometries covering the dimensions of bond length, dihedral, (optional) angle, (optional) twist

- Example command:

    ```bash
    python make_offgrid.py \
    ref_geoms/butene.xyz \
    --num-geometries 50 \
    --atom-indices 1 2 5 10 \
    --bond-range 1.45 1.65 \
    --dihedral-range 0 180 \
    --angle-range 110 120 \
    --angle-symmetric false \
    --twist-range 0 90 \
    --twist-symmetric true \
    --output butene_offgrid_.xyz
    ```

## Augmentation

Helpers to augment geometries using normal vibration modes

### General notes

- The xyz files in `normal_vibration_modes/` contain the displacement vectors for each molecule's normal modes; the augmentation helpers require these files

- The subfolders (`*_normal_modes/`) contain trajectories demonstrating each vibration mode

- `--preserve-bond` recovers a specified bond length after augmentation, eg if you want to augment a grid of geometries but keep its grid nature

### `generate_normal_modes.py` (CLI script)

Relaxes a molecule and generates its GFN2-xTB normal modes, writing outputs into `normal_vibration_modes/`

- Example command:

    ```bash
    python generate_normal_modes.py \
    substituted_azobenzene_ref_geom.xyz
    ```

### `thermal_augment.py` (CLI script)

Augment geometries by applying normal mode vectors to them according to a thermal Boltzmann distribution

- Example command:

    ```bash
    python thermal_augment.py \
    ethene_grid.xyz \
    normal_vibration_modes/ethene_normal_modes.xyz \
    --augmentations-per-geometry 10 \
    --temperature 500 \
    --seed 42 \
    --modes 14 15 16 17 \
    --preserve-bond 1 2 \
    --output ethene_grid_augmented_500K.xyz
    ```

### `uniform_augment.py` (CLI script)

Augment geometries by applying normal mode vectors to them according to a uniform distribution

- Specify a range for each mode in Angstrom; the magnitude of the vectors applied will be randomly uniformly sampled from within the ranges

- Example command:

    ```bash
    python uniform_augment.py \
    ethene_grid.xyz \
    normal_vibration_modes/ethene_normal_modes.xyz \
    --augmentations-per-geometry 10 \
    --mode-range 14 -0.05 0.05 \
    --mode-range 15 -0.1 0.1 \
    --mode-range 16 0 0.1 \
    --mode-range 17 -0.1 0 \
    --seed 42 \
    --preserve-bond 1 2 \
    --output ethene_grid_augmented_uniform.xyz
    ```

## Azoflip

### `msgpack_to_xyz.py` (CLI script)

Reads the azoflip MessagePack files and converts them to xyz files usable in this project

- Picks out one species using InChIKey

- Writes S0 and S1 information (energy in eV, forces in eV/Å, NACs in Å^-1, permanent dipoles in e\*Å, transition dipoles in e\*Å) only

- Example command:

    ```bash
    python msgpack_to_xyz.py \
    switches.msgpack \
    --inchikey FKXNWSFIEGEDQM-NXVVXOECNA-N \
    --output bridged_azobenzene.xyz
    ```

## Filtering

Helpers for filtering data according to geometry and energy criteria

### General notes

- Filter settings must be placed in json files

    `shnitsel_defaults.json` contains the defaults from SHNITSEL tools' `sanity_check()` function

- Helpers need the (one-based) atom indices for all bonds in the geometries

### `filter_sharc.py` (module)

Filters `SharcTrajectory` and `SharcEnsemble` objects from the `sharc_scripts/` workflow

### `filter_ase_atoms.py` (module)

Filters lists of ASE `Atoms` objects

- `filter_clashes()` only filters geometries where unbonded atoms are too close together

    Closer than the sum of their covalent radii multiplied by `clash_threshold`

- `filter_static()` applies only the geometry criteria in filter settings

- `filter_dynamic()` applies the geometry and energy criteria in filter settings

    However, the extended xyz files we are using in this project don't retain dynamic information like trajectory index and active state

    So the original NetCDF files (from https://zenodo.org/records/15482819) must be used

    `filter_dynamic()` applies filters onto the NetCDF data, then maps the retained geometries back onto the list of ASE `Atoms`

    `filter_method` can be `truncate`, `omit`, or `transect`

## Test set sampling

### `near_CI.py` (module)

Samples geometries near CIs by assigning sampling probabilities proportional to `exp(-0.5 * (gap / sigma) ** 2)`

- `choose_sigma()` helps to choose a suitable sigma that ensures `effective_sample_size >= 2 * n_samples`

## Transfer learning sampling

Samples data for transfer learning by calculating a descriptor vector for each geometry, then selecting from them

### `descriptors.py` (module)

Defines different descriptor vectors

- Currently implemented: `bond_lengths`, `pairwise_distances`, `energies`, `soap`, `acsf`, `mbtr`

- `calculate_descriptors()` is the main access point

- Relevant CLI arguments:

    `bond_lengths`: `--bond-pairs` (one-based atom indices)

### `selectors.py` (module)

Defines different selection methods

- Currently implemented: `random`, `fps`, `kmeans`, `stratified`

- `select_atoms()` is the main access point

- Relevant CLI arguments:

    `random`: `--seed`

    `fps`: `--initialize` (initial geometry, default: 0)

    `kmeans`: `--seed`, `--n-clusters`, `--weighted` (weight sample allocation by cluster size, ie sample more from bigger clusters)

    `stratified`: `--seed`, `--fractions` (size of each stratum, should sum to 1), `--allocations` (how many samples to take from each stratum, should sum to `--n-samples`)

### `run_tl_sampling.py` (CLI script)

Specify the desired descriptor and selector, run sampling, write output

- Example commands:

    ```bash
    python run_tl_sampling.py \
    ethene_dataset.xyz \
    ethene_sampled_by_energies_random.xyz \
    --n-samples 100 \
    # energies
    --descriptor energies \
    # random
    --selector random \
    --seed 42
    ```

    ```bash
    python run_tl_sampling.py \
    ethene_dataset.xyz \
    ethene_sampled_by_bondlengths_fps.xyz \
    --n-samples 100 \
    # bond lengths
    --descriptor bond_lengths \
    --bond-pairs 1-3 1-4 2-5 2-6 \
    # fps
    --selector fps \
    --initialize 1
    ```

    ```bash
    python run_tl_sampling.py \
    ethene_dataset.xyz \
    ethene_sampled_by_energies_kmeans.xyz \
    --n-samples 100 \
    # energies
    --descriptor energies \
    # kmeans
    --selector kmeans \
    --seed 42 \
    --n-clusters 10 \
    --weighted
    ```

    ```bash
    python run_tl_sampling.py \
    ethene_dataset.xyz \
    ethene_sampled_by_energies_stratified.xyz \
    --n-samples 100 \
    # energies
    --descriptor energies \
    # stratified
    --selector stratified \
    --seed 42 \
    --fractions 0.2 0.3 0.5 \
    --allocations 80 10 10
    # take 80 samples from the top 20%,
    # 10 samples each from the next 30% and 50%
    ```