# ANALYSIS TOOLS

This folder contains tools to analyse data and model prediction results

## Analyse data

Tools to visualise data

### `geometry_coordinates.py` (module)

Internal utility module that defines `BondLengthCoordinate`, `BondAngleCoordinate`, and `DihedralCoordinate`

- Can be constructed via `BondLengthCoordinate((a,b))`, `BondAngleCoordinate((a,b,c))`, and `DihedralCoordinate((a,b,c,d))` with one-based atom indices

### `plot_energy_surfaces.py` (module)

`plot_energy_scatter()`

- Plots a scatterplot projection of the dataset onto the 2D surface defined by `x_coordinate` and `y_coordinate`

- `REF_energy` is encoded by colour

`plot_energy_mesh_triangulate()`

- The same 2D projection, but the energy surface is triangulated

### `plot_geometry_features.py` (module)

`plot_bond_length_distribution()`, `plot_bond_angle_distribution()`, `plot_dihedral_distribution()`

- Plots a histogram of the specified geometry feature across the dataset

- Multiple equivalent features can be included

    Eg each ethene geometry can contribute 4 C-H bond lengths to the bond length histogram

### `visualise_dataset.ipynb` (notebook)

Applies a data filter, then plots all the 2D projections and histograms above

## Analyse model

Tools to visualise model prediction results

### `run_predictions.py` (CLI script)

Runs X-MACE predictions on a xyz file and writes `PRED_energy`, `PRED_forces`, `PRED_nacs` in place

- Example command:

    ```bash
    python run_predictions.py \
    ethene_data.xyz \
    ethene_xmace_model.pt \
    --device cuda \
    --batch-size 64 \
    --compute-force \
    --compute-nacs # requires a model trained on NACs
    ```

### `plot_predictions.py` (module)

`plot_pred_energy_mesh()`

- Plots a heatmap of `PRED_energy` onto the 2D surface defined by `x_coordinate` and `y_coordinate`

### `plot_residuals.py` (module)

Residuals are defined by `PRED_energy - REF_energy`

`plot_residual_heatmaps()`

- Plots a heatmap of the residual onto the 2D surface defined by `x_coordinate` and `y_coordinate`

- `absolute` takes the absolute value of the residual

`plot_residual_vs_coordinate()`

- Plots a scatterplot of the residual against a `coordinate`

`plot_residual_vs_gap()`

- Plots a scatterplot of the residual against the specified energy gap (defined by `REF_energy` difference)

### `analyse_model.ipynb` (notebook)

Takes in a xyz file with `PRED_energy` and (optionally) `REF_energy`, and plots the heatmaps and scatterplots above