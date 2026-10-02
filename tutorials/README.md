# X-MACE class tutorials

These tutorials document the model and training classes, their responsibilities, and the conventions needed to use them together.

| Notebook | Focus | Status |
|---|---|---|
| [00 — Model inspection](00_model_inspection.ipynb) | Changes from the earlier model: autoencoder-head ownership, routing, targets, and NAC outputs | Included |
| [01 — Base model training](01_base_model_training.ipynb) | Step-by-step Python cells for loaders, model, loss, training, and saving weights | Included |
| 02 — Freezing and LoRA | Parameter selection, adapters, optimiser groups, and merging | Planned |
| 03 — Multi-head training and data loading | Head-labelled datasets, batching, E0s, and correction routes | Planned |

## Reading tutorial 00

Tutorial 00 assumes familiarity with the earlier X-MACE autoencoder architecture. It concentrates on what changed and what those changes mean for inspecting models and adapting scripts. The original [model inspection notebook](https://github.com/naythanyeo/X-MACE/blob/30db816e451ebdf0abafe3b04c510df167912604/nicer_tutorials/03_model_inspection.ipynb) provides background explanations; it remains unchanged.

Open tutorial 00 in GitHub's notebook viewer, Jupyter, or VS Code. All cells are Markdown. No data, checkpoint, Python environment, or execution is required; fenced snippets illustrate implementation names.

## Reading tutorial 01

[01 — Base model training](01_base_model_training.ipynb) is a step-by-step notebook with Python cells and Markdown explaining each step. It builds a single-head model from energy, force, and raw NAC labels, trains it, and saves the best weights. The cells are unexecuted and use placeholder training, validation, and output paths. Replace those paths and use your own compatible data and local X-MACE environment to run it; no data is bundled.

## Comparison scope

The guide compares the local pre-head implementation at [b63f0d3](https://github.com/naythanyeo/X-MACE/tree/b63f0d3f9df61f7ff091f4d69a2ef72b13ee9e4b) with the `nacnacnac` implementation at [30db816](https://github.com/naythanyeo/X-MACE/tree/30db816e451ebdf0abafe3b04c510df167912604). This is a specific code comparison, not a claim about every upstream release or the training conventions of historical checkpoints.

The notebook adapts the original inspection explanations with attribution and links to the relevant implementation. Planned tutorials are not yet included.
