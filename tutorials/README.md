# X-MACE class tutorials

These tutorials document the model and training classes, their responsibilities, and the conventions needed to use them together.

| Notebook | Focus | Status |
|---|---|---|
| [00 — Model inspection](00_model_inspection.ipynb) | Changes from the earlier model: autoencoder-head ownership, routing, targets, and NAC outputs | Included |
| [01 — Base model training](01_base_model_training.ipynb) | Step-by-step Python cells for loaders, model, loss, training, full-model saving, and Tester evaluation | Included |
| [02 — Transfer learning strategies](02_transfer_learning_strategies.ipynb) | NaiveStrategy, freezing, LoRA selection, and trainable parameters | Included |
| [03 — Multiheaded training](03_multiheaded_training.ipynb) | LF/HF loaders, per-head E0s, balanced sampling, independent heads, LoRA, and correction targets | Included |