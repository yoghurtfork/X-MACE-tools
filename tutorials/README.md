# X-MACE class tutorials

These few notebooks document the various helper classes that were coded within X-MACE for use in training a model / for transfer learning. 

| Notebook | Focus | 
|---|---|
| [00 — Model inspection](00_model_inspection.ipynb) | Changes from the earlier model: autoencoder-head ownership, routing, targets, and NAC outputs | 
| [01 — Base model training](01_base_model_training.ipynb) | Step-by-step Python cells for loaders, model, loss, training, full-model saving, and Tester evaluation | 
| [02 — Transfer learning strategies](02_transfer_learning_strategies.ipynb) | NaiveStrategy, freezing, LoRA selection, and trainable parameters | 
| [03 — Multiheaded training](03_multiheaded_training.ipynb) | LF/HF loaders, per-head E0s, balanced sampling, independent heads, LoRA, and correction targets | 

```text
           o
           |
      .----------.
      |  o    o  |           H
      |    ▿     |           |
      '----------'       H---C---H
        /|      |\           |
       / |  ⚛   | '----[ ]    H
     [ ] |______|       \___/
          |    |
         _|    |_        happy training!
        [__]  [__]
```
