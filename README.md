# Towards Reasonable Concept Bottleneck Models

Official code implementation of the paper [Towards Reasonable Concept Bottleneck Models](https://arxiv.org/abs/2506.05014).

---

### What `simple_main.py` does

`simple_main.py` is the main entry point and runs the full pipeline for a given config:

1. **Load** — reads the YAML config, seeds everything, instantiates the dataset and the right model class for the requested `mode`.

2. **Train & validate** — calls `trainer.fit()` via PyTorch Lightning. Gradient clipping is applied automatically for hard concept representations.

3. **Test** — calls `trainer.test()` and records accuracy and other metrics.

4. **Efficiency benchmarks & intermediate values** — runs an inference benchmark on the test dataloader and saves the latent activations (concepts + side channel) for both train and test sets to disk, used downstream for importance estimation.

5. **SAGE / PFI importance** (if a side channel is present) — computes Permutation Feature Importance (PFI, 100 repeats) and grouped SAGE values over concepts vs. side channel. The key output metric is the **Concept Contribution Index (CCI)**: the fraction of total SAGE importance attributed to the concept group. SAGE has a 1-hour timeout.

6. **Interventions** — sets an increasing number of concepts to their ground-truth values (from 0 up to `num_concepts`) and re-evaluates the model at each level, producing an intervention accuracy curve. Group interventions are also evaluated when a `softmax_mask` is provided.

7. **Save results** — all metrics, hyperparameters, timing, and intervention curves are written to a CSV in `<default_root_dir>/<dataset>/<mode>/<model>/metrics/`.

---

# How to Run

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

All scripts take a single `--config` argument pointing to a YAML file.

```bash
# Standard CBM training
python simple_main.py --config path/to/config.yaml

# With propagating interventions
python training_and_propagating_interventions.py --config path/to/config.yaml

# With removed concepts (CUB dataset only)
python training_with_removed_concepts.py --config path/to/config.yaml
```

The `mode` field in the YAML controls what gets trained:

| `mode` | What it does |
|---|---|
| `train_cbm` | Full pipeline: image → concepts → labels |
| `train_x2c` | Concept extractor only (image → concepts) |
| `train_c2y` | Prediction head only (concepts → labels) |
| `train_x2y` | Black-box baseline (image → labels, no concepts) |
| `train_backbone` | Image backbone only |

---

## SAGE Feature Importance (standalone)

```bash
python sage_importance.py --config path/to/config.yaml
```

Requires `paths.input_model_path` to point to a trained checkpoint.

---

## Hyperparameter Grid Search

**Step 1 — Generate configs** from a parent YAML where list-valued fields define the search space. _This search space includes seeds_:

```bash
python generate_grid_hypertune.py --config path/to/parent_config.yaml --output_dir generated_yamls/
```

**Step 2 — Run all configs:**

```bash
for cfg in generated_yamls/my_config/*.yaml; do
    python simple_main.py --config "$cfg"
done
```

**Step 3 — Aggregate results:**

```bash
python aggregate_results_hypertune.py --result_folder path/to/results/ --output_dir aggregated_results/
```

---

## Pretrained Models

Download the pretrained models folder from:

[https://cloud.mi.uni-saarland.de/s/RYStkW2TgJoJPKr](https://cloud.mi.uni-saarland.de/s/RYStkW2TgJoJPKr)

Place the downloaded folder at the root of the repository so the path `./pretrained_models/` matches what is expected in the config files.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kalampalikis2026reasonableconceptbottleneckmodels,
      title={Towards Reasonable Concept Bottleneck Models}, 
      author={Nektarios Kalampalikis and Kavya Gupta and Georgi Vitanov and Isabel Valera},
      year={2026},
      eprint={2506.05014},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.05014}, 
}
```
