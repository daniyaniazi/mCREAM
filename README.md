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

**Step 1 — Generate configs** from a parent YAML where list-valued fields define the search space:

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
