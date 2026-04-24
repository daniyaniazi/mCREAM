# CREAM Experiments Guide

## Entry Points

| Script | Purpose |
|---|---|
| `simple_main.py` | Main script — trains, tests, runs SAGE importance, interventions, benchmarks |
| `training_and_propagating_interventions.py` | Trains with interventions that propagate through the concept graph |
| `training_with_removed_concepts.py` | CUB only — trains with concept subsets removed |
| `sage_importance.py` | Standalone SAGE importance on a trained checkpoint |
| `generate_grid_hypertune.py` | Generates YAML grid from a parent config |
| `aggregate_results_hypertune.py` | Aggregates CSVs from a hyperparameter sweep |

---

## Training Modes (set via `mode` in YAML)

| `mode` | Description |
|---|---|
| `train_cbm` | **Main mode.** Full pipeline: image → concepts → task. Trains CREAM or CBM. |
| `train_x2y` | Black-box baseline: image → task (no concepts). Uses a pretrained backbone. |
| `train_c2y` | Oracle baseline: true concepts → task (estimates Ctrue→Y ceiling). |
| `train_x2c` | Concept extractor only: image → concepts. |
| `train_backbone` | Backbone only: image → embedding (no concepts). |

---

## What `simple_main.py` Does End-to-End

1. Loads config YAML and seeds everything (`pl.seed_everything`)
2. Instantiates dataset (auto-downloads if needed)
3. Instantiates the model class from the registry in `src/utils.py`
4. Runs `trainer.fit()` (train + val)
5. Runs `trainer.test()` and saves metrics to CSV
6. Saves latent activations (concepts + side-channel) to disk
7. Computes PFI (Permutation Feature Importance, 100 repeats) and SAGE values → CCI
8. Runs intervention sweep (0 → num\_concepts interventions) and logs accuracy at each step
9. Writes all results to `<default_root_dir>/<dataset>/<mode>/<model>/last_metrics/`

---

## Simplest Experiment to Run First

**CREAM on incomplete FashionMNIST (iFMNIST)** — fastest dataset, no special data download needed:

```bash
cd ~/mCREAM
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_ifmnist_soft_config.yaml
```

Expected outputs:
- Task accuracy ~92%, Concept accuracy ~99%
- CCI > 0.5 (concepts dominate side-channel)
- Intervention curve showing accuracy rises with each corrected concept

---

## All Best-Config Experiments

### CREAM (main model)
```bash
# iFMNIST (incomplete FashionMNIST, K=8)
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_ifmnist_soft_config.yaml

# cFMNIST (complete FashionMNIST, K=11)
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml

# CUB (200 bird species, K=112) — slow, 300 epochs
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cub_soft_config.yaml

# CelebA (smiling prediction, K=7) — requires CelebA download
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_celeba.yaml
```

### Baselines
```bash
# Vanilla CBM (no reasoning structure)
python simple_main.py --config all_configs/best_hparams/CBM/CBM_cfmnist.yaml

# Black-box (no concepts)
python simple_main.py --config all_configs/blackbox_configs/blackbox_fmnist.yaml

# Ctrue→Y oracle (upper bound using true concept values)
python simple_main.py --config all_configs/c2y_configs/c2y_cfmnist_train.yaml
```

### CREAM without side-channel
```bash
python simple_main.py --config all_configs/best_hparams/CREAM_no_side_channel/CREAM_no_side_best_cfmnist_soft_config.yaml
```

---

## Config File Structure

```yaml
mode: train_cbm               # training mode
seed: [42, 7, 1, 134, 89]    # list = runs all seeds sequentially
dataset_name: Concept_FMNIST  # dataset key (see dataset_setup.md)
dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true
model_name: Standard_FashionMNIST  # backbone key (see src/utils.py registry)
paths:
  default_root_dir: ./experiments/
  metric_dir: ./last_metrics/
  DAG_file: ./data/FashionMNIST/Concept_FMNIST_DAG.csv   # reasoning graph
  input_model_path: ./pretrained_models/FMNIST/...        # pretrained backbone
  softmax_mask: ./data/FashionMNIST/mutually_exclusive_relationships.json  # mutex groups
hyperparameters_model2:          # bottleneck (CREAM/CBM part)
  num_classes: 10
  num_concepts: 8                # K
  num_exogenous: 76              # dC×K + num_side_channel
  num_side_channel: 20           # side-channel size (0 for no side-channel)
  concept_representation: group_soft
  masking_algorithm: zuko        # 'zuko' = StrNN, 'none' = vanilla CBM
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: 128
  last_layer_mask: true
  side_dropout: true
  dropout_prob: 0.9              # p - higher = more concept-reliant
hyperparameters:                 # outer wrapper (Template_CBM_MultiClass)
  learning_rate: 0.001
  lambda_weight: 1               # λ - weight of concept loss
  frozen_model1: true            # freeze backbone during training
trainer_param:
  max_epochs: 50
```

---

## Key Config Fields for Ablations

| Change | Field |
|---|---|
| Remove side-channel | `num_side_channel: 0`, `side_dropout: false` |
| Vanilla CBM (no structure) | `masking_algorithm: none`, `last_layer_mask: false` |
| More/less concept regularization | Increase/decrease `dropout_prob` |
| Hard concepts | `concept_representation: hard` or `group_hard` |
| Deeper StrNN | Increase `num_hidden_layers_in_maskedmlp` |

---

## Output Location

Results saved to:
```
./experiments/<dataset_name>/<mode>/<model_name>/<config_folder>/last_metrics/<config_stem>.csv
```

TensorBoard logs also saved there. View with:
```bash
tensorboard --logdir ./experiments/
```

---

## Hyperparameter Search

```bash
# 1. Generate grid configs from a parent YAML (list fields = search axes)
python generate_grid_hypertune.py \
    --config all_configs/others/ablations/... \
    --output_dir generated_yamls/

# 2. Run all configs
for cfg in generated_yamls/my_config/*.yaml; do
    python simple_main.py --config "$cfg"
done

# 3. Aggregate to single CSV
python aggregate_results_hypertune.py \
    --result_folder experiments/... \
    --output_dir aggregated_results/
```

---

## Running on HTCondor (Server)

See [server.md](../../server.md) for full details. Typical job script pattern:

```bash
# run_cream.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mCREAM"
source "$HOME/.venvs/mcream/bin/activate"
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml
```

```condor
# cream_job.sub
universe = docker
docker_image = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable = run_cream.sh
request_cpus = 8
request_memory = 64 GB
request_gpus = 1
+WantGPUHomeMounted = true
output = logs/cream.$(ClusterId).$(ProcId).out
error  = logs/cream.$(ClusterId).$(ProcId).err
log    = logs/cream.$(ClusterId).log
queue 1
```
