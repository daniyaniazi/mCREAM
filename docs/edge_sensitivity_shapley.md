# CREAM edge sensitivity and Shapley workflow

This workflow compares the trained fixed-backbone CelebA and Complete Concept FMNIST `replace_000` and `replace_100` models.

## What it measures

For an existing C→C edge, the evaluation script zeros the entire corresponding source-U-group to target-concept weight block while holding all learned parameters fixed. It reports task accuracy, overall concept accuracy, every individual concept accuracy, and the task/concept intervention-accuracy curves. Edge-removal drop is computed as baseline accuracy minus accuracy after removing that edge; positive values mean the removal hurt.

Sampled Shapley values separately estimate each edge's average marginal change in the same metrics over random edge orderings. They use the model's learned weights fixed and evaluate coalitions in which edges are enabled or disabled. Shapley values and single-edge drops are predictive importance summaries under this mask operation.

## Stage 1: fixed-weight evaluation and Shapley

The analysis jobs are configured by:

- `all_configs/analysis/edge_sensitivity/celeba_replace_000_vs_replace_100.yaml`
- `all_configs/analysis/edge_sensitivity/cfmnist_replace_000_vs_replace_100.yaml`

Important fields include checkpoint globs, `evaluation_splits`, `shapley_split`, `shapley_permutations`, `intervention_seed`, `selection_metric`, and `top_edges_per_model`. The defaults evaluate edge ablations on validation and test, calculate Shapley on validation, and sample 100 edge orderings.

From the server repository root submit both jobs:

```bash
bash server_scripts/cream_experiment/analysis/submit_edge_sensitivity_all.sh
```

Results go to `experiments/edge_sensitivity/celeba/` and `experiments/edge_sensitivity/cfmnist/`:

- `edge_list.csv`: C→C edges and concept names.
- `coalition_metrics.csv`: baseline, single-edge ablation, and sampled coalition metrics in long format, including all per-concept accuracy values and intervention curves.
- `shapley_values.csv`: sampled Shapley value, permutation standard deviation/SE, for every edge and metric.
- `shapley_permutation_paths.csv`: sampled edge ordering steps.
- `run_metadata.json`: split and protocol details.

Open `notebook/edge_sensitivity_shapley_analysis.ipynb`, set `DATASET` to `celeba` or `cfmnist`, and run its cells. Set `SELECTION_METRIC` to `task_accuracy`, `concept_accuracy`, `concept_accuracy_###`, or an intervention AUC field. The export cell writes the top validation-set removals per model to `selected_edges_for_retraining.csv`.

## Stage 2: retrain selected removals

After the notebook writes the selected-edge CSV, generate one altered DAG and one fixed-backbone training config per selected model/edge/seed:

```bash
python scripts/generate_edge_sensitivity_retraining.py \
  --analysis-config all_configs/analysis/edge_sensitivity/celeba_replace_000_vs_replace_100.yaml
```

For CFMNIST, replace the config name with `cfmnist_replace_000_vs_replace_100.yaml`. The script uses the notebook-selected edge rows and `training_seeds` from the analysis config. It writes a retraining plan under that dataset's analysis output directory. Submit the generated jobs on the server with the printed command, for example:

```bash
bash experiments/edge_sensitivity/celeba/retraining_plan/submit_retraining.sh
```

Training outputs use `edge_sensitivity/<model>/remove_<edge>/seed_<seed>` experiment names and preserve the base experiment's frozen-backbone, U2U/U2C training, dataset split, and optimizer settings.

## Notes

- Use validation drops to choose edges. The notebook displays test results for final reporting, avoiding selection based on test performance.
- Intervention masks and the train-set concept percentiles are held fixed across edge coalitions for a model so coalition metrics are comparable. CelebA reports individual-concept interventions; CFMNIST reports both group and individual interventions as supported by its config.
- Shapley permutations and metric batch size are configurable. More permutations reduce Monte Carlo noise but increase runtime.
- Retraining selection defaults to the top three edges per model ranked by validation task-accuracy drop. Change `selection_metric`, `top_edges_per_model`, and `training_seeds` before running the analysis if needed.
