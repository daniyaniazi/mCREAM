# Conditional U information bottleneck experiment

This is an optional CREAM training objective for the fixed-backbone topology runs. It is disabled for existing configs unless `u_information_bottleneck.enabled: true` is added.

## Objective

For each concept-assigned U group, the experiment adds a variational upper bound on

`sum_k I(X; U_k | C_k)`.

The deterministic U2U output `mu_k(X)` is treated as the mean of a noisy Gaussian encoder

`q(U_k | X) = Normal(mu_k(X), noise_std^2 I)`.

For each binary concept label, a trainable diagonal Gaussian prior models

`r(U_k | C_k) = Normal(prior_mean[k, C_k], diag(exp(prior_logvar[k, C_k])))`.

The minibatch penalty is the mean over examples of the summed analytic KL divergence `KL(q(U_k|X) || r(U_k|C_k))`. Ground-truth concept labels condition the prior during training. Noise is sampled during training and disabled during evaluation, so evaluation uses the deterministic U2U mean.

The CREAM loss becomes

`L = task_loss + lambda_CREAM * concept_loss + lambda_IB * sum_k KL(q(U_k|X) || r(U_k|C_k))`.

This is a variational upper bound under the chosen Gaussian family, not an exact estimate of conditional mutual information.

## What this does and does not enforce

The U2C graph mask still determines which U groups may influence each concept output. The information bottleneck encourages each U group to retain less image-specific variation after its assigned concept label is known. It does not guarantee that a group contains only the assigned concept: other image attributes correlated with that concept can remain encoded. Use the per-concept probes and group ablations to assess the learned representations.

## Experiment configs

Five isolated CelebA configs are under `all_configs/fixed_backbone/celeba_concept_concept_fixcount_topology_no_side_channel_fixed_backbone_u_ib/`. They reuse the same fixed-backbone feature cache and topology DAGs as the existing runs. The new config folder and experiment name keep checkpoints separate from the original experiments.

Current starting settings are `lambda_weight: 0.001` and `noise_std: 1.0`. These are experiment hyperparameters, not validated universal defaults. The KL sums over all U dimensions and groups, so the effective strength depends on its scale. Compare a small sweep of `lambda_weight` values if the validation loss or concept/task accuracy degrades sharply.

Submit all five topology jobs from the server repository root:

```bash
bash server_scripts/cream_experiment/celeba/concept_concept_fixcount_topology_no_side_channel_fixed_backbone_u_ib/submit_all_celeba_fixed_backbone_u_ib.sh
```

The separate analysis config compares the new replace_000 and replace_100 checkpoints. Submit it after training completes:

```bash
bash server_scripts/cream_experiment/analysis/submit_celeba_fixed_backbone_u_ib_information_sharing.sh
```

Then open `notebook/u_information_sharing_conditional_ib_results.ipynb` to inspect the outputs.
