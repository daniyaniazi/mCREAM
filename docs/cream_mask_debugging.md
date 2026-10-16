# CREAM Mask and Concept Debug Trace

## Forward path

For a `train_cbm` CREAM run, `simple_main.py` performs the setup in this order:

1. It reads `paths.DAG_file` with pandas and converts the adjacency values to a boolean tensor. The index and columns of the CSV are the node labels; they are passed into the debug trace for readable edge names.
2. It loads the frozen backbone checkpoint. `Template_CBM_MultiClass` later calls `model1.concept_extractor(image)` to produce the backbone vector `x`.
3. It constructs `UtoY_model` with the DAG and CREAM hyperparameters, then wraps the backbone concept extractor and `UtoY_model` in `Template_CBM_MultiClass`.
4. When `debug_masking.enabled` is true, it configures the diagnostic on `UtoY_model`. The trace goes to `<experiment output directory>/mask_debug.log`.

Inside `UtoY_model.__init__`:

1. `num_exogenous` is the output width of the representation splitter `u2u_model`; `num_side_channel` is the tail reserved for the direct task side channel.
2. `Uc_dim = num_exogenous - num_side_channel`. `init_representation_splitter()` maps the backbone vector to `num_exogenous` values, then ReLU is applied.
3. The C--C graph is sliced as `causal_graph[:-num_classes, :-num_classes]`. The C--Y/task block is sliced separately. The concept mask is created from the C--C block.
4. `_replicate_columns()` expands the concept mask so each source concept's latent input block has one mask column per latent coordinate. With no side channel, the expanded mask should have shape `[num_concepts, Uc_dim]`.
5. `MaskedMLP` is built with this expanded adjacency. With `num_hidden_layers_in_maskedmlp: 0`, this is the direct masked linear concept layer.

Each `UtoY_model.forward(x)` then does:

1. `u = u2u_model(x)`: produce the latent vector.
2. Split `u` into `Uc` and `Uy`. With `num_side_channel: 0`, `Uc` is the full vector and `Uy` is empty.
3. `concept_before_activation = u2c_model(Uc)`: apply the C--C masked layer(s), producing concept logits.
4. `c = concept_activation_function(concept_before_activation)`: for `soft`, sigmoid is applied per concept; for `group_soft`, mutually exclusive groups use softmax and other concepts use sigmoid.
5. Feed `c` to the task layer. With no side channel, the task input is exactly `c`.

The intervention path does not change the learned C--C mask. `forward_with_interventions()` first computes `c`, copies it, overwrites selected entries with ground-truth concepts, and sends the resulting vector to the task head. Therefore intervention curves alone do not prove that two C--C masks produced the same pre-intervention concept vector. Compare the logits and activated concepts in the mask trace.

## What the debug trace records

The first two `UtoY_model` forwards in each run are traced. Each trace includes:

- Experiment name, DAG path, dimensions, side-channel width, and latent coordinates per concept.
- True positions in the loaded C--C adjacency, reported as CSV row/column indices and node names.
- Expanded mask shape, active count, and allowed input concept blocks for each output row.
- Backbone vector, `u2u_model` output, and `Uc` immediately before C--C masked layers.
- For each masked layer found inside the concept and task modules: input, mask shape/count, raw weights, effective `raw_weight * mask`, active coordinates and their weights, and output.
- Concept logits before activation, concept vector after activation, task-head input, and task logits.

`debug_masking` is opt-in and defaults to off. The generated no-side-channel graph experiment configs enable it with `max_forwards: 2`. To turn it on in another config, add:

```yaml
debug_masking:
  enabled: true
  max_forwards: 2
```

## Reading comparisons

Compare logs for the same dataset, seed, and sample ordering across graph variants:

- If the loaded C--C edge lists or expanded masks are identical, graph generation, config DAG paths, or graph loading is the first place to investigate.
- If the expanded masks differ but effective masked weights do not, inspect layer mask application and weight/mask dimensions.
- If effective weights differ but the layer output and concept logits match, inspect the logged `Uc` values and whether changed edges receive near-zero weights or nearly identical inputs.
- If concept logits differ but activated concept vectors look nearly equal, inspect the activation. Sigmoid or group softmax can compress differences.
- If activated concept vectors differ but intervention curves appear equal, compare intervention masks, test samples, and task-head behavior. The task head can be insensitive to changed concept values.

For a fair graph-only comparison, use the same seed, backbone checkpoint, dataset split, latent dimension, optimizer settings, and training budget. Each graph variant is trained independently, so the learned weights need not match even when the parameter shapes do.
