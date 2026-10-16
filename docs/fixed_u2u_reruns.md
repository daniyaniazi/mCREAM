# Fixed-U2U Graph Reruns

The six no-side-channel graph families can be rerun with a graph-free shared
representation. The pipeline has three ordered stages:

1. Cache frozen-backbone features for the train, validation, and test splits.
   CelebA and CUB use their deterministic evaluation transform for the cached
   training images, so each image has one fixed feature vector.
2. Train one unmasked, no-side-channel concept/task model per dataset using
   those cached features. This reference has no DAG, and its learned `U2U`
   projection is saved to `shared_u2u.pt`.
3. Rerun each graph variant using the same cache and frozen `U2U` file. Only
   the downstream graph-constrained model layers are trained per variant.

The original graph's `U2U` is not used: it was trained end-to-end with the
original graph active, so its gradients were graph-influenced. The reference
`U2U` is trained once without a DAG. All generated configs and results have a
`fixed_u2u` suffix and do not overwrite the existing runs.

## Cluster Run Order

From `~/mCREAM`, run each stage only after the previous Condor jobs complete:

```bash
bash server_scripts/cream_experiment/fixed_u2u/submit_all_feature_caches.sh
```

After all three feature-cache jobs finish:

```bash
bash server_scripts/cream_experiment/fixed_u2u/submit_all_u2u_references.sh
```

After all three graph-free reference jobs finish and each `shared_u2u.pt`
exists:

```bash
bash server_scripts/cream_experiment/fixed_u2u/submit_all_graph_variants_fixed_u2u.sh
```

The cache files are written under `data/fixed_u2u_feature_cache/{celeba,cub,cfmnist}/`.
The shared projections are saved under each dataset's
`experiments/<dataset>/train_cbm/<model>/fixed_u2u_reference/` directory.

These cached-feature runs are for training, predictive metrics, and
intervention analysis. Because their input is a feature vector rather than an
image tensor, image-based CAM export should use the regular image-input runs.
