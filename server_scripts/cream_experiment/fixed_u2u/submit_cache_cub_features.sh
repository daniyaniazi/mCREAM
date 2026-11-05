#!/usr/bin/env bash
set -euo pipefail
condor_submit server_scripts/cream_experiment/fixed_u2u/cub/cache/cache_cub_frozen_backbone_features_job.sub
