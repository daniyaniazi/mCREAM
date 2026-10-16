#!/usr/bin/env bash
set -euo pipefail
condor_submit server_scripts/cream_experiment/fixed_u2u/celeba/cache/cache_celeba_frozen_backbone_features_job.sub
