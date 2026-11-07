#!/usr/bin/env bash
set -euo pipefail

chmod +x server_scripts/cream_experiment/analysis/run_celeba_fixed_backbone_u_ib_information_sharing.sh
condor_submit server_scripts/cream_experiment/analysis/celeba_fixed_backbone_u_ib_information_sharing_job.sub
