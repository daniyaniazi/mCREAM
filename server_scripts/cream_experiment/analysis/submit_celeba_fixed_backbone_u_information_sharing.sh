#!/usr/bin/env bash
set -euo pipefail

condor_submit server_scripts/cream_experiment/analysis/celeba_fixed_backbone_u_information_sharing_job.sub
