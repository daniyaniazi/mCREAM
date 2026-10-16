#!/usr/bin/env bash
set -euo pipefail
condor_submit server_scripts/cream_experiment/fixed_u2u/celeba/reference/train_celeba_graph_free_u2u_reference_job.sub
