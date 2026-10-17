#!/usr/bin/env bash
set -euo pipefail
condor_submit server_scripts/cream_experiment/fixed_u2u/cfmnist/reference_fixed_u2c/train_cfmnist_graph_free_u2u_u2c_reference_job.sub
