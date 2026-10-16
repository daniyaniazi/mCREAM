#!/usr/bin/env bash
set -euo pipefail
condor_submit server_scripts/cream_experiment/fixed_u2u/cfmnist/reference/train_cfmnist_graph_free_u2u_reference_job.sub
