#!/usr/bin/env bash
set -euo pipefail

bash server_scripts/cream_experiment/fixed_u2u/submit_cache_celeba_features.sh
bash server_scripts/cream_experiment/fixed_u2u/submit_cache_cub_features.sh
bash server_scripts/cream_experiment/fixed_u2u/submit_cache_cfmnist_features.sh
