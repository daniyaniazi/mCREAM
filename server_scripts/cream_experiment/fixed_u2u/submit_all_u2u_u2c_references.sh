#!/usr/bin/env bash
set -euo pipefail

bash server_scripts/cream_experiment/fixed_u2u/submit_reference_celeba_u2u_u2c.sh
bash server_scripts/cream_experiment/fixed_u2u/submit_reference_cub_u2u_u2c.sh
bash server_scripts/cream_experiment/fixed_u2u/submit_reference_cfmnist_u2u_u2c.sh
