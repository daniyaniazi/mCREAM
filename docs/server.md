# SIC HTCondor Quick Reference

## Login

Log in to one of the submit nodes from inside the university network:

```bash
ssh <username>@conduit.hpc.uni-saarland.de
ssh <username>@conduit2.hpc.uni-saarland.de
```

Use your SIC credentials. Submit jobs from these machines only.

## Docker Pattern

Use the Docker universe. This is the minimal pattern:

```condor
universe = docker
docker_image = pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

executable = run_job.sh

request_cpus = 4
request_memory = 32 GB
request_gpus = 1

+WantGPUHomeMounted = true

output = logs/job.$(ClusterId).$(ProcId).out
error = logs/job.$(ClusterId).$(ProcId).err
log = logs/job.$(ClusterId).log

queue 1
```

Use these fields carefully:

- `universe = docker` is required.
- `docker_image` should be a public image you can actually run.
- `executable` is the script HTCondor starts in the container.
- `request_cpus`, `request_memory`, and `request_gpus` must match your real needs.
- `+WantGPUHomeMounted = true` mounts your `/home` into the container.

## Sample

### `run_cream.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/CREAM"
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml
```

Make it executable:

```bash
chmod +x run_cream.sh
```

### `cream_job.sub`

```condor
universe = docker
docker_image = pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

executable = run_cream.sh

request_cpus = 8
request_memory = 64 GB
request_gpus = 1

+WantGPUHomeMounted = true

output = logs/cream.$(ClusterId).$(ProcId).out
error = logs/cream.$(ClusterId).$(ProcId).err
log = logs/cream.$(ClusterId).log

queue 1
```

Submit it with:

```bash
condor_submit cream_job.sub
```

## Commands

```bash
# submit a job
condor_submit my_job.sub

# interactive job
condor_submit -i my_job.sub

# show your jobs
condor_q

# why is a job held?
condor_q -hold <jobid>

# why is a job not matching?
condor_q -analyze <jobid>
condor_q -better-analyze <jobid>

# release / hold / remove
condor_release <jobid>
condor_hold <jobid>
condor_rm <jobid>
condor_rm -a

# inspect workers
condor_status

# edit a job requirement
condor_qedit <jobid> RequestMemory 64GB
```

## Common

- Job stays `Idle`: your resource request is too strict, or no matching worker is free.
- Job goes `Held`: check `condor_q -hold <jobid>` first.
- Script not found: make sure `executable` exists and is executable.
- Files missing inside container: mount `/home` with `+WantGPUHomeMounted = true`.
- Output disappears: anything written only to `/tmp` is temporary.
- Interactive job exits immediately: the Docker image may have an entrypoint that finishes right away.

## Commas

When asking for resources, be conservative: too small and the job fails or is held, too large and it waits longer in the queue.

<!-- SSH FIRST -->

SSH to submit node and prepare repo once
ssh username@conduit.hpc.uni-saarland.de
cd $HOME
git clone <your-repo-url> mCREAM
cd mCREAM
mkdir -p logs
Create a reusable venv in your home directory
This is created once and reused across jobs.

<!-- CREATE ENV -->

python3 -m venv $HOME/.venvs/mcream
source $HOME/.venvs/mcream/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, pytorch_lightning, yaml; print('torch=', torch.version, 'cuda=', torch.cuda.is_available())"

<!-- .SH FILE -->

Create run script for Condor job
Put this in run_cream.sh:

#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/mCREAM"
source "$HOME/.venvs/mcream/bin/activate"

echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
python -V
nvidia-smi || true

python -c "import torch; print('torch', torch.version, 'cuda', torch.cuda.is_available())"
python simple_main.py --config CREAM_best_cfmnist_soft_config.yaml

Then:

chmod +x run_cream.sh

<!-- CLUSTER FILE -->

Create Condor submit file
Put this in cream_job.sub:

universe = docker
docker_image = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable = run_cream.sh

request_cpus = 8
request_memory = 64 GB
request_gpus = 1
+WantGPUHomeMounted = true

output = logs/cream.(ClusterId).(ClusterId).(ProcId).out
error = logs/cream.(ClustedId).(ClusterId).(ProcId).err
log = logs/cream.$(ClusterId).log

queue 1

Submit and check if project is running
Submit:

condor_submit cream_job.sub

Check queue:

condor_q

If stuck/held:

condor_q -hold <jobid>
condor_q -better-analyze <jobid>

Watch logs:

tail -f logs/cream.<ClusterId>.<ProcId>.out
tail -f logs/cream.<ClusterId>.<ProcId>.err

What to look for in output log:

Line with Using config: ...
Torch version and cuda True
Training/test progress messages
No immediate Python import errors
Important compatibility note:

My requirements.txt pins torch 2.4.0 and torchvision 0.19.0.
If you keep docker image 2.2.2, you may get version conflicts.
Using a 2.4.0 PyTorch image (as above) is safer.
