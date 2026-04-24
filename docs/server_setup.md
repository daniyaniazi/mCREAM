# mCREAM Server Setup (Conduit + HTCondor)

This guide summarizes the exact steps to run a quick demo job first, then run CREAM experiments on the university server.

## 1. Login to server

From local terminal:

```bash
ssh dani00003@conduit.hpc.uni-saarland.de
```

## 2. Repository and folders

```bash
cd ~
# clone once (if not present)
# git clone https://github.com/daniyaniazi/mCREAM.git

cd ~/mCREAM
mkdir -p logs
```

## 3. Python environment on server

Create venv once:

```bash
python3 -m venv ~/.venvs/mcream
source ~/.venvs/mcream/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick check:

```bash
python -c "import torch, pytorch_lightning, yaml; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
```

Note: On login node, cuda may show False. GPU check must be done inside a Condor GPU job.

## 4. Copy pretrained models from local machine

From local Windows PowerShell:

```powershell
scp -r D:\cbm_hiwi\mCREAM\pretrained_models dani00003@conduit.hpc.uni-saarland.de:~/mCREAM/
```

Verify on server:

```bash
ls ~/mCREAM/pretrained_models
```

## 5. Demo smoke test job (recommended before CREAM)

Files used:

- server_scripts/demo_experiment/test_docker.sub
- server_scripts/demo_experiment/test_execute.sh
- server_scripts/demo_experiment/demo.py

Submit:

```bash
cd ~/mCREAM/server_scripts/demo_experiment
chmod +x test_execute.sh
condor_submit test_docker.sub
condor_q
```

Inspect:

```bash
condor_q -better-analyze <jobid>
tail -f ~/mCREAM/logs/cream.<ClusterId>.<ProcId>.out
tail -f ~/mCREAM/logs/cream.<ClusterId>.<ProcId>.err
```

If the job disappears from queue, check final status:

```bash
cat ~/mCREAM/logs/cream.*.log
```

## 6. Common Condor issues and fixes

### Return value 127

Usually means command not found or invalid interpreter path in script.

Checklist:

```bash
cd ~/mCREAM/server_scripts/demo_experiment
chmod +x test_execute.sh
dos2unix test_execute.sh  # if available
```

Also ensure Python path exists in the container or use container Python fallback in script.

### Job stays Idle

Run:

```bash
condor_q -better-analyze <jobid>
```

Interpretation:

- 0 willing to run now: wait for resources.
- would match if drained: matching slots are busy but valid.

## 7. Run CREAM experiment

After demo works, run the main config:

```bash
cd ~/mCREAM
source ~/.venvs/mcream/bin/activate
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml
```

Or submit via Condor using a run script and submit file with:

- universe = docker
- +WantGPUHomeMounted = true
- request_GPUs = 1
- appropriate CPU/memory

## 8. GitHub authentication from server (SSH)

If git push asks for username/password, switch to SSH.

Generate key on server:

```bash
ssh-keygen -t ed25519 -C "farihadania@hotmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key at:

- https://github.com/settings/keys

Test auth:

```bash
ssh -T git@github.com
```

Switch repo remote:

```bash
cd ~/mCREAM
git remote set-url origin git@github.com:daniyaniazi/mCREAM.git
git remote -v
git push
```

## 9. Keep repo clean

- Keep miniconda installer outside repo root.
- Keep runtime outputs in logs/.
- Do not store copied key text in project env files.

Useful checks:

```bash
cd ~/mCREAM
git status
condor_q
```
