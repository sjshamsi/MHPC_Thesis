## Walrus architecture-scaling runs on Leonardo

Mirrors the pattern in `../ExampleRunScripts` (Hydra CLI overrides templated into SLURM
scripts), retargeted at CINECA Leonardo (`boost_usr_prod`, account `ICT26_MHPC_0`, 4x A100/node)
and the `env1` Python environment.

### Environment

`env1` (`/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1`) is a plain venv (no
`--system-site-packages`) built from a pyenv-managed Python 3.11.9, with `torch==2.5.1+cu121`,
`torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`, and the rest of `pyproject.toml`'s
dependencies, plus `walrus` installed editable (`pip install -e .`) so source edits take effect
immediately. Every script here does `module purge; module load python; module load cuda/12.2`
before activating it — `module purge` is required because the login/compute node's default
profile auto-loads `cineca-ai/4.3.0`, which injects an older `torch` (2.2.0a0) onto `PYTHONPATH`
ahead of the venv and silently shadows it otherwise.

### Two tiers

1. **`smoke_test.sh`** — hand-written, not generated. Tiny model + tiny data
   (`data=debug model=debug`, `active_matter`, `max_samples=8`), `boost_qos_dbg` (max 30 min).
   Run this first to confirm the whole chain works: modules → venv → torchrun → data loading from
   our downloaded dataset path → training → checkpoint → logging.
   ```
   sbatch smoke_test.sh
   ```
   A checkpoint should appear under `/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/smoke_test/0/checkpoints/`.

2. **`gen_scaling_runs.py`** — generator for the real architecture sweep. Vary one axis
   (`model.hidden_dim`, `model.processor_blocks`, `model.processor.space_mixing.mlp_dim`) at a
   time around a base anchor (edit the `base`/`options` dicts at the top of the file to change
   the sweep), using the full `all_2_3d_leonardo` data mixture (all 18 downloaded datasets minus
   the gpuxl-only `flowbench` entry).
   ```
   python gen_scaling_runs.py   # writes scaling_run_0.sh .. scaling_run_N.sh + run_all.sh
   bash run_all.sh              # submits every generated job
   # or: sbatch scaling_run_3.sh   # submit one at a time
   ```

### Logs, checkpoints, W&B

Every script sets `++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs`, so each
run's `extended_config.yaml` snapshot and `checkpoints/` land under
`logs/<name>/<auto-incrementing-run-idx>/`. Note `<name>` is auto-decorated by Walrus
(`automatic_setup: True`) with model/optimizer descriptors, e.g. `name=smoke_test` becomes a
folder like `smoke_test-debug-delta-Isotr[Space-Adapt-]-AdamW-0.0002/`.

`WANDB_MODE=offline` is set everywhere since Leonardo's compute nodes have no outbound internet,
and `WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs` redirects wandb's local run
storage there too (its default is the current working directory, which would otherwise dump
`wandb/offline-run-*/` folders into the repo checkout under your home directory). SLURM's own
stdout/stderr go to `logs/slurm/`.

**One-time setup** (from the login node, which has internet — compute nodes don't):
```
module purge
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
wandb login   # paste an API key from https://wandb.ai/authorize, or: wandb login <KEY>
```
This writes credentials to `~/.netrc`, used by both `wandb sync` and any future online runs.

**Syncing a run** afterward:
```
wandb sync /leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/wandb/offline-run-<timestamp>-<id>
# or sync every not-yet-synced run at once:
wandb sync --sync-all /leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/wandb/
```
