## Walrus architecture-scaling runs on Leonardo

Mirrors the pattern in `../ExampleRunScripts` (Hydra CLI overrides templated into SLURM
scripts), retargeted at CINECA Leonardo (`boost_usr_prod`, account `ICT26_MHPC_0`, 4x A100/node)
and the `env1` Python environment.

### Environment

`env1` (`/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1`) is a plain venv (no
`--system-site-packages`) built from a pyenv-managed Python 3.11.9, with `torch==2.5.1+cu121`,
`torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`, and the rest of `pyproject.toml`'s
dependencies, plus `walrus` installed editable (`pip install -e /leonardo/home/userexternal/sshamsi0/MHPC_Thesis`)
so source edits under this repo's `walrus/` take effect immediately. (This editable install used
to point at a since-removed `~/Walrus` checkout — if `import walrus` ever fails again with
`ModuleNotFoundError`, re-run that `pip install -e` from the repo root.) Every script here does
`module purge; module load python; module load cuda/12.2`
before activating it — `module purge` is required because the login/compute node's default
profile auto-loads `cineca-ai/4.3.0`, which injects an older `torch` (2.2.0a0) onto `PYTHONPATH`
ahead of the venv and silently shadows it otherwise.

### Three tiers

1. **`smoke_test.sh`** — hand-written, not generated. Tiny model + tiny data
   (`data=debug model=debug`, `active_matter`, `max_samples=8`), `boost_qos_dbg` (max 30 min).
   Run this first to confirm the whole chain works: modules → venv → torchrun → data loading from
   our downloaded dataset path → training → checkpoint → logging.
   ```
   ./submit.sh smoke_test.sh
   ```
   A checkpoint should appear under `/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/smoke_test/0/checkpoints/`.

2. **`gen_star_search_runs.py`** — generator for the real architecture star search. Vary one axis
   (`model.hidden_dim`, `model.processor_blocks`, `model.processor.space_mixing.mlp_dim`) at a
   time around a base anchor (edit the `base`/`options` dicts at the top of the file to change
   the sweep), using `data=smallset_leonardo` (3 of the datasets actually downloaded on this
   cluster's scratch — a lighter mixture than tier 3's `available_leonardo`, keeping each of the
   many single-node jobs in this sweep cheap).
   ```
   python gen_star_search_runs.py   # writes star_search_run_0.sh .. star_search_run_N.sh + run_star_search_all.sh
   bash run_star_search_all.sh      # submits every generated job
   # or: sbatch star_search_run_3.sh   # submit one at a time
   ```

3. **`full_dimension_run.sh`** — hand-written baseline the dimension sweep gets compared
   against: Walrus's *original*, unmodified paper-scale architecture (`hidden_dim=1408`,
   `processor_blocks=40`, `projection_dim=48`, `intermediate_dim=352`, `groups=16` — copied
   verbatim from `../walrus/run_scripts/pretrain_example_distributed_walrus.sh`, not the smaller
   anchor `gen_star_search_runs.py` sweeps around), trained on `data=available_leonardo` — a new data
   config (`../walrus/configs/data/available_leonardo.yaml`) listing only the 6 datasets actually
   downloaded under `well_base_path` here (`active_matter`, `gray_scott_reaction_diffusion`,
   `helmholtz_staircase`, `turbulent_radiative_layer_2D`, `viscoelastic_instability`, `MHD_64`),
   as opposed to `all_2_3d_leonardo`'s full 18-dataset list (most of which isn't downloaded here
   and would fail at data-loading time) or `smallset_leonardo`'s 3-dataset subset. Runs on 1 node
   with `distribution=fsdp` — the original script used 24 nodes with `distribution=hsdp` and
   `local_size=4`, meaning parameters were only ever sharded within a single node's 4 GPUs and
   merely replicated data-parallel-style across nodes, so a single-node FSDP job gives each GPU
   the same memory footprint as the original, just less data-parallel throughput.
   ```
   ./submit.sh full_dimension_run.sh
   ```
   24h is the max single job on `boost_usr_prod`/`normal`; `auto_resume=True` plus periodic
   checkpointing mean re-submitting the same script picks up from the last checkpoint if
   `max_epoch=101` isn't reached in one submission. See "Logs, checkpoints, W&B" below for how to
   deliberately start a fresh attempt instead of resuming.

4. **`full_dimension_run_multinode.sh`** — same architecture/data as tier 3, spread across
   `--nodes=4` (16 GPUs) with `distribution=hsdp` instead of `fsdp` (matching how the original
   24-node script actually sharded: within-node only, replicated across nodes) and
   `--dependency=singleton` to guard against overlapping resubmissions. Change the node count by
   editing the single `--nodes=` line — `--nnodes`/`--nproc_per_node` in the `torchrun` call
   already derive from SLURM env vars and adapt automatically. Note this does **not** make a
   fixed `max_epoch` finish faster: `data.module_parameters.max_samples` caps micro-batches
   *per rank* (see `walrus/data/mixed_dset_sampler.py`'s `BatchedMultisetSampler`), so
   steps-per-epoch is fixed regardless of node count. More nodes instead gives a bigger
   effective global batch per optimizer step (`batch_size * grad_acc_steps * world_size`), at
   the cost of burning GPU-hours proportionally faster for the same wall-clock duration.
   ```
   sbatch full_dimension_run_multinode.sh
   ```

5. **`anchor_dimension_run.sh`** — same idea as tier 3, but at the *sweep anchor* architecture
   (`hidden_dim=768`, `processor_blocks=12`, `mlp_dim=2048` — `gen_star_search_runs.py`'s `base` dict)
   instead of the paper-scale one. That anchor point is never generated as its own
   `star_search_run_N.sh` (the sweep's `options` lists all exclude the anchor's own value), so this
   materializes it, trained on `data=available_leonardo` (all 6 datasets) rather than the sweep's
   `data=smallset_leonardo` (3 datasets). At ~203M params (vs ~1.29B at paper scale) it needs
   none of tiers 3/4's FSDP/HSDP sharding — `distribution=ddp` and `gradient_checkpointing_freq=1`
   are enough, matching `gen_star_search_runs.py`'s own template.
   ```
   ./submit.sh anchor_dimension_run.sh
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
`wandb/offline-run-*/` folders into the repo checkout under your home directory).

**SLURM stdout/stderr** go to `logs/slurm/<job_name>/<job_id>.out`/`.err` — grouped in a
subdirectory per `--job-name` rather than one flat pile, so all the SLURM output for a given
named experiment (e.g. every resubmission of `walrus_full_dimension` across its 24h chunks) sorts
together. SLURM does **not** create that `<job_name>` subdirectory itself, so submit hand-written
scripts through the `./submit.sh` wrapper (e.g. `./submit.sh anchor_dimension_run.sh`) instead of
calling `sbatch` directly — it `mkdir -p`s the log directory first, then forwards to `sbatch`.
Generated scripts (`star_search_run_N.sh`, `tuning_run_N.sh`) don't need this: their generators
(`gen_star_search_runs.py`, `gen_tuning_scripts.py`) already create the directory when they write
each script, so plain `sbatch star_search_run_3.sh` / `bash run_star_search_all.sh` is fine.

**Tracing a checkpoint back to its SLURM job(s)**: `walrus/train.py::main` appends the
`SLURM_JOB_ID` of every job that has written to a given experiment folder into
`logs/<name>/<run-idx>/slurm_job_ids.txt`. Since `auto_resume=True` (see below) means a single
logical run can span several 24h SLURM submissions, this file lets you go from a checkpoint
directory to the exact `logs/slurm/<job_name>/<job_id>.out` for every job that contributed to it.

**Resuming vs. starting fresh under the same job name**: `configure_experiment`
(`walrus/utils/experiment_utils.py`) picks up the *latest* existing `<run-idx>` folder (and,
transitively, the same `wandb_run_id.txt` → same W&B run) whenever `auto_resume=True` and a
previous run already exists under that experiment name — this is exactly what you want when a job
times out and you resubmit the same script to continue it. But it means blindly re-running the
same script to start an unrelated fresh attempt (e.g. after tweaking something that isn't part of
the auto-decorated experiment name) silently merges into the old run's checkpoints and W&B history
instead of registering as new. Every non-smoke-test script here now exposes this as an
overridable `AUTO_RESUME` env var (default `True`, i.e. today's resume behavior unchanged):
```
./submit.sh anchor_dimension_run.sh                                   # resume latest run (default)
./submit.sh --export=ALL,AUTO_RESUME=False anchor_dimension_run.sh    # force a brand-new run_idx + W&B run
```
(`smoke_test.sh` always passes `auto_resume=False` directly — a smoke test should never resume.)

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
