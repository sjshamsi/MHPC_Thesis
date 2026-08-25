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
   A checkpoint should appear under `/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/smoke_test/0/checkpoints/`.

2. **`gen_star_search_runs.py`** — generator for the real architecture star search. Vary one axis
   (`model.hidden_dim`, `model.processor_blocks`, `model.processor.space_mixing.mlp_dim`) at a
   time around a base anchor (edit the `base`/`options` dicts at the top of the file to change
   the sweep), using `data=smallset_leonardo` (3 of the datasets actually downloaded under this
   cluster's `well_base_path` — a lighter mixture than tier 3's `available_leonardo`, keeping each
   of the many single-node jobs in this sweep cheap).
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

### Storage placement benchmark

Before committing to a `well_base_path` for the real dataset copy, `storage_bench.sh` +
`storage_bench_walrus.sh` empirically compare the four Leonardo storage candidates
(`leonardo_work` vs. `leonardo_scratch/fast`, under either the `ICT26_MHPC_0` or `ICT26_MHPC`
project) rather than guessing from filesystem specs alone.

1. **`storage_bench.sh`** (+ `storage_bench_io.py`) - raw sequential write/read throughput plus
   parallel random-window reads out of a real Well HDF5 file (mirroring how
   `MixedWellDataModule`'s multi-worker `DataLoader` actually samples), run from a compute node
   (not the login node) so the numbers reflect what a training job sees. Cheap: 1 GPU/8 CPUs,
   `boost_qos_dbg`, no `--exclusive`. Bench files are deleted immediately after each phase.
   ```
   ./submit.sh storage_bench.sh
   ```
2. **`storage_bench_walrus.sh`** - the authoritative check: stages a small `active_matter` subset
   (~1.5GB) under each of the four candidate paths in turn and runs the real `train.py` +
   `model=debug` (hidden_dim=8, so GPU compute is negligible and any timing gap is attributable to
   storage) for a few hundred steps each, back to back on the same GPU/node. Compare the per-batch
   `data <x>s` timing that `Trainer.run_epoch` (`walrus/trainer/training.py`) logs per path - see
   the script's header comment for the exact `grep`/`awk` to average it. Staged files are removed
   at the end of each path's block.
   ```
   ./submit.sh storage_bench_walrus.sh
   ```

Both scripts hardcode `PATHS` arrays at the top - edit those (or comment one out) rather than
templating a generator, since this is a one-off comparison, not a sweep. **As of 2026-08-24,
`cinQuota` shows `/leonardo_scratch/fast/ICT26_MHPC_0` already over its 1T quota** (grace period) -
check `cinQuota` before including that path, since these scripts still write (small, transient)
files there.

### Logs, checkpoints, W&B

Every script sets `++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs`, so each
run's `extended_config.yaml` snapshot and `checkpoints/` land under
`logs/<name>/<auto-incrementing-run-idx>/`. Note `<name>` is auto-decorated by Walrus
(`automatic_setup: True`) with model/optimizer descriptors, e.g. `name=smoke_test` becomes a
folder like `smoke_test-debug-delta-Isotr[Space-Adapt-]-AdamW-0.0002/`.

`WANDB_MODE=offline` is set everywhere since Leonardo's compute nodes have no outbound internet,
and `WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs` redirects wandb's local run
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

**Resuming vs. starting fresh under the same job name**: `configure_experiment`
(`walrus/utils/experiment_utils.py`) picks up the *latest* existing `<run-idx>` folder whenever
`auto_resume=True` and a previous run already exists under that experiment name — this is exactly
what you want when a job times out and you resubmit the same script to continue it. Training
resume follows the checkpoint in that folder as usual. W&B resume piggybacks on the same folder:
the first time an experiment folder is created, `walrus/train.py::get_or_create_wandb_run_id`
generates a W&B run id and writes it to `<run-idx>/wandb_run_id.txt`; every later launch that
resolves to that same folder reads the id back and passes it to `wandb.init(id=..., resume="allow")`,
so the new offline run reattaches to the same W&B run instead of starting a separate one. Since
`WANDB_MODE=offline` still writes a fresh local `offline-run-<timestamp>-<id>/` directory per
launch, this only takes effect once you `wandb sync` those directories (see below) — but because
they share an `id`, syncing them all with `--append` (see below) merges into one online run's
history rather than several. Sharing an `id` alone isn't enough for that history to line up
correctly, though: each launch is a fresh process, so without an explicit `step=` on every
`wandb.log()` call its internal step counter would restart at 0 every time and each resumed
launch would overwrite the previous one's history instead of extending it. `Trainer.train()`
(`walrus/trainer/training.py`) passes `step=epoch` to every `wandb.log()` call specifically so
each launch's data lands at its own non-overlapping, globally-meaningful step.

Blindly re-running the same script to start an unrelated fresh attempt (e.g. after tweaking
something that isn't part of the auto-decorated experiment name) would otherwise silently merge
into the old run's checkpoints and W&B history instead of registering as new. Every non-smoke-test
script here exposes this as an overridable `AUTO_RESUME` env var (default `True`, i.e. today's
resume behavior unchanged):
```
./submit.sh anchor_dimension_run.sh                                   # resume latest run (default)
./submit.sh --export=ALL,AUTO_RESUME=False anchor_dimension_run.sh    # force a brand-new run_idx + W&B run
```
(`smoke_test.sh` always passes `auto_resume=False` directly — a smoke test should never resume.)

**Resubmitting after a timeout**: there's no automatic requeue — when a job hits its `--time`
limit, resubmit the same script by hand, e.g. `./submit.sh --dependency=singleton anchor_dimension_run.sh`.
`--dependency=singleton` makes SLURM hold the new submission until no job with that script's
`--job-name` is still pending/running for you, so you can't accidentally double-submit while the
timed-out job is still draining; the multinode scripts already bake this into their `#SBATCH`
header, but it's equally safe to pass it as an extra `submit.sh`/`sbatch` argument for the
single-node ones. (An earlier `watch_and_resubmit.sh` poller that automated this loop has been
removed — it predated `--dependency=singleton` and had no awareness of the W&B run-id continuation
above, so a manual resubmit is now the supported path.)

**Queueing several resumptions ahead of time**: `--dependency=singleton` isn't limited to "one
resubmit after the fact" — SLURM's own semantics for it are "begin execution after *any*
previously launched jobs sharing this job name and user have terminated" (completed, failed, or
timed out all count), not just the immediately preceding one. So submitting the same script
several times in a row up front,
```
./submit.sh --dependency=singleton anchor_dimension_run.sh
./submit.sh --dependency=singleton anchor_dimension_run.sh
./submit.sh --dependency=singleton anchor_dimension_run.sh
```
serializes them: the 2nd waits for the 1st to terminate, the 3rd waits for *both* the 1st and 2nd,
which in practice means they run strictly one at a time in submission order, each auto-resuming
from whatever checkpoint the previous one left. Two things worth knowing before relying on this:
account/QOS submit limits (`sacctmgr show assoc` / `show qos` — `boost_qos_dbg` in particular caps
submitted jobs per user quite low) can reject queueing too many at once; and a queued job that
starts after training has already reached `max_epoch` isn't free — it still launches, finds
nothing left to train, and just runs the final test-validation pass before exiting, so queue
roughly the count you actually expect to need rather than padding it.

### Job submission cheat sheet: resume, fresh, or avoiding collisions

Everything above explains the *mechanism*; this is the actionable version.

**To resume a run that timed out or crashed:**
1. Resubmit the exact same script with the exact same CLI overrides, unchanged (add
   `--dependency=singleton` if you want the double-submit guard from above; several such
   submissions queued up front will run one after another automatically, see below).
2. That's it — `auto_resume=True` (the default) makes `configure_experiment` find the same
   `<run-idx>` folder, load `checkpoints/last`, and reattach to the same `wandb_run_id.txt`.
3. **Don't change anything under `model.*`** (`hidden_dim`, `processor_blocks`, `mlp_dim`,
   `num_heads`, encoder/decoder/processor choice, etc.) between submissions of the same run.
   `frozen_components` in `experiment/defaults.yaml` documents an intent to auto-import the old
   model config on resume, but only `frozen_components: [all]` is actually implemented
   (`configure_experiment` in `walrus/utils/experiment_utils.py`) — the default `[model]` value
   does **not** currently protect you, so an architecture change here will try to load a
   checkpoint into a differently-shaped model instead of erroring cleanly.
4. Increasing `trainer.max_epoch` between submissions is fine and expected (that's exactly what
   `test_resume_wandb.sh` exercises) — it's a loop bound, not part of any checkpointed shape.

**To start a genuinely new, independent run:**
- If the script exposes `AUTO_RESUME` (every hand-written script here except `smoke_test.sh`
  does): `./submit.sh --export=ALL,AUTO_RESUME=False <script>.sh`. Same `name=`, but a brand-new
  `<run-idx>` folder, fresh weights, and a fresh W&B run — nothing from the old run carries over.
- Otherwise, or as a more explicit alternative: give it a different `name=`. This always creates a
  new folder regardless of `auto_resume`, since the folder path starts from `name`.

**Avoiding accidental collisions — how the folder name actually works:** `get_experiment_name`
(`walrus/utils/experiment_utils.py`) builds the on-disk folder name as
`{name}-{data}-{prediction_type}-{model}[{encoder}-{decoder}-{processor}]-{optimizer}-{lr}` —
your `name=`, the data config's `wandb_data_name`, `trainer.prediction_type`, and the *class
names* of model/encoder/decoder/processor/optimizer plus `optimizer.lr`. Notably, it does **not**
include `model.hidden_dim`, `processor_blocks`, `mlp_dim`, `num_heads`, `batch_size`,
`trainer.max_epoch`, or basically anything else you'd typically sweep. Two consequences:
- Two runs that differ only in one of those un-encoded values (e.g. a `hidden_dim` sweep) but
  share the same `name=`/data/model-class/optimizer/lr will resolve to the **same** folder and
  silently resume into each other's checkpoint instead of running independently. Always bake
  whatever you're varying into `name=` itself if it isn't already part of the decorated name —
  exactly what `gen_star_search_runs.py` already does
  (`job_name = f"star{count}_hidden{hidden_dim}_depth{depth}_mlp{mlp}"`, passed as `name=`).
- The reverse mistake is just as easy: if you're trying to *resume* a run but tweak something
  that *is* part of the decorated name (e.g. nudge `optimizer.lr`), you won't get an error — you
  silently land in a brand-new `<run-idx>` folder with fresh weights instead of continuing.

**Testing resume + W&B continuation**: `test_resume_wandb.sh` runs the tiny `data=debug
model=debug trainer=debug` config four times back to back under one job - a fresh 1-epoch run,
then three simulated resubmissions each training one epoch further (to epoch 2, 3, then 4) - and
asserts the checkpoint resume and the `wandb_run_id.txt` reuse described above actually happened
at every step, rather than trusting it by inspection.
```
./submit.sh test_resume_wandb.sh
```
Safe to rerun any time; each invocation starts its own fresh `run_idx` (phase 1 always passes
`auto_resume=False`) so repeated test runs never collide with each other or with real experiments
(it uses a separate `wandb_project_name=walrus_leonardo_test`). Compute nodes have no outbound
internet, so the script itself can only leave behind four offline W&B run directories sharing one
run id; it prints the exact follow-up command to run from the login node afterward, scoped to just
those four runs so it doesn't also sweep in every other unsynced real run sitting in the same
shared `WANDB_DIR`:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --append \
    --include-offline --include-globs="run-<run-id>.wandb"
```
(`--sync-all` always scans `./wandb` relative to the current directory and ignores any path
argument, hence the `cd`; `--include-globs` matches the `.wandb` filename *inside* each
`offline-run-*/` directory, e.g. `run-<run-id>.wandb`, not the `offline-run-*/` directory name;
`--append` tells the server to attach each directory's history to the existing run id instead of
starting a new one each time.)
Then check the `walrus_leonardo_test` project on wandb.ai for that run id — it should show up as
a **single** run whose `train/epoch` metric rises continuously from 1 to 4, not four separate
runs. See the script's own comments for exactly what it checks.

**One-time setup** (from the login node, which has internet — compute nodes don't):
```
module purge
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
wandb login   # paste an API key from https://wandb.ai/authorize, or: wandb login <KEY>
```
This writes credentials to `~/.netrc`, used by both `wandb sync` and any future online runs.

**Syncing a run** afterward:
```
wandb sync /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/wandb/offline-run-<timestamp>-<id>
# or sync every not-yet-synced run at once - note --sync-all always scans ./wandb relative to
# the current directory and silently ignores any path argument, so cd there first:
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all
```
If a run was ever resumed (`auto_resume=True` picking up an existing `wandb_run_id.txt`, see
above), add `--append` so the later offline directories attach to the same online run instead of
each creating a separate one:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --append \
    --include-offline --include-globs="run-<run-id>.wandb"
```
