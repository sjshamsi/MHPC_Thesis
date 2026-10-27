# Pricing a WALRUS job in FLOPs: `count_WALRUS_flops.py` + the dataset registry

Give it a run script, get the job's training FLOPs, broken down by dataset and by model part,
without running anything.

Related: `walrus/utils/flops.py` (the per-micro-batch formulas, the same ones `FlopAccountant`
logs to W&B) and `Docs/flop_accounting_wandb.md` (what each logged key means).

---

## 1. The files

| file | role |
|---|---|
| `well_registry.py` | Shared library: load the registry, window counts, `encoder_shapes_from_tokens`, and `dataset_geometry()` (registry entry → the `BatchGeometry` that `flops.forward_flops` prices). |
| `build_well_registry.py` | Scans HDF5 metadata and writes `well_dataset_registry.json`. `--verify` checks the registry against real batches. |
| `count_WALRUS_flops.py` | Reads a run script, composes the Hydra config, prices the job. |
| `well_dataset_registry.json` | The registry (copied from the old repo; built 2026-10-09). |

Run from the repo root in the walrus venv. The login node can be slow to import torch; the
registry check also has a SLURM job, `Runs/verify_registry/verify_registry.sh`.

```bash
module purge; module load python/3.11.7; module load cuda/12.6
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate
```

---

## 2. The dataset registry

One JSON entry per dataset under `well_base_path` (18 on 2026-10-06, 2D and 3D, held-out
included). Per dataset it stores:

- resolution, `n_spatial_dims`, `grid_type`, per-axis BCs (from the_well's own reader);
- native field names, and the model-format field list after 2D→3D padding (fields, then
  constants), whose length is `n_channels_in`, the channel count the encoder actually sees
  (the 3 BC flags come on top);
- token grid (32 per axis in 2D, 16 in 3D, +1 on each non-periodic axis);
- per split: files, bytes on disk, and `(n_steps, n_traj)` groups.

Window counts are **not** stored; they depend on `n_steps_input` / `n_steps_output` /
`min_dt_stride` and are derived when pricing.

```bash
python Tools/flop_planner/build_well_registry.py                    # full rescan, ~3 min
python Tools/flop_planner/build_well_registry.py --datasets MHD_64  # rescan one, keep the others
python Tools/flop_planner/build_well_registry.py --path /some/dir   # an external Well-format dataset
python Tools/flop_planner/build_well_registry.py --verify-only      # check, ~2 min for all 18
python Tools/flop_planner/build_well_registry.py --csv out.csv      # also a flat summary table
```

**`--verify`** loads one real window per dataset through the training pipeline (datamodule →
formatter → a tiny `IsotropicModel`), lets `FlopAccountant` observe the geometry, and checks it
equals what the registry implies (channels, padded encoder input, strides, periodic axes, token
grid), and that both give identical `forward_flops` dicts (every granular term, not just the
total). Axis order is ignored (`RandomRotation90` permutes axes), and so is the BC flag of a
size-1 axis (it never affects cost). **All 18 passed** on the old repo (2026-10-06). Rerun it
after every rebuild.

Rebuild when datasets are downloaded or deleted. The registry header records `built_at`, `root`
and the git SHA.

---

## 3. Pricing a job

```bash
python Tools/flop_planner/count_WALRUS_flops.py <script>
python Tools/flop_planner/count_WALRUS_flops.py <script> --exact           # replay the sampler's dataset draws
python Tools/flop_planner/count_WALRUS_flops.py <script> -H 1408 -L 40 -M 5632 --nodes 4   # what-if
python Tools/flop_planner/count_WALRUS_flops.py <script> --start-epoch 1 --end-epoch 30    # a slice
python Tools/flop_planner/count_WALRUS_flops.py <script> --json
python Tools/flop_planner/count_WALRUS_flops.py --data Leonardo_pretrain9_2d --override trainer.max_epoch=10
```

What it does:

1. Takes the Hydra overrides after `train.py` in the script (shell placeholders `$AUTO_RESUME`,
   `$STAGE_DIR` substituted, `hydra.*` dropped, `server` pinned to `leonardo`) and the
   `#SBATCH --nodes` / `--gpus-per-node`. CLI flags are applied on top; for any key the last
   value wins.
2. Builds the model on the meta device to read its shape (`flops.ModelShape.from_model`) and
   parameter count.
3. For each dataset in the data config: effective batch and `n_steps_input` after the
   per-dataset downsample factors (same flooring as the code), geometry from the registry, then
   `forward_flops` and `training_step_flops`.
4. Counts micro-batches per dataset over epochs 1..`max_epoch` (all ranks):
   - **default**: `max_samples × world_size × epochs / K` each (the sampler is uniform over
     datasets);
   - **`--exact`**: replays `BatchedMultisetSampler`'s dataset-choice generator
     (seed `100·epoch + sampling_rank`), one stream per distinct sampling rank: 1 under `fsdp`,
     one per node under `hsdp`, one per GPU under `ddp`. It models sub-sampler exhaustion and
     recycling from the registry's window counts. `local` runs reuse epoch 0's sequence every
     epoch, as the code does.

FLOP basis: forward + backward (backward = 2 × forward), exactly as `FlopAccountant` logs, with
gradient-checkpointing recompute shown separately. The planner's "total" is measured
(useful + recompute); W&B's `cumulative_flops` is the useful part only. Encoder / processor /
decoder splits are sums of the granular terms (`flops.stage_totals`). **Validation is not
included.**

If the data config names a dataset that isn't in the registry, it stops and prints the builder
command.

---

## 4. Worked examples (2026-10-06, old repo; run scripts not yet ported)

### PT9 noise floor, seed 0 (`768/12/2048`, `fsdp`, 1 node, 100 epochs)

```
total training FLOPs   7.072 EF   (recompute 0)
  encoder / processor / decoder   356 PF / 6.386 EF / 329 PF   (enc+dec 9.7 % of forward)
per epoch              70.7 PF          per optimizer step   141 TF
micro-batches          800,000          tokens seen          10.18 G (101.8 M/epoch)
unique model tokens    3.26 G           repetition           3.12x
```

Per dataset: 8.4–9.3 TF per micro-batch; enc+dec 8.7–12.0 % of forward; repetition from 1.11×
(gray_scott) to 11.0× (planetswe). These reproduce `SCALING_STRATEGY_2D.md` §2.5 and §3.1.
(§2.1 used to list planetswe with 4 channels, giving 9.3 % in §3.1; the pipeline gives **3**,
confirmed by `--verify`, and both tables were corrected on 2026-10-06.)

`--exact` gives 7.073 EF, within 0.02 % of the expectation; per dataset the replay deviates by
up to ±1.3 %.

### Upstream WALRUS pretraining (`1408/40/5632`, `hsdp`, 24 nodes, 201 epochs)

Run with `--override '~data.module_parameters.well_dataset_info.flowbench_FPO_NS_2D_512x128_harmonics'`,
since that external dataset isn't on disk here.

```
total training FLOPs   4873 EF  = 4159 EF useful + 714 EF recompute (gradient_checkpointing_freq=2)
micro-batches          38.6 M   (the paper reports "38M micro-batches")
tokens seen            500 G    repetition 125x (27x gray_scott ... 3840x viscoelastic)
```

3D micro-batches cost 146–182 TF vs ~107–115 TF for 2D, at a similar token count; that is the
attention cost of 4096-token frames. `--exact` (24 streams × 201 epochs, ~1 min) agrees to
0.004 %.

---

## 5. Caveats

- `walrus/utils/flops.py::unique_model_tokens` reads per-axis BCs from each dataset's first file
  (`_dataset_periodic_axes`) and matches the registry on all 18 datasets. If a file can't be
  read, training logs a warning and omits `repetition_factor` (no guess from the set of BC
  types, which can't say which axis is which). FLOPs and `tokens_seen` don't depend on this.
- `n_states` only sizes the stem's weights, never its FLOPs. With the 3-entry `bc_only_override`
  the script assumes 67 for the parameter count.
- `restrict_train_num_*` is ignored (warning printed): windows and unique tokens use the full
  train split.
- `--exact` mirrors `mixed_dset_sampler.py:167-210` and `training.py:358-376` at the time of
  writing, including upstream's sampler seed `100·epoch + 10·seed + rank`; if those change, the
  default (expected) mode is unaffected. The old repo's `flopcount` branch changed that seed
  (seeding work), so its `--exact` counts differ from this one by up to ~1 % per dataset (more
  for short runs); compare against its `flopsonly` branch instead.
- What-if shapes must still build: the planner instantiates the model on the meta device, so
  e.g. `-H 512` with the default `model.groups` fails with `num_channels must be divisible by
  num_groups`. Pass the run's `model.groups` / `num_heads` too.

---

## 6. Checks (new repo, 2026-10-27)

Every check below passed. The planner was run from the repo root in `env2`; the "old repo"
runs used `env1` with the old repo's `flopsonly` planner (`git show flopsonly:scripts/...`) or,
where noted, its `flopcount` working tree.

| # | what it checks | how | result |
|---|---|---|---|
| 1 | default mode, unchanged by the port (granular-only `flops.py`, `stage_totals`) | PT9 mix, `gc_freq=2`, 2 epochs, `--override ... --json`, new vs old planner | every job and per-dataset number identical (≤ 1e-16) |
| 2 | `--exact` against **real training**: hooks on FSDP-wrapped encoders, sum over 4 GPUs, resume | `sbatch Runs/test_flops_fsdp/test_flops_fsdp.sh` (`fsdp`, 1 node × 4 GPUs, PT9 mix, 256/4/512, `gc_freq=2`, epochs 1–2 then a resume for 3–4); `check_flops_fsdp.py` calls this planner with `--exact --start-epoch e --end-epoch e` | logged useful FLOPs, recompute, tokens, encoder/processor/decoder, `enc_dec_flop_fraction`, `flops_per_microbatch`, `repetition_factor` equal the planner to 1e-9 every epoch; **ALL PASS (63 checks)**, run twice (SLURM 62458062, 62465325) |
| 3 | reading a real SLURM script (`parse_run_script`), `hsdp`, 24 nodes, 3D data, 201 epochs | `walrus/run_scripts/pretrain_example_distributed_walrus.sh --override '~data.module_parameters.well_dataset_info.flowbench_FPO_NS_2D_512x128_harmonics' --json`, with and without `--exact`, new vs old | identical to the old planner (default mode: vs `flopcount`; `--exact`: vs `flopsonly`); **38,592,000 micro-batches** (the paper reports ~38 M); 4873 EF = 4159 useful + 714 recompute; 4.997e11 tokens; repetition 125×; expected vs `--exact` total differ by 4.3e-5 |
| 4 | `ddp` across 2 nodes (8 sampler streams) and the what-if flags | `--data Leonardo_pretrain9_2d --distribution ddp --nodes 2 --gpus-per-node 4 -H 512 -L 8 -M 1024 --batch-size 4 --max-samples 100 --grad-acc-steps 2 --epochs 3 --exact --override data/field_index_map_override=full_well_field_index model/processor/space_mixing=full_spatial_attention model.gradient_checkpointing_freq=2 model.groups=16 model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 --json`, new vs `flopsonly` | identical (0 difference): 2400 micro-batches, 1.017e8 tokens, 24.8 PF = 21.0 useful + 3.8 recompute |
| 5 | the registry against the real model | `sbatch Runs/verify_registry/verify_registry.sh` (`build_well_registry.py --verify-only`) | all 18 datasets `ok` / `exact`: geometry and every granular `forward_flops` term |
| 6 | unit tests, incl. `encoder_shapes_from_tokens` round trips and the token-grid rule | `sbatch Runs/test_flops/test_flops.sh` | 26 / 26 passed |

Against the old `flopcount` working tree, checks 3 (`--exact`) and 4 differ by up to 0.7 % and
11 % per dataset (job totals ≤ 0.5 %). That is its different sampler seed (§5), not a pricing
difference: default mode and the `flopsonly` comparison are identical.

**Not yet checked:** `--start-epoch` slices other than one epoch at a time; the
`restrict_train_num_*` warning; `hsdp` against real training (only the replay, check 3).
