#!/bin/bash -l
#SBATCH --job-name=fullDims_FT_ActiveMatter
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.err

# Loads the released pretrained Walrus checkpoint (walrus_pretrained/walrus.pt, downloaded
# via ../scripts/download_walrus_checkpoint.sh) and finetunes it on a single dataset
# (active_matter). Model architecture below matches the checkpoint's own model.* section
# exactly (verified: 857/857 state_dict keys match, 0 shape mismatches), since
# model.load_state_dict(..., strict=True) needs shapes to line up on load.
#
# Runs under FSDP+gradient_checkpointing_freq=2, and validation/test both run end to end).
# max_samples/max_epoch/warmup/cooldown/rollout settings below are sized from that smoke
# test's *measured* per-phase timing, not guessed, targeting ~70% of the 2h budget so
# there's real margin left over:
#   - training: ~2.5s/microbatch (measured: 50 microbatches took 116-135s across its
#     two epochs) -> max_samples=150 costs ~375s/epoch.
#   - one-step valid: ~1s/batch on a short (short_validation_length-capped) pass, but
#     ~0.66s/batch x 225 batches (~148s) on the final epoch, which always runs the FULL
#     valid set regardless of short_validation_length (see
#     walrus/trainer/training.py::validate_if_necessary's `epoch >= self.max_epoch`
#     check) - cheap either way relative to training.
#   - rollout valid: this active_matter split only has 6 full-trajectory-mode batches
#     eligible for rollout validation regardless of short_validation_length, so its cost
#     is ~fixed per occurrence and scales only with max_rollout_steps (measured ~8.5s/
#     batch at max_rollout_steps=20 in the smoke test). Bumped max_rollout_steps to 60
#     here (matching fullDims_FSDP_GC1.sh's pretraining-scale value) to actually exercise
#     long-horizon rollout stability instead of the smoke test's speed-only 20 steps -
#     since that triples this cost per occurrence (~153s), rollout_val_frequency is
#     loosened to 2 (was 1) to keep its *recurring* cost down; val_frequency stays at 1
#     since one-step valid is cheap.
#   - the final one-step test (~161s) and final rollout test (~176s at max_rollout_steps=
#     60) each run exactly once regardless of max_epoch (train.py runs a full test pass
#     after the epoch loop unconditionally), so they're fixed costs, not per-epoch ones.
#   - lr_scheduler=inv_sqrt_w_sqrt_ramps defaults to warmup_epochs=10, cooldown_epochs=10
#     (see walrus/configs/lr_scheduler/inv_sqrt_w_sqrt_ramps.yaml) - with max_epoch=10
#     that would be 100% warmup+cooldown and 0 epochs of actual cruise/decay, so both are
#     overridden down to 3 here, leaving 4 cruise epochs.
#   - total: startup(~110s) + 10*375s(train) + [9*10s + 148s](one-step valid) +
#     5*153s(rollout valid, epochs 2/4/6/8/10) + 161s(final test) + 176s(final rollout
#     test) + ~20s(overhead) ~= 5220s (~87min) of a 7200s (2h) allocation - ~27% margin.
#
# Hydra overrides changed here relative to the pretraining baseline and why:
#   - data/field_index_map_override=full_well_field_index (new): byte-identical to the
#     checkpoint's own field->index map, so checkpoint.align_fields (below) becomes a
#     pure identity copy for every field active_matter actually has - no embedding
#     rescaling needed.
#   - optimizer.lr=1.e-4 (was 2.e-4): finetuning a converged checkpoint should use a
#     lower LR than pretraining-from-scratch used.
#   - experiment=finetune_leonardo (was defaults): sets finetune=True (tells train.py
#     this is loading a pretrained model into a new run, not resuming one in place) and
#     sets config_override to walrus_pretrained/extended_config.yaml, which points
#     configure_experiment's old_field_index_map lookup at the checkpoint's own config,
#     so the field-alignment logic above knows what field->index mapping the checkpoint
#     was actually trained with. (Leonardo-local variant of upstream's
#     experiment=finetune_example, which points config_override at a Flatiron-only path.)
#   - checkpoint=finetune_leonardo (was defaults): the checkpoint group with
#     coalesced_checkpoint_path pre-filled to walrus_pretrained/walrus.pt - the actual
#     pretrained weights - train.py::load_from_coalesced_checkpoint loads this
#     state_dict into the model before any training happens. (Leonardo-local variant of
#     upstream's checkpoint=finetune, which points at a Flatiron-only path; "defaults"
#     has no coalesced_checkpoint_path at all.)
#   - finetuning_mods=all (was defaults, a no-op group): makes the RoPE coefficients
#     learnable during finetuning, matching Walrus's own canonical finetuning recipe
#     (walrus/run_scripts/finetuning_example_distributed_walrus.sh).
#   - trainer.log_interval=10 (was 100): log more often since max_epoch/max_samples are
#     tiny for this smoke test, so a log_interval of 100 would never fire.
#   - model.* architecture (projection_dim, intermediate_dim, hidden_dim, groups,
#     processor_blocks, drop_path, space_mixing, num_heads x2, causal_in_time,
#     jitter_patches, use_periodic_fixed_jitter, input_field_drop): UNCHANGED from
#     fullDims_FSDP_GC1.sh on purpose - must match walrus_pretrained/walrus.pt's shapes
#     exactly for the strict=True state_dict load to succeed.
#   - data.module_parameters.max_samples=150 (was 2000 in fullDims_FSDP_GC1.sh): sized
#     per the timing derivation above, not the pretraining-scale value - see that
#     comment block.
#   - trainer.short_validation_length=10 (was 20): kept from the smoke test - cheap
#     either way relative to training, see timing derivation above.
#   - trainer.max_rollout_steps=60 (was 20 in the smoke test, matches
#     fullDims_FSDP_GC1.sh's own value): see timing derivation above.
#   - trainer.val_frequency=1 (unchanged from the smoke test): one-step valid is cheap,
#     validate every epoch.
#   - trainer.rollout_val_frequency=2 (was 1 in the smoke test): rollout valid is
#     expensive at max_rollout_steps=60, see timing derivation above.
#   - lr_scheduler.warmup_epochs=3 and lr_scheduler.cooldown_epochs=3 (new; group
#     default is 10/10): see timing derivation above.
#   - trainer.max_epoch=10 (was 2 in the smoke test, 20 in fullDims_FSDP_GC1.sh): sized
#     per the timing derivation above.
#   - checkpoint=defaults/experiment=defaults/finetuning_mods=defaults from
#     fullDims_FSDP_GC1.sh are replaced by the checkpoint=finetune_leonardo/experiment=
#     finetune_leonardo/finetuning_mods=all entries above - all three are part of the
#     same "this is a finetune run" change, not independent tweaks.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs
# `sbatch --export=ALL,AUTO_RESUME=False fullDims_FT_ActiveMatter.sh`) to force a
# brand-new run_idx folder and a brand-new wandb run for a deliberate
# attempt under the same job name.
export AUTO_RESUME=${AUTO_RESUME:-True}

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py \
            distribution=fsdp \
            server=leonardo \
            data=activematter_leonardo \
            data/field_index_map_override=full_well_field_index \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            name=fullDims_FT_ActiveMatter \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=1.e-4 \
            logger.wandb_project_name="fullDims_FT_ActiveMatter" \
            experiment=finetune_leonardo \
            checkpoint=finetune_leonardo \
            finetuning_mods=all \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
            trainer.log_interval=10 \
            trainer.clip_gradient=10 \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.hidden_dim=1408 \
            model.groups=16 \
            model.processor_blocks=40 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.num_heads=16 \
            model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True \
            model.jitter_patches=True \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0.0 \
            data.module_parameters.max_samples=150 \
            trainer.short_validation_length=10 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            lr_scheduler.warmup_epochs=3 \
            lr_scheduler.cooldown_epochs=3 \
            trainer.val_frequency=1 \
            trainer.rollout_val_frequency=2 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            trainer.max_epoch=10 \
            data_workers=8 \
            model.override_dimensionality=0 \
            ++trainer.skip_spectral_metrics=True \
            auto_resume=$AUTO_RESUME \
            trainer.video_validation=False
