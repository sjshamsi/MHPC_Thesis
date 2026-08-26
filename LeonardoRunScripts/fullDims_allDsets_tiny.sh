#!/bin/bash -l
#SBATCH --job-name=fullDims_allDsets_tiny
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Tiny/fast variant of fullDims_allDsets.sh for quick iteration on the debug queue -
# same paper-scale architecture and same data=Leonardo_smallest8_2_3d (all 6 datasets), so
# this is still a genuine test of that configuration, just sized to run in minutes
# instead of hours. Two independent things had to shrink together, not just one:
#   - data.module_parameters.max_samples=50 (was 2000): this is what sets wall-clock
#     cost *per epoch*, independent of architecture - shrinking it is what makes many
#     more epochs affordable in a short queue slot.
#   - lr_scheduler.warmup_epochs=2 / cooldown_epochs=2 (group defaults are 10/10):
#     fullDims_allDsets.sh's own max_epoch=71 only works with the *default* 10/10
#     ramps because it can afford 71 (expensive) epochs. A tiny run can't afford that
#     many epochs, so the ramps must shrink to match whatever max_epoch it settles on
#     instead - otherwise a small max_epoch is all warmup+cooldown with zero actual
#     decay phase (the exact problem this file exists to sidestep).
#   - trainer.max_epoch=6: leaves 6-2-2=2 epochs of actual cruise/decay - modest but
#     non-zero. This is a smoke/calibration test, not a real training budget.
#   - trainer.short_validation_length=10, trainer.max_rollout_steps=20,
#     trainer.val_frequency=1, trainer.rollout_val_frequency=1: shrunk/frequent for the
#     same reason as max_samples - keep every phase cheap and visible within a few
#     epochs, matching the pattern already used for the smoke-test-scale finetuning
#     runs in this directory.
#   - trainer.log_interval=10 (was 100): with max_samples=50, a log_interval of 100
#     would never fire.
#
# Everything else (model.* architecture, data=Leonardo_smallest8_2_3d,
# model.gradient_checkpointing_freq=0, grad_acc_steps, optimizer, checkpoint/experiment/
# finetuning_mods=defaults) is unchanged from fullDims_allDsets.sh on purpose.
#
# No measured timing for this exact combination (full architecture x all 6 datasets x
# gradient_checkpointing_freq=0) exists yet - the closest data point is the finetuning
# smoke test (job 50566497: ~2.5s/microbatch, but that was gradient_checkpointing_freq=2
# on a single 2D dataset). gradient_checkpointing_freq=0 should be faster per step (no
# recompute) but uses more memory, and this mixture includes 3D data (MHD_64) which
# likely costs more per step than 2D. This run's own log is the real calibration data -
# treat it the same way the original smoke_test.sh was treated: confirm it completes
# and look at its actual per-phase timing before sizing a real fullDims_allDsets.sh
# budget (max_epoch=71 as currently set may still be far more than a single 24h
# submission can afford - check this run's numbers before trusting that value).

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
# True (default) continues the latest run under this name - correct for resubmitting
# after a timeout/crash. Set to False (e.g.
# `./submit.sh --export=ALL,AUTO_RESUME=False fullDims_allDsets_tiny.sh`) to force a
# brand-new run_idx folder and a brand-new wandb run for a deliberate fresh attempt
# under the same job name.
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
            data=Leonardo_smallest8_2_3d \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            name=fullDims_allDsets_tiny \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="fullDims_allDsets_tiny" \
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
            data.module_parameters.max_samples=50 \
            trainer.short_validation_length=10 \
            trainer.max_rollout_steps=20 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            lr_scheduler.warmup_epochs=2 \
            lr_scheduler.cooldown_epochs=2 \
            trainer.val_frequency=1 \
            trainer.rollout_val_frequency=1 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            trainer.max_epoch=6 \
            data_workers=8 \
            model.override_dimensionality=0 \
            ++trainer.skip_spectral_metrics=True \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            finetuning_mods=defaults \
            trainer.video_validation=False
