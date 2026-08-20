#!/bin/bash -l
#SBATCH --job-name=torchprofile_fullDims_6DS
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

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs

module purge
module load python
module load cuda/12.2
export PATH="/leonardo/home/userexternal/sshamsi0/.local/opt/ffmpeg-9.0.1/bin:$PATH"
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

# torch.profiler dry run of the fullDims_6DS architecture/data/distribution config, for
# the "use torch.profiler first" pass of the training-speed investigation (see
# LeonardoRunScripts/README.md and the walrus/trainer/training.py profiler_* Trainer
# args added on this branch). Everything below is copied verbatim from
# ../fullDims_6DS.sh (same model, same data=available_leonardo, same distribution=fsdp,
# same batch_size/grad_acc_steps/gradient_checkpointing_freq) EXCEPT:
#   - name/logger.wandb_project_name: distinct, so this doesn't share fullDims_6DS's
#     experiment folder/auto-resume state or wandb run.
#   - trainer.max_epoch=2, data.module_parameters.max_samples=30: 71 epochs x 2000
#     microbatches was sized for a real 12h run; 2 epochs x 30 microbatches is plenty to
#     get past the profiler's wait+warmup+active window (see ++trainer.profiler_*
#     below) while keeping the training portion of this job seconds, not hours.
#   - trainer.val_frequency=1, trainer.rollout_val_frequency=1 (from 2/5): with
#     max_epoch=2 these no longer divide epoch 1, so without this change epoch 1 would
#     skip validation entirely and the *first* validation pass profiled would be epoch
#     2's - which, being the final epoch, is forced to run FULL validation (entire
#     val/rollout-val set, all 6 datasets - see `full=(epoch >= self.max_epoch ...)` in
#     Trainer.validate_if_necessary). Validating every epoch instead means epoch 1 does
#     the cheap, short_validation_length-capped validation, and that's the one the
#     profiler actually captures (see profiler_val logic in Trainer.validation_loop).
#   - trainer.log_interval=5 (from 200): 200 would never fire in a 30-microbatch epoch.
#   - auto_resume=False: this is a disposable debug run, not something to resume.
#   - ++trainer.profiler_enabled=True plus the profiler_wait/warmup/active/with_stack
#     knobs: see "Where results end up" in the PR/commit description for what these
#     produce and where.
#
# NOTE: epoch 2 (the final epoch) still triggers Trainer's unconditional full
# validation, AND Trainer.train() always runs one more full one-step + rollout pass
# over the *test* split after the epoch loop regardless of max_epoch - this is existing
# behavior, not something this script adds, and is why --time is 2h rather than a
# debug-queue-sized 20-30 min despite the training portion itself being very short. If
# you only care about the training-step trace and want to kill the job the moment it
# appears, watch for "torch.profiler enabled for 'train'" then "torch.profiler enabled
# for 'valid'" in the .out log (see below) and scancel once both trace files show up
# under viz/profiler_traces/.

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py \
            distribution=fsdp \
            model=isotropic_model \
            name=torchprofile_fullDims_6DS \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            server=leonardo \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="torchprofile_fullDims_6DS" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
            trainer.log_interval=5 \
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
            data.module_parameters.max_samples=30 \
            trainer.short_validation_length=20 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.val_frequency=1 \
            trainer.rollout_val_frequency=1 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            data=available_leonardo \
            trainer.max_epoch=2 \
            data_workers=10 \
            model.override_dimensionality=0 \
            auto_resume=False \
            checkpoint=defaults \
            experiment=defaults \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0.0 \
            ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            trainer.video_validation=True \
            ++trainer.profiler_enabled=True \
            ++trainer.profiler_wait=5 \
            ++trainer.profiler_warmup=3 \
            ++trainer.profiler_active=5 \
            ++trainer.profiler_with_stack=True
