#!/bin/bash -l
#SBATCH --job-name=halfDims_activematter_tiny
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
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.err

# Half-dimension counterpart to fullDims_activematter_tiny.sh: same
# data=activematter_leonardo and same shrunk max_samples=50/max_epoch=6/lr_scheduler
# ramps=2+2 (sized to fit a real train+valid+rollout-valid cycle in the boost_qos_dbg
# 30-minute slot), but at HalfWalrus sizing - hidden_dim=1408->1088,
# processor_blocks=40->30, model.processor.space_mixing.mlp_dim set explicitly to 4352
# (= hidden_dim*4), matching halfDims_allDsets_tiny.sh. With a smaller processor and only
# one (lighter, 2D) dataset, this should run faster per epoch than either
# fullDims_activematter_tiny.sh or halfDims_allDsets_tiny.sh - check this run's own
# timing before deciding whether to raise max_epoch.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs
# True (default) continues the latest run under this name - correct for resubmitting
# after a timeout/crash. Set to False (e.g.
# `./submit.sh --export=ALL,AUTO_RESUME=False halfDims_activematter_tiny.sh`) to force a
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
            data=activematter_leonardo \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            name=halfDims_activematter_tiny \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="halfDims_activematter_tiny" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
            trainer.log_interval=10 \
            trainer.clip_gradient=10 \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.hidden_dim=1088 \
            model.groups=16 \
            model.processor_blocks=30 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.mlp_dim=4352 \
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
