#!/bin/bash -l
#SBATCH --job-name=fullDims_6DS_N8
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
export AUTO_RESUME=${AUTO_RESUME:-True}

module purge
module load python
module load cuda/12.2
export PATH="/leonardo/home/userexternal/sshamsi0/.local/opt/ffmpeg-9.0.1/bin:$PATH"
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

# No need to specify model=isotropic_model but we do
# trainer.val_frequency=2 from 10
# trainer.rollout_val_frequency=5 from 10
# trainer.max_epoch=71 not 201 because only 6 datasets

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py \
            distribution=fsdp \
            model=isotropic_model \
            name=fullDims_6DS_N8 \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            server=leonardo \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="fullDims_6DS" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
            trainer.log_interval=200 \
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
            data.module_parameters.max_samples=2000 \
            trainer.short_validation_length=20 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.val_frequency=2 \
            trainer.rollout_val_frequency=5 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            data=available_leonardo \
            trainer.max_epoch=71 \
            data_workers=10 \
            model.override_dimensionality=0 \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0.0 \
            ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            trainer.video_validation=True