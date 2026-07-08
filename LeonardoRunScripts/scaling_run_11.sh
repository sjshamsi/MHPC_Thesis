#!/bin/bash -l
#SBATCH --job-name=scale11_hidden768_depth12_mlp1536
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x-%j.err

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
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/Walrus/walrus

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py distribution=ddp server=leonardo data=smallset_leonardo \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            name=scale11_hidden768_depth12_mlp1536 trainer.grad_acc_steps=4 optimizer=adam optimizer.lr=0.0002 \
            logger.wandb_project_name="walrus_leonardo_scaling" \
            trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=100 trainer.clip_gradient=10 \
            data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \
            model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=768 model.groups=16 \
            model.processor_blocks=12 model.drop_path=0.1 \
            model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim=1536 \
            model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 \
            trainer.short_validation_length=20 trainer.max_rollout_steps=20 lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.val_frequency=5 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" trainer.max_epoch=101 data_workers=8 \
            model.override_dimensionality=0 ++trainer.skip_spectral_metrics=True auto_resume=True \
            trainer.video_validation=False
