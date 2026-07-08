#!/bin/bash -l
#SBATCH --time=168:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J grid55_lr0.0002_8_hidden1088_mlp_512 
#SBATCH --output=/mnt/home/polymathic/ceph/MPPX_logging/runs/logs/slurmid%j--grid55_lr0.0002_8_hidden1088_mlp_512.log
#SBATCH -C h100

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=torch_worker_log.json
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch the training script
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Launch the training script

source /mnt/home/pmukhopadhyay/projects/multiple_physics_pretraining/myenv3.10/bin/activate

srun python -u `which torchrun` \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--nproc_per_node=$SLURM_GPUS_PER_NODE \
		--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
			train.py distribution=ddp ++experiment_dir=/mnt/home/polymathic/ceph/MPPX_logging/runs name=grid55_lr0.0002_8_hidden1088_mlp_512 trainer.grad_acc_steps=4 server=rusty optimizer=adam optimizer.lr=0.0002 logger.wandb_project_name="MPPX_Scaling_Grid" \
				trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \
				model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=1088 model.groups=16 model.processor_blocks=8 model.drop_path=0.1 \
				model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim=512 model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \
				model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=20 trainer.max_rollout_steps=40 \
				lr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=2 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \
				trainer.prediction_type="delta" data=rusty_all_data trainer.max_epoch=120 data_workers=6 model.override_dimensionality=0 auto_resume=True 
