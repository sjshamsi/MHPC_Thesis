#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH -p gpuxl
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=23
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J star4_lr0.00025_40_hidden368_mlp_5632 
#SBATCH --output=logs/slurmid%j-star4_lr0.00025_40_hidden368_mlp_5632.log

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=torch_worker_log.json
export NCCL_DEBUG=WARN

# Launch the training script
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Launch the training script

srun -u --mpi=pmix \
	--container-image=/lustre/fs0/scratch/mmccabe/sqsh-files/nv_25.01-py3.sqsh \
	--container-mounts=/lustre/fs0/scratch/ \
	--container-workdir=/lustre/fs0/scratch/mmccabe/code/temporary_mppx_name/temporary_mppx_name \
	torchrun \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--nproc_per_node=$SLURM_GPUS_PER_NODE \
		--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
			train.py distribution=hsdp name=star4_lr0.00025_40_hidden368_mlp_5632 trainer.grad_acc_steps=4 server=gpuxl optimizer=adam optimizer.lr=0.00025 logger.wandb_project_name="MPPX_Scaling" \
				trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \
				model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=368 model.groups=16 model.processor_blocks=40 model.drop_path=0.1 \
				model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim=5632 model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \
				model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=20 trainer.max_rollout_steps=20 \
				lr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=5 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \
				trainer.prediction_type="delta" data=all_2_3d trainer.max_epoch=101 data_workers=10 model.override_dimensionality=0 ++trainer.skip_spectral_metrics=True auto_resume=True 
