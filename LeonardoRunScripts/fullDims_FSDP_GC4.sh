#!/bin/bash -l
#SBATCH --job-name=fullDims_FSDP_GC4
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

# Baseline run at Walrus's *original*, unmodified paper-scale architecture
# (hidden_dim=1408, processor_blocks=40, projection_dim=48, intermediate_dim=352,
# groups=16 - the exact model.* values from
# ../walrus/run_scripts/pretrain_example_distributed_walrus.sh, not the smaller
# single-node anchor used by gen_star_search_runs.py).
#
# Currently configured as a throughput experiment rather than the original
# memory-matched-to-the-24-node-pretraining baseline:
#   - trainer.max_epoch=20 instead of 101, for faster iteration while testing
#     these throughput changes.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs
# `sbatch --export=ALL,AUTO_RESUME=False fullDims_FSDP_GC4.sh`) to force a
# brand-new run_idx folder and a brand-new wandb run for a deliberate fresh
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
            data=available_leonardo \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            name=fullDims_FSDP_GC4 \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="fullDims_FSDP_GC4" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=4 \
            trainer.log_interval=100 \
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
            data.module_parameters.max_samples=2000 \
            trainer.short_validation_length=20 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.val_frequency=10 \
            trainer.rollout_val_frequency=10 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            trainer.max_epoch=20 \
            data_workers=8 \
            model.override_dimensionality=0 \
            ++trainer.skip_spectral_metrics=True \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            finetuning_mods=defaults \
            trainer.video_validation=False
