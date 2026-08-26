#!/bin/bash -l
#SBATCH --job-name=fullDims_activematter_multinode
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Single-dataset counterpart to fullDims_allDsets_multinode.sh: same paper-scale
# architecture and same --nodes=4/distribution=hsdp/local_size=4 split, but
# data=Leonardo_active_matter (just active_matter) instead of data=Leonardo_smallest8_2_3d
# (all 6 datasets), and on boost_qos_dbg instead of normal. boost_qos_dbg caps wall time
# at 00:30:00 (sacctmgr show qos boost_qos_dbg), so --time is forced down from 08:00:00
# regardless. 4 nodes x 4 GPUs = 16 GPUs stays within boost_qos_dbg's
# MaxTRESPerUser (node=8, gpu=32).
#
# Only the dataset and queue changed here - trainer.max_epoch=71/max_samples=2000/
# val_frequency=10/etc. are all still unchanged from fullDims_allDsets_multinode.sh's
# throughput-experiment values, sized for an 8h normal-qos run, not a 30-minute debug
# slot, so this will very likely hit boost_qos_dbg's TIME LIMIT well before max_epoch=71 -
# useful as a debug-queue calibration run, not sized to actually complete. Shrink
# max_epoch/max_samples/lr_scheduler ramps if you want a run that finishes within the
# slot.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
# True (default) continues the latest run under this name - correct for resubmitting
# after a timeout/crash. Set to False (e.g.
# `./submit.sh --export=ALL,AUTO_RESUME=False fullDims_activematter_multinode.sh`) to
# force a brand-new run_idx folder and a brand-new wandb run for a deliberate fresh
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
            distribution=hsdp \
            distribution.local_size=4 \
            server=leonardo \
            data=Leonardo_active_matter \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            name=fullDims_activematter_multinode \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="fullDims_activematter_multinode" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
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
            trainer.max_epoch=71 \
            data_workers=8 \
            model.override_dimensionality=0 \
            ++trainer.skip_spectral_metrics=True \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            finetuning_mods=defaults \
            trainer.video_validation=False
