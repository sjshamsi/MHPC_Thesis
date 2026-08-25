#!/bin/bash -l
#SBATCH --job-name=fullDims_allDsets_multinode
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
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

# Multi-node counterpart to fullDims_allDsets.sh: same paper-scale architecture and
# same data=available_leonardo (all 6 datasets), spread across --nodes=4 (16 GPUs)
# instead of 1. --dependency=singleton guards against an overlapping resubmission
# landing while a previous one is still running, matching
# anchor_dimension_run_multinode.sh's precedent.
#
# distribution=hsdp (was fsdp) + distribution.local_size=4 (new), instead of just
# bumping --nodes= and leaving distribution=fsdp: plain FSDP shards parameters across
# *every* GPU in the job regardless of node count (configure_distribution builds one
# flat device mesh over the full world_size), so at 4 nodes that means every
# all-gather/reduce-scatter crosses the inter-node network fabric instead of staying
# on fast intra-node NVLink/PCIe - expensive at this model size (~1.3B params). HSDP
# with local_size=4 instead shards only *within* each node's 4 GPUs (fast interconnect)
# and replicates DDP-style *across* the 4 nodes (only needs a gradient all-reduce over
# the slower inter-node link, not a full parameter all-gather every step) - this is
# also exactly what the released checkpoint's own pretraining recipe used
# (walrus_pretrained/extended_config.yaml: distribution: hsdp, local_size: 4, just at
# 24 nodes instead of 4).
#
# Also fixed here (present in fullDims_allDsets.sh, unrelated to multinode): the
# `name=`/`logger.wandb_project_name=` lines were each missing their trailing `\`,
# which truncates the whole train.py invocation at that point - see below, both now
# properly continued.
#
# Everything else (model.* architecture, data=available_leonardo, max_samples=2000,
# max_epoch=71, gradient_checkpointing_freq=0, etc.) is unchanged from
# fullDims_allDsets.sh on purpose. Note this does NOT make max_epoch=71 finish faster:
# data.module_parameters.max_samples caps micro-batches *per rank*, so steps-per-epoch
# is fixed regardless of node count (see walrus/data/mixed_dset_sampler.py's
# BatchedMultisetSampler). More nodes instead gives a bigger effective global batch per
# optimizer step (batch_size * grad_acc_steps * world_size), at proportionally higher
# GPU-hour cost for the same wall-clock duration - test on a shorter budget before
# trusting max_epoch=71 here, same caution as the single-node version.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
# True (default) continues the latest run under this name - correct for resubmitting
# after a timeout/crash. Set to False (e.g.
# `./submit.sh --export=ALL,AUTO_RESUME=False fullDims_allDsets_multinode.sh`) to force
# a brand-new run_idx folder and a brand-new wandb run for a deliberate fresh attempt
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
            distribution=hsdp \
            distribution.local_size=4 \
            server=leonardo \
            data=available_leonardo \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            name=fullDims_allDsets_multinode \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="fullDims_allDsets_multinode" \
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
