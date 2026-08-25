#!/bin/bash -l
#SBATCH --job-name=fullDims_FT_ActiveMatter
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# lr_scheduler=inv_sqrt_w_sqrt_ramps_longer defaults to warmup_epochs=20/cooldown_epochs=10

# trainer=defaults (the repo example uses uses trainer=globalnorm, i.e.
# GlobalRevNormalization instead of SamplewiseRevNormalization).
# walrus_pretrained/extended_config.yaml was trained with SamplewiseRevNormalization

# walrus_pretrained/extended_config.yaml was trained with drop_path=0.05.
# walrus/configs/model/isotropic_model.yaml#L11 also has drop_path=0.05.
# walrus/models/isotropic_model.py#L81 shows that per-block drop probabilities
# are np.linspace(0, drop_path, processor_blocks), so the probability of dropping
# the first processor block is 0 and dropping the last block is drop_path.
# This is finetuning so drop_path=0.0

# ++trainer.epsilon=1e-8 is not added from
# walrus/run_scripts/finetuning_example_distributed_walrus.sh.
# ++ is there because epsilon is a variable of class Trainer with default value 1e-5.
# walrus/trainer/training.py#L178 shows that epsilon is passed to
# compute_stats(..., epsilon=...) so if a stat is lower than epsilon, it can be clamped
# to epsilon. This can help avoid devide by zero stuff. The example trains on
# euler_multi_quadrants_openBC which might need a tighter epsilon? But if there
# is some constant value in activematter RMSs happens in activamatter and
# then, say, a vcariance is calculated then it'll turn from 0 to ie-8 which may be
# too small, so we'll just let it be the default 1e-5.

# ++data.module_parameters.start_rollout_valid_output_at_t=17 is in
# walrus/run_scripts/finetuning_example_distributed_walrus.sh but be do not add it here.
# It forces evry validation sample to start scoring predictions at the same absolute
# timestep in the trajectory (site-packages/the_well/data/datasets.py:820).
# site-packages/the_well/data/datasets.py:820 says that it helps when you are comparing
# models that require different input contexts for a fair prediction. We don't add
# it to keep it at the default -1.

# We add data/field_index_map_override=full_well_field_index.
# walrus/data/multidatamodule.py#L114-L115 says that this is an "Optional dictionary
# to override the field to index mapping. Useful when loading an old checkpoint."
# walrus/configs/data/field_index_map_override/full_well_field_index.yaml has a full
# index assignment of all the ~40 fields throuhgout the datasets of Walrus.
# walrus/configs/data/Leonardo_active_matter.yaml#L2 declares
# field_index_map_override: bc_only_override. 
# The model has one input and one output weight matrix with a fixed number of columns,
# one for each of the ~40 Walrus fields. full_well_field_index means a new field to
# index map is not made.

# trainer.video_validation=False because I don't have a good FFMPEG compilation yet.

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
            model=isotropic_model \
            name=fullDims_finetuneActiveMatter_longRun \
            trainer=defaults \
            trainer.grad_acc_steps=1 \
            server=leonardo \
            optimizer=adam \
            optimizer.lr=1.e-4 \
            logger.wandb_project_name="fullDims_finetuneActiveMatter" \
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
            model.drop_path=0.0 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.num_heads=16 \
            model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True \
            model.jitter_patches=True \
            data.module_parameters.max_samples=2000 \
            trainer.short_validation_length=20 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps_longer \
            trainer.val_frequency=5 \
            trainer.rollout_val_frequency=5 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=1 \
            trainer.prediction_type="delta" \
            data=Leonardo_active_matter \
            trainer.max_epoch=51 \
            data_workers=10 \
            model.override_dimensionality=0 \
            auto_resume=$AUTO_RESUME \
            checkpoint=finetune_leonardo \
            experiment=finetune_leonardo \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0 \
            ++trainer.skip_spectral_metrics=True \
            finetuning_mods=all \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            data/field_index_map_override=full_well_field_index \
            trainer.video_validation=False
