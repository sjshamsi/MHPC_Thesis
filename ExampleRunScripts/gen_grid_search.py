# ruff: noqa: E501
from typing import Any, Dict, List

heads = 16
lrs = [2.e-4]

all_options = []
options_3: Dict[str, List[Any]] = {
     "mlp": [416, 608, 768,],
    "hidden_dim": [48*16, 68*16,85*16,], 
    "depth": [6,  10, 12,], 
}
options_last: Dict[str, List[Any]] = {
    "mlp": [512],
    "hidden_dim": [58*16], 
    "depth": [8], 
}

for mlp_val in options_3["mlp"]:
    for hidden_dim_val in options_3["hidden_dim"]:
        for depth_val in options_3["depth"]:
            all_options.append(
                {
                    "mlp": mlp_val,
                    "hidden_dim": hidden_dim_val,
                    "depth": depth_val,
                }
            )

total_options = {k: options_3[k] + options_last[k] for k in options_3.keys()}

print("total_options", total_options)
for mlp_val in total_options["mlp"]:
    for hidden_dim_val in total_options["hidden_dim"]:
        for depth_val in total_options["depth"]:
            if depth_val not in options_last["depth"] and mlp_val not in options_last["mlp"] and hidden_dim_val not in options_last["hidden_dim"]:
                continue
            
            all_options.append(
                {
                    "mlp": mlp_val,
                    "hidden_dim": hidden_dim_val,
                    "depth": depth_val,
                }
            )

# random.shuffle(all_options)
# Create a list of the cross product of all options in options

scripts = []

SLURM_CPUS_ON_NODE = "{SLURM_CPUS_ON_NODE}"

count = 0

for new_dict in all_options:
    depth = new_dict["depth"]
    hidden_dim = new_dict["hidden_dim"]
    mlp = new_dict["mlp"]
    for lr in lrs:
        print(count, lr, new_dict)
        base_tuning_string = f"""#!/bin/bash -l
#SBATCH --time=168:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J grid{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp} 
#SBATCH --output=/mnt/home/polymathic/ceph/MPPX_logging/runs/logs/slurmid%j--grid{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp}.log
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

srun python -u `which torchrun` \\
\t\t--nnodes=$SLURM_JOB_NUM_NODES \\
\t\t--nproc_per_node=$SLURM_GPUS_PER_NODE \\
\t\t--rdzv_id=$SLURM_JOB_ID \\
\t\t--rdzv_backend=c10d \\
\t\t--rdzv_endpoint=$SLURMD_NODENAME:29500 \\
\t\t\ttrain.py distribution=ddp ++experiment_dir=/mnt/home/polymathic/ceph/MPPX_logging/runs name=grid{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp} trainer.grad_acc_steps=4 server=rusty optimizer=adam optimizer.lr={lr} logger.wandb_project_name="MPPX_Scaling_Grid" \\
\t\t\t\ttrainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=3 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \\
\t\t\t\tmodel.projection_dim=48 model.intermediate_dim=352 model.hidden_dim={hidden_dim} model.groups=16 model.processor_blocks={depth} model.drop_path=0.1 \\
\t\t\t\tmodel/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim={mlp} model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \\
\t\t\t\tmodel.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=20 trainer.max_rollout_steps=20 \\
\t\t\t\tlr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=2 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \\
\t\t\t\ttrainer.prediction_type="delta" data=rusty_all_data trainer.max_epoch=101 data_workers=6 model.override_dimensionality=0 auto_resume=True 
"""
        script_name = f"example_run_scripts/grid_search_{count}.sh"
        with open(script_name, "w") as f:
            f.write(base_tuning_string)
        count += 1
        scripts.append("sbatch " + script_name + "\nsleep .2")

with open("example_run_scripts/run_grid_search.sh", "w") as f:
    f.write("\n".join(scripts))