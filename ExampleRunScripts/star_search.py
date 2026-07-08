# ruff: noqa: E501

from typing import Any, Dict, List


base: Dict[str, List[Any]] = {
    "depth": 40,
    "hidden_dim": 1408,
    "mlp": 5632
}

heads = 16
lrs = [2.5e-4]
options: Dict[str, List[Any]] = {
    "hidden_dim": [
        # heads*int(88*.8), 
                   heads*int(88*.8**2), heads*int(88*.8**3), heads*int(88*.8**4), heads*int(88*.8**5), heads*int(88*.8**6)], 
    "depth": [
        # 24, 
        20, 16, 12, 10, 8], 
     "mlp": [
        #  4504, 
         3604, 2872, 2308, 1830, 1056],

}

# mlp_name_map = {""}
scripts = []

SLURM_CPUS_ON_NODE = "{SLURM_CPUS_ON_NODE}"

count = 0

for key in options:
    new_dict = base.copy()
    for option in options[key]:
        new_dict[key] = option
        depth = new_dict["depth"]
        hidden_dim = new_dict["hidden_dim"]
        mlp = new_dict["mlp"]
        for lr in lrs:
            print(count, lr, new_dict)
            base_tuning_string = f"""#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH -p gpuxl
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=23
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J star{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp} 
#SBATCH --output=logs/slurmid%j-star{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp}.log

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=torch_worker_log.json
export NCCL_DEBUG=WARN

# Launch the training script
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Launch the training script

srun -u --mpi=pmix \\
\t--container-image=/lustre/fs0/scratch/mmccabe/sqsh-files/nv_25.01-py3.sqsh \\
\t--container-mounts=/lustre/fs0/scratch/ \\
\t--container-workdir=/lustre/fs0/scratch/mmccabe/code/temporary_mppx_name/temporary_mppx_name \\
\ttorchrun \\
\t\t--nnodes=$SLURM_JOB_NUM_NODES \\
\t\t--nproc_per_node=$SLURM_GPUS_PER_NODE \\
\t\t--rdzv_id=$SLURM_JOB_ID \\
\t\t--rdzv_backend=c10d \\
\t\t--rdzv_endpoint=$SLURMD_NODENAME:29500 \\
\t\t\ttrain.py distribution=hsdp name=star{count}_lr{lr}_{depth}_hidden{hidden_dim}_mlp_{mlp} trainer.grad_acc_steps=4 server=gpuxl optimizer=adam optimizer.lr={lr} logger.wandb_project_name="MPPX_Scaling" \\
\t\t\t\ttrainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \\
\t\t\t\tmodel.projection_dim=48 model.intermediate_dim=352 model.hidden_dim={hidden_dim} model.groups=16 model.processor_blocks={depth} model.drop_path=0.1 \\
\t\t\t\tmodel/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim={mlp} model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \\
\t\t\t\tmodel.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=20 trainer.max_rollout_steps=20 \\
\t\t\t\tlr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=5 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \\
\t\t\t\ttrainer.prediction_type="delta" data=all_2_3d trainer.max_epoch=101 data_workers=10 model.override_dimensionality=0 ++trainer.skip_spectral_metrics=True auto_resume=True 
"""
            script_name = f"example_run_scripts/star{count}_tuning_script_{count}.sh"
            with open(script_name, "w") as f:
                f.write(base_tuning_string)
            count += 1
            if f"name=lr{lr}_{40}_hidden{1408}_mlp_{3.2}" not in base_tuning_string:
                scripts.append("sbatch " + script_name)

with open("example_run_scripts/star_run_all.sh", "w") as f:
    f.write("\n".join(scripts))