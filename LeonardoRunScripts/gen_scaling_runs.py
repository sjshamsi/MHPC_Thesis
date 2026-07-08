# ruff: noqa: E501
"""Generate single-node (1x boost_usr_prod node, 4x A100) SLURM scripts that sweep
Walrus architecture size, modeled on ExampleRunScripts/star_search.py's "vary one
axis at a time around a base anchor" pattern.

Usage:
    python gen_scaling_runs.py
    bash run_all.sh          # submits every generated job
    # or: sbatch scaling_run_3.sh   # submit one at a time
"""

import os
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_DIR = "/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs"
WALRUS_DIR = "/leonardo/home/userexternal/sshamsi0/Walrus/walrus"
VENV_ACTIVATE = "/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate"
WANDB_PROJECT = "walrus_leonardo_scaling"

# Base architecture anchor, sized for a single node's 4x A100 rather than the
# original multi-node study. Edit these to taste.
base: Dict[str, Any] = {
    "depth": 12,       # model.processor_blocks
    "hidden_dim": 768,  # model.hidden_dim
    "mlp": 2048,       # model.processor.space_mixing.mlp_dim
}

heads = 16
lrs = [2.0e-4]

# Star sweep: for each axis below, hold the other two at their base value and
# vary just that one, one job per value.
options: Dict[str, List[Any]] = {
    "hidden_dim": [384, 512, 640, 896, 1024],
    "depth": [6, 8, 10, 14, 16],
    "mlp": [1024, 1536, 2560, 3072],
}

SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
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
#SBATCH --output={experiment_dir}/slurm/%x-%j.out
#SBATCH --error={experiment_dir}/slurm/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=${{SLURM_CPUS_ON_NODE}}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR={experiment_dir}

module purge
module load python
module load cuda/12.2
source {venv_activate}

cd {walrus_dir}

srun python -u `which torchrun` \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --nproc_per_node=$SLURM_GPUS_PER_NODE \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \\
        train.py distribution=ddp server=leonardo data=smallset_leonardo \\
            ++experiment_dir={experiment_dir} \\
            name={job_name} trainer.grad_acc_steps=4 optimizer=adam optimizer.lr={lr} \\
            logger.wandb_project_name="{wandb_project}" \\
            trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=100 trainer.clip_gradient=10 \\
            data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \\
            model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim={hidden_dim} model.groups=16 \\
            model.processor_blocks={depth} model.drop_path=0.1 \\
            model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.mlp_dim={mlp} \\
            model.processor.space_mixing.num_heads={heads} model.processor.time_mixing.num_heads={heads} \\
            model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 \\
            trainer.short_validation_length=20 trainer.max_rollout_steps=20 lr_scheduler=inv_sqrt_w_sqrt_ramps \\
            trainer.val_frequency=5 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \\
            trainer.prediction_type="delta" trainer.max_epoch=101 data_workers=8 \\
            model.override_dimensionality=0 ++trainer.skip_spectral_metrics=True auto_resume=True \\
            trainer.video_validation=False
"""


def main():
    scripts = []
    count = 0

    for axis in options:
        combo = base.copy()
        for value in options[axis]:
            combo[axis] = value
            depth, hidden_dim, mlp = combo["depth"], combo["hidden_dim"], combo["mlp"]
            for lr in lrs:
                job_name = f"scale{count}_hidden{hidden_dim}_depth{depth}_mlp{mlp}"
                script_body = SBATCH_TEMPLATE.format(
                    job_name=job_name,
                    experiment_dir=EXPERIMENT_DIR,
                    venv_activate=VENV_ACTIVATE,
                    walrus_dir=WALRUS_DIR,
                    wandb_project=WANDB_PROJECT,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    mlp=mlp,
                    heads=heads,
                )
                script_path = os.path.join(SCRIPT_DIR, f"scaling_run_{count}.sh")
                with open(script_path, "w") as f:
                    f.write(script_body)
                os.chmod(script_path, 0o755)
                scripts.append(f"sbatch scaling_run_{count}.sh")
                count += 1

    run_all_path = os.path.join(SCRIPT_DIR, "run_all.sh")
    with open(run_all_path, "w") as f:
        f.write("#!/bin/bash\n" + "\n".join(scripts) + "\n")
    os.chmod(run_all_path, 0o755)

    print(f"Generated {count} scaling run scripts + run_all.sh in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
