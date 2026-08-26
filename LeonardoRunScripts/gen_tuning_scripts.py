# ruff: noqa: E501
"""Generate single-node (1x boost_usr_prod node, 4x A100) SLURM scripts that grid-search
training/modeling hyperparameters, modeled on ExampleRunScripts/gen_tuning_scripts.py's
sweep - but retargeted at Leonardo and at the sweep-anchor architecture (the same
768/12/2048 base gen_star_search_runs.py holds constant) instead of paper scale, so a full
5-axis grid is actually affordable here.

Unlike gen_star_search_runs.py (which fixes hyperparameters and varies architecture size),
this sweeps training/modeling *choices* at a fixed architecture size - complementary
axes, not overlapping ones.

Usage:
    python gen_tuning_scripts.py
    bash run_tuning_all.sh          # submits every generated job
    # or: sbatch tuning_run_3.sh    # submit one at a time
"""

import itertools
import os
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_DIR = "/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs"
WALRUS_DIR = "/leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus"
VENV_ACTIVATE = "/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate"
WANDB_PROJECT = "walrus_leonardo_tuning"

# Sweep-anchor architecture - same base gen_star_search_runs.py holds constant while
# varying hidden_dim/depth/mlp, and the same one anchor_dimension_run.sh trains at.
# Kept fixed here since this script sweeps hyperparameters, not architecture size.
ARCH = {
    "hidden_dim": 768,
    "depth": 12,       # model.processor_blocks
    "mlp": 2048,       # model.processor.space_mixing.mlp_dim
}
heads = 16

# Full grid, same 5 axes/values as ExampleRunScripts/gen_tuning_scripts.py.
options: Dict[str, List[Any]] = {
    "lr": [0.001, 0.0001, 0.01],
    "prediction_type": ['"delta"', '"full"'],
    "causal_in_time": [False, True],
    "jitter_patches": [True, False],
    "space_mixing": ["full_spatial_attention", "axial_spatial_attention"],
}

SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output={experiment_dir}/slurm/{job_name}/%j.out
#SBATCH --error={experiment_dir}/slurm/{job_name}/%j.err

set -euo pipefail

export OMP_NUM_THREADS=${{SLURM_CPUS_ON_NODE}}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR={experiment_dir}
# True (default) continues the latest run under this name - correct for
# resubmitting after a timeout/crash. Set to False (e.g.
# `sbatch --export=ALL,AUTO_RESUME=False {job_name}.sh`) to force a brand-new
# run_idx folder and a brand-new wandb run for a deliberate fresh attempt
# under the same job name.
export AUTO_RESUME=${{AUTO_RESUME:-True}}

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
        train.py distribution=ddp server=leonardo data=Leonardo_smallest8_2_3d \\
            ++experiment_dir={experiment_dir} \\
            name={job_name} trainer.grad_acc_steps=4 optimizer=adam optimizer.lr={lr} \\
            logger.wandb_project_name="{wandb_project}" \\
            trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=100 trainer.clip_gradient=10 \\
            data.module_parameters.batch_size=4 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 \\
            model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim={hidden_dim} model.groups=16 \\
            model.processor_blocks={depth} model.drop_path=0.1 \\
            model/processor/space_mixing={space_mixing} model.processor.space_mixing.mlp_dim={mlp} \\
            model.processor.space_mixing.num_heads={heads} model.processor.time_mixing.num_heads={heads} \\
            model.causal_in_time={causal_in_time} model.jitter_patches={jitter_patches} data.module_parameters.max_samples=2000 \\
            trainer.short_validation_length=20 trainer.max_rollout_steps=20 lr_scheduler=inv_sqrt_w_sqrt_ramps \\
            trainer.val_frequency=5 trainer.rollout_val_frequency=10 data.module_parameters.max_dt_stride=5 \\
            trainer.prediction_type={prediction_type} trainer.max_epoch=21 data_workers=8 \\
            model.override_dimensionality=0 ++trainer.skip_spectral_metrics=True auto_resume=$AUTO_RESUME \\
            trainer.video_validation=False
"""


def main():
    scripts = []
    count = 0

    axis_names = list(options)
    for combo in itertools.product(*(options[axis] for axis in axis_names)):
        params = dict(zip(axis_names, combo))

        pred_short = params["prediction_type"].strip('"')
        space_short = "full" if params["space_mixing"] == "full_spatial_attention" else "axial"
        job_name = (
            f"tune{count}_lr{params['lr']}_pred{pred_short}"
            f"_causal{params['causal_in_time']}_jitter{params['jitter_patches']}_{space_short}"
        )
        script_body = SBATCH_TEMPLATE.format(
            job_name=job_name,
            experiment_dir=EXPERIMENT_DIR,
            venv_activate=VENV_ACTIVATE,
            walrus_dir=WALRUS_DIR,
            wandb_project=WANDB_PROJECT,
            lr=params["lr"],
            hidden_dim=ARCH["hidden_dim"],
            depth=ARCH["depth"],
            mlp=ARCH["mlp"],
            heads=heads,
            prediction_type=params["prediction_type"],
            causal_in_time=params["causal_in_time"],
            jitter_patches=params["jitter_patches"],
            space_mixing=params["space_mixing"],
        )
        script_path = os.path.join(SCRIPT_DIR, f"tuning_run_{count}.sh")
        with open(script_path, "w") as f:
            f.write(script_body)
        os.chmod(script_path, 0o755)
        # SLURM won't create the %x subdirectory in --output/--error itself.
        os.makedirs(os.path.join(EXPERIMENT_DIR, "slurm", job_name), exist_ok=True)
        scripts.append(f"sbatch tuning_run_{count}.sh")
        count += 1

    run_all_path = os.path.join(SCRIPT_DIR, "run_tuning_all.sh")
    with open(run_all_path, "w") as f:
        f.write("#!/bin/bash\n" + "\n".join(scripts) + "\n")
    os.chmod(run_all_path, 0o755)

    print(f"Generated {count} tuning run scripts + run_tuning_all.sh in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
