#!/bin/bash -l
#SBATCH --job-name=test_lr_resume
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.err

# Runs tests/test_lr_resume.py (LR schedule unchanged by checkpoint resume)
# on a boost node.
#
# Usage (submit.sh creates the --output folder, which SLURM won't):
#   Tools/slurm/submit.sh Runs/test_lr_resume/test_lr_resume.sh
#
# Expect PASSED for both [linear_ramps] and [sqrt_ramps].

module purge
module load python/3.11.7
module load cuda/12.6
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate

cd /leonardo/home/userexternal/sshamsi0/Public_Thesis
python -m pytest tests/test_lr_resume.py -v -W ignore