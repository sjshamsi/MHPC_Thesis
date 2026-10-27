#!/bin/bash -l
#SBATCH --job-name=verify_registry
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

# Checks Tools/flop_planner/well_dataset_registry.json against the real pipeline: one real
# window per dataset through datamodule -> formatter -> a tiny IsotropicModel. The geometry
# FlopAccountant observes must equal the one the registry implies, and both must give
# identical forward_flops. Rerun after every registry rebuild.
#
# Usage (submit.sh creates the --output folder, which SLURM won't):
#   Tools/slurm/submit.sh Runs/verify_registry/verify_registry.sh
#
# Expect "ok" and "exact" on every dataset row, and exit code 0.

module purge
module load python/3.11.7
module load cuda/12.6
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate

cd /leonardo/home/userexternal/sshamsi0/Public_Thesis
python Tools/flop_planner/build_well_registry.py --verify-only
