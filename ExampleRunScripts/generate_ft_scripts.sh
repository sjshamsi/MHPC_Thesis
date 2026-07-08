#!/bin/bash -l
#SBATCH --partition=mem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500G
#SBATCH --output=ft_scripts_%j.out
#SBATCH --error=ft_scripts_%j.err

# Optional environment setup
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate your environment
source /mnt/home/pmukhopadhyay/projects/multiple_physics_pretraining/myenv3.10/bin/activate
export PYTHONPATH=$PYTHONPATH:/mnt/home/pmukhopadhyay/projects/temporary_mppx_name/

# Just run the Python script
python example_run_scripts/generate_ft_scripts.py