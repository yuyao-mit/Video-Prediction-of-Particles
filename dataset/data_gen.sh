#!/bin/bash
#SBATCH --job-name=data_gen               # Job name
#SBATCH --output=data_gen_%j.log          # Standard output and error log
#SBATCH --gres=gpu:volta:1                # Volta GPUs
#SBATCH --cpus-per-task=40                # Request 40 CPUs per task
#SBATCH --time=24:00:00                   # Set maximum runtime (adjust as needed)
#SBATCH --mem=300G                        # Memory allocation (adjust as needed)

# Load necessary modules
source /etc/profile
module load anaconda/Python-ML-2024b

# Run the Python script
python data_gen_B.py
