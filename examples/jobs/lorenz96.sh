#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=blanca-bortz
#SBATCH --account=blanca-bortz
#SBATCH --qos=blanca-bortz
#SBATCH --gres=gpu:1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --output=sample-%j.out

module load anaconda

conda activate torch_env

python lorenz96/l96-ttwsindy.py