#!/bin/sh
#SBATCH --job-name mnihg
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --export=ALL
#SBATCH --time=07-00:00:00
srun --output output/mnihg.log --error output/mnihg.stderr sh configs/train_hifigan.sh