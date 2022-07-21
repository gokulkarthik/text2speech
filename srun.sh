#!/bin/sh
#SBATCH --job-name fastpitch
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --partition dgxnp
#SBATCH --export=ALL,WANDB_MODE=offline
#SBATCH --time=07-00:00:00
srun --output output/fastpitch.log --error output/fastpitch.stderr sh configs/train_fastpitch.sh