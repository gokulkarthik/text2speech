#!/bin/sh
#SBATCH --job-name ta-mul
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --partition ai4bp
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090
#SBATCH --time=07-00:00:00
#SBATCH --nodelist=scn32-100g
#SBATCH --output=output/slurm-%N-%j.out
srun --output output/fastpitch3.log --error output/fastpitch3.stderr sh configs/train_fastpitch.sh