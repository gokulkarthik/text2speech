#!/bin/sh
#SBATCH --job-name fp_te_2x
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --partition ai4bp
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090
#SBATCH --time=07-00:00:00
#SBATCH --output=output/slurm-%N-%j.out
srun --output output/fp_te_2x.log --error output/fp_te_2x.stderr sh configs/train_fastpitch.sh