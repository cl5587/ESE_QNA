#!/bin/bash
#SBATCH --job-name=train_model_test     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G         # memory per cpu-core (4G is default)
#SBATCH --time=01:50:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --constraint=a100        # only a100 gpu
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vi6908@princeton.edu

module purge
module load anaconda3/2022.5

conda activate hf


HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
accelerate launch train_model.py