#!/bin/bash
#SBATCH --mail-user=21422885@life.hkbu.edu.hk
#SBATCH --mail-type=end
#SBATCH --time=5:00:00
# std oupt
#SBATCH -o log.o
##SBATCH --partition=compute

#SBATCH --job-name="cp"
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account="research-ceg-tp"

module load 2023r1 cuda/11.7

module load miniconda3
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate diffusion
cd ${HOME}/Devs/Crowd\-Flow\-Inference
#python main.py --mode ood --file dcrnn
python Online_Update.py

#export conda_env=${HOME}/anaconda3/envs/frl
