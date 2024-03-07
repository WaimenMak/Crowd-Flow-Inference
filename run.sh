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
# pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
# pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
# pip install openpyxl
# pip install xgboost
#python main.py --mode ood --file dcrnn
python Online_Update.py --model_type="Online_Diffusion_UQ" --lags=5 --chunk_size=10 --pred_horizons=5 --train_steps=130
echo "####### Job1 finished #######"
python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=10 --pred_horizons=5 --train_steps=130
#export conda_env=${HOME}/anaconda3/envs/frl
