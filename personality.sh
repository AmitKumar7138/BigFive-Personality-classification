#!/bin/bash

#SBATCH --job-name= My_job
#SBATCH --mail-user= example@gamil.com
#SBATCH --mail-type=FAIL
#SBATCH --time =2 
#SBATCH --ntasks=1
#SBATCH --mem =8gb
#SBATCH --partition= gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=trase_extract.out
##SBATCH --qos= name_example

echo "Start Date : $(date)"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"

# Setting up conda environment
module load conda
conda activate env-PATH 

echo "Personality"

python3 {Current_directory_PATH}/main.py


echo "End Date    : $(date)"