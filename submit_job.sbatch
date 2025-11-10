#!/bin/bash
#SBATCH -JCS6220Project                 # Job name
#SBATCH -N4 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=32G                       # Memory per core
#SBATCH -t16:00:00                              # Duration of the job 
#SBATCH --gres=gpu:H100:1
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=ymai8@gatech.edu           # E-mail address for notifications

# activate environment
source cs6200/bin/activate
# check GPU, will be logged to output file
nvidia-smi
# add your command to run the training script here, for example:
