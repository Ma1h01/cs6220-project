# cs6220-project

## Setup
1. `python3 -m venv cs6200` - creates a virtual environment
2. `source cs6200/bin/activate` - activates the env
3. `pip install -r requirements.txt` - install all dependencies
4. Download a subsampled ChestXpert dataset, which originally has ~410GB, with same amount of data in ~11GB. Either through:
    1. Huggingface `danjacobellis/chexpert`, or
    2. Kaggle `#!/bin/bash
    curl -L -o ~/Downloads/chexpert-v10-small.zip\
    https://www.kaggle.com/api/v1/datasets/download/willarevalo/chexpert-v10-small` - this download the dataset into your local directory instead of being cached in huggingface.

## How to request PACE GPU
### Local Dev/Testing
1. `salloc --gres=gpu:H100:1 --ntasks-per-node=1` - submit a request for one node with `H100` GPU and one task per node
2. `nvidia-smi` - verift you have access to GPU

### Computationally Internsive Task
1. Change email to yours to receive slurm job status through email in `submit_job.sbatch`, and modify the run commands if you need.
2. `sbatch submit_job.sbatch` - submit a slurm job that runs in the background