#!/bin/bash
#SBATCH -J train_lm1b_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

python -u main.py \
    loader.global_batch_size=100 \
    loader.eval_global_batch_size=100 \
    model=small \
    algo=mdlm \
    data=smiles \
    model.length=64 \
    wandb.name=mdlm-smiles \
    trainer.val_check_interval=1.0 \
    algo.ignore_bos=false \
    data.raw_data_path=/data/yqw/bd3lms-alpha/data/DrugLikeSMILSE-debug \
    data.cache_dir=/cache/yqw/bd3lms-alpha/data/DrugLikeSMILES_packed1024_debug211