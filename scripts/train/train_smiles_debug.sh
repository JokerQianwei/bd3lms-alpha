#!/bin/bash
#SBATCH -J train_smiles_bd3lm           # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=32G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

BLOCK_SIZE=4
SEQ_LEN=64
PRETRAIN_CKPT=null  # to train from scratch; set to HF/ckpt path to warm start
MAX_STEPS=330_000

python -u main.py \
    loader.global_batch_size=100 \
    loader.eval_global_batch_size=100 \
    model=medium \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=smiles_debug \
    model.length=${SEQ_LEN} \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-smiles-len${SEQ_LEN}-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    trainer.max_steps=$MAX_STEPS \
    trainer.val_check_interval=0.1 \
    trainer.limit_val_batches=0.01

