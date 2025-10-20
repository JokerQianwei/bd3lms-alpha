#!/bin/bash
set -euo pipefail
#SBATCH -J genppl_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# 支持缺省参数：未传参时使用 LENGTH=64, SEED=1, block_size不传默认等于
LENGTH=${1:-64}
SEED=${2:-1}

# use model trained w/o eos for variable-length generation
python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=smiles \
    algo=mdlm \
    algo.T=5000 \
    model.length=$LENGTH \
    eval.checkpoint_path=$PWD/outputs/smiles/2025.10.20/205857/checkpoints/best.ckpt \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=25 \
    sampling.nucleus_p=0.9 \
    sampling.logdir=$PWD/sample_logs/samples_mdlm_len${LENGTH}
