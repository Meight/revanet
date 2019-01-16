#!/usr/bin/env bash
#SBATCH --job-name=training-full-val
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/bisenet-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/bisenet-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=9000M

set -e

# Various script and dataset paths.
VENV_PATH="/users/thesepizenberg/mlebouch/venv-tf"
TRAIN_SCRIPT_DIR="/projets/thesepizenberg/deep-learning/revanet"

# Begin script.

# Create a virtual environment from the Docker container.

srun keras-py3-tf virtualenv --system-site-packages /users/thesepizenberg/mlebouch/venv
wait

srun keras-py3-tf ${VENV_PATH}/bin/python "$TRAIN_SCRIPT_DIR/train_multi_gpu.py" \
                --model-name=${1} \
                --backbone-name=${2} \
                --input-size=${3} \
                --batch-size=${4} \
                --dataset-name=${5} \
                --train-annotations-folder=${6} \
                --number-of-epochs=15 \
                --save-weights-every=5 \
                --validate-every=1 \
                --number-of-gpus=1 \
                --number-of-cpus=4
wait