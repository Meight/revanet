#!/usr/bin/env bash
#SBATCH --job-name=bisenet-gpu
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/gpu-bisenet-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/gpu-bisenet-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=9000M

set -e

# Various script and dataset paths.
TRAIN_SCRIPT_DIR="/projets/thesepizenberg/deep-learning/revanet"
VENV_PATH=$TRAIN_SCRIPT_DIR

# Begin script.

# Create a virtual environment from the Docker container.

srun keras-py3-tf virtualenv --system-site-packages ${VENV_PATH}
wait

# srun keras-py3-tf ${VENV_PATH}/bin/pip install -r ${TRAIN_SCRIPT_DIR}/requirements.txt

echo "Launching with on train folder: ${6}"
srun keras-py3-tf ${VENV_PATH}/bin/python "$TRAIN_SCRIPT_DIR/train.py" \
                --model-name=${1} \
                --backbone-name=${2} \
                --input-size=${3} \
                --batch-size=${4} \
                --dataset-name=${5} \
                --train-annotations-folder=${6} \
		--learning-rate=${7} \
                --number-of-epochs=75 \
                --save-weights-every=24 \
                --validate-every=1
wait
