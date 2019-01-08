#!/usr/bin/env bash
#SBATCH --job-name=training-full-val
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/bisenet-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/bisenet-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=9000M

set -e

# Various script and dataset paths.
VENV_PATH="/users/thesepizenberg/mlebouch/venv-tf"
TRAIN_SCRIPT_DIR="/projets/thesepizenberg/deep-learning/revanet"

# Begin script.

# Create a virtual environment from the Docker container.

srun keras-py3-tf virtualenv --system-site-packages ${VENV_PATH}
wait

srun keras-py3-tf ${VENV_PATH}/bin/python "$TRAIN_SCRIPT_DIR/train.py" \
                --number-of-epochs=75 \
                --save-weights-every=24 \
                --validate-every=1 \
                --number-of-validation-images=1449 \
                --model-name=BiSeNet \
                --backbone-name=ResNet101 \
                --input-size=384 \
                --batch-size=1 \
                --dataset-name=augmented-pascal
wait