#!/usr/bin/env bash
#SBATCH --job-name=training-full-val
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/parse-text-files.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/parse-text-files.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=7500M


set -e

# Various script and dataset paths.
SCRIPT_PATH="/projets/thesepizenberg/deep-learning/revanet/datasets"
TEXT_FILES_DIRECTORY="/projets/thesepizenberg/deep-learning/datasets/VOC2012-fresh"
DATASET_NAME="augmented-pascal"
TARGET_DATASET_ROOT="/projets/thesepizenberg/deep-learning/revanet/datasets"

# Begin script.

# Create a virtual environment from the Docker container.

srun keras-py3-tf virtualenv --system-site-packages /users/thesepizenberg/mlebouch/venv
wait

srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/python "$SCRIPT_PATH/text_to_subfolders.py" \
               --text-files-directory=${TEXT_FILES_DIRECTORY} \
               --dataset-name=${DATASET_NAME} \
               --target-dataset-root=${TARGET_DATASET_ROOT}
wait