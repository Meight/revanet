#!/usr/bin/env bash
#SBATCH --job-name=alter-annotations
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/alter-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/alter-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=7500M


set -e

# Various script and dataset paths.
PYTHON_PATH="/users/thesepizenberg/mlebouch/venv/bin"
SCRIPT_PATH="/projets/thesepizenberg/deep-learning/revanet/datasets"
TEXT_FILES_DIRECTORY="/projets/thesepizenberg/deep-learning/datasets/VOC2012-fresh"
DATASET_NAME="augmented-pascal-altered"
TARGET_DATASET_ROOT="/projets/thesepizenberg/deep-learning/revanet/datasets/${DATASET_NAME}"
DATASET_ROOT_PATH="/projets/thesepizenberg/deep-learning/revanet/datasets/${DATASET_NAME}"
ANNOTATIONS_PATH="${DATASET_ROOT_PATH}/train_labels"
TARGET_PATH=$DATASET_ROOT_PATH

# Begin script.
srun ${PYTHON_PATH}/python \
               "$SCRIPT_PATH/alter_annotations.py" \
               --annotations-path=${ANNOTATIONS_PATH} \
               --dataset-root-path=${DATASET_ROOT_PATH} \
               --target-path=${TARGET_PATH} \
               --transformation=${1} \
               --target-average-miou=${2} \
               --precision-tolerance=${3}
wait