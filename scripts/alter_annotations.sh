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
DATASET_NAME="augmented-pascal"
TARGET_DATASET_ROOT="/projets/thesepizenberg/deep-learning/revanet/datasets"
DATASET_ROOT_PATH="/projets/thesepizenberg/deep-learning/revanet/datasets"
ANNOTATIONS_PATH="${DATASET_ROOT_PATH}/train_labels"
TARGET_PATH=$DATASET_ROOT_PATH

# Begin script.
srun ${PYTHON_PATH}/python \
               "$SCRIPT_PATH/alter_annotations.py" \
               --annotations-paths=${ANNOTATIONS_PATH} \
               --dataset-root-path=${DATASET_ROOT_PATH} \
               --target-path=${TARGET_PATH} \
               --transformation=${0} \
               --target-average-miou=${1} \
               --precision-tolerance=${2}
wait