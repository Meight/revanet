#!/usr/bin/env bash

DATASETS_PATH="/projets/thesepizenberg/deep-learning/revanet/datasets"
SCRIPT_PATH="/projets/thesepizenberg/deep-learning/revanet/scripts"
MODEL_NAMES=("BiSeNet") # "FC-DenseNet103" "GCN" "DeepLabV3_plus" "RefineNet" "DenseASPP" "PSPNet" "DDSC" "AdapNet")
FRONTEND_NAMES=("ResNet101")
INPUT_SIZE=(256)
BATCH_SIZE=(4)
DATASET="augmented-pascal-altered"

ALTERATION="erosion"

# TRAIN_FOLDERS=(${DATASETS_PATH}/$DATASET/$ALTERATION*)
TRAIN_FOLDERS=("erosion_90.00_95.68")

for input_size in "${INPUT_SIZE[@]}"; do
    for batch_size in "${BATCH_SIZE[@]}"; do
        for model_name in "${MODEL_NAMES[@]}"; do
            for frontend_name in "${FRONTEND_NAMES[@]}"; do
                for ((i=0; i<${#TRAIN_FOLDERS[@]}; i++)); do
                    sbatch "$SCRIPT_PATH"/train_multi_gpu.sh ${model_name} ${frontend_name} ${input_size} ${batch_size} ${DATASET} "$(basename ${TRAIN_FOLDERS[$i]})"
                done
            done;
        done;
    done;
done;
