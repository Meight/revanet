#!/usr/bin/env bash

WORKING_DIR="/projets/thesepizenberg/deep-learning/revanet/scripts"
TRANSFORMATIONS=("erosion" "dilation")
PRECISION_TOLERANCE=0.09

for transformation in "${TRANSFORMATIONS[@]}"; do
    current_target_miou=60
    step=2
    upper_bound=98

    while (( current_target_miou <= 98 )); do
        sbatch ${WORKING_DIR}/alter_annotations.sh ${transformation} ${current_target_miou} ${PRECISION_TOLERANCE}
        echo "Launched annotations alterations with transformation '${transformation}' and target mIOU of ${current_target_miou}."
        (( current_target_miou += step ))
    done
done
