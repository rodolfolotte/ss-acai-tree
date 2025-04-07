#!/bin/bash

FOLDER1="/media/rodolfo/data/ss-acai-tree/artefacts/predictions/256/"
FOLDER2="/media/rodolfo/data/ss-acai-tree/data/image/256/test/"
OUTPUT_FOLDER="/media/rodolfo/data/ss-acai-tree/artefacts/predictions_overlap/256/"

mkdir -p "$OUTPUT_FOLDER"

OPACITY=50

for base_image in "$FOLDER1"/*; do
    filename=$(basename -- "$base_image")
    overlay_image="$FOLDER2/$filename"
    output_image="$OUTPUT_FOLDER/$filename"

    if [ -f "$overlay_image" ]; then
        composite -dissolve 50 -gravity Center "$base_image" "$overlay_image" -alpha Set "$output_image"
        echo "Processed: $filename"
    else
        echo "Skipping: $filename (No matching overlay)"
    fi
done

echo "Processing complete!"