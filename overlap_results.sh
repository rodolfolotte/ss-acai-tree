#!/bin/bash

FOLDER1="/media/rodolfo/data/sacha/"
FOLDER2="/home/rodolfo/git/private/ss-acai-tree/data/test_128/"
OUTPUT_FOLDER="/media/rodolfo/data/sacha_overlap/"

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