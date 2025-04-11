#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <folder_original_images> <folder_predictions> <output_folder>"
  exit 1
fi

FOLDER1="$1"
FOLDER2="$2"
OUTPUT_FOLDER="$3"

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