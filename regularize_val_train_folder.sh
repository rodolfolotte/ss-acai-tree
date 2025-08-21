#!/bin/bash
# Usage: ./split_val.sh image/train image/val labels/train

set -euo pipefail
shopt -s nullglob

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <image_train_dir> <image_val_dir> <labels_train_dir>"
  exit 1
fi

image_train_dir="${1%/}"
image_val_dir="${2%/}"
labels_train_dir="${3%/}"
labels_val_dir="$(dirname "$labels_train_dir")/val"

# Ensure required dirs exist
for d in "$image_train_dir" "$image_val_dir" "$labels_train_dir"; do
  [[ -d "$d" ]] || { echo "ERROR: Directory not found: $d"; exit 1; }
done
mkdir -p "$labels_val_dir"

declare -A seen

echo "Image train : $image_train_dir"
echo "Image val   : $image_val_dir"
echo "Labels train: $labels_train_dir"
echo "Labels val  : $labels_val_dir"
echo

# Process each image in image/val
for val_img in "$image_val_dir"/*; do
  [[ -f "$val_img" ]] || continue

  filename="$(basename "$val_img")"
  base="${filename%.*}"        # e.g., image_5 from image_5.png

  # Skip duplicate bases (if any)
  if [[ -n "${seen[$base]:-}" ]]; then
    continue
  fi
  seen[$base]=1

  echo ">>> Processing base: $base"

  # 1) Remove from image/train:
  #    - exact match: base.<ext> (e.g., image_4.png)
  #    - variants:    base_* (e.g., image_4_resize.png, image_4_zoom.png)
  train_matches=( "$image_train_dir/$base."* "$image_train_dir/${base}_"* )
  if (( ${#train_matches[@]} )); then
    rm -f -- "${train_matches[@]}"
    echo "Removed ${#train_matches[@]} image(s) from train for '$base'."
  else
    echo "No train images to remove for '$base'."
  fi

  # 2) Move labels from labels/train to labels/val:
  #    - exact:       base.png (e.g., image_4.png)
  #    - variants:    base_*.png (e.g., image_4_zoom.png, image_4_size.png)
  label_matches=( "$labels_train_dir/$base.png" "$labels_train_dir/${base}_"*.png )
  if (( ${#label_matches[@]} )); then
    mv -- "${label_matches[@]}" "$labels_val_dir"/
    echo "Moved ${#label_matches[@]} label file(s) to labels/val for '$base'."
  else
    echo "No labels to move for '$base'."
  fi

  echo
done

echo "Done!"