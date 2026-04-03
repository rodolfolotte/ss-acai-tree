if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <input_folder> <output_folder> <size> <overlap>"
  exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
SIZE="$3"
OVERLAP="$4"

mkdir -p "$OUTPUT_FOLDER"

for img in "$INPUT_FOLDER"/*.tif; do
  [ -e "$img" ] || continue

  WIDTH=$(identify -format "%w" "$img")
  HEIGHT=$(identify -format "%h" "$img")

  STEP=$((SIZE - OVERLAP))
  BASE=$(basename "$img" | cut -d. -f1)

  tile_id=0
  for ((y=0; y<HEIGHT; y+=STEP)); do
    for ((x=0; x<WIDTH; x+=STEP)); do
      crop_width=$TILE_SIZE
      crop_height=$TILE_SIZE

      # If tile goes beyond image edge, adjust crop area (then pad to SIZE)
      if (( x + SIZE > WIDTH )); then
        crop_width=$((WIDTH - x))
      fi
      if (( y + SIZE > HEIGHT )); then
        crop_height=$((HEIGHT - y))
      fi

      convert "$img"["${crop_width}x${crop_height}+$x+$y"] \
      -background black -gravity northwest -extent ${SIZE}x${SIZE} \
      "$OUTPUT_FOLDER/${BASE}_tile_$(printf "%03d" $tile_id).tif"

      ((tile_id++))
    done
  done
done