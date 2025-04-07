if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_folder> <output_folder> <size>"
  exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
SIZE="$3"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Loop through each image (you can adjust the pattern if needed)
for img in "$INPUT_FOLDER"/*.png; do
  [ -e "$img" ] || continue  # skip if no files match

  # Get the base name without extension
  filename=$(basename -- "$img")
  base="${filename%.*}"

  # Crop into 128x128 tiles
  convert "$img" -crop "$SIZE"x"$SIZE" +repage +adjoin "$OUTPUT_FOLDER/${base}_tile_%03d.png"
done