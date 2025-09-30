#!/bin/bash
#SBATCH -J diffuseIT
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yifanlu@kth.se
#SBATCH --output /proj/berzelius-2024-331/users/x_yiflu/DiffuseIT/logs/%J_main.out
#SBATCH --error  /proj/berzelius-2024-331/users/x_yiflu/DiffuseIT/logs/%J_main.err

conda activate DiffuseIT
cd /proj/berzelius-2024-331/users/x_yiflu/DiffuseIT

# ---- config ----
SRC_DIR="/home/x_yiflu/dataset/Datasets/ffhq/ffhq_1k"             # 1k FFHQ subset
PROMPT="${1:-Make Up}"               # pass a prompt as arg; default "Make up"
RAW_OUT="outputs/_raw_out"           # where main.py writes output_i_0_b_0.png
# ----------------

# slugify prompt for a folder name: "Make up" -> "make_up"
PROMPT_SLUG=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g;s/^_|_$//g')
OUT_DIR="/home/x_yiflu/dataset/Datasets/ffhq/DiffuseIT_ffhq_${PROMPT_SLUG}"

mkdir -p "$OUT_DIR" "$RAW_OUT"

for img in "$SRC_DIR"/*.png; do
  fname=$(basename "$img")  # keep original filename, e.g. 69999.png

  /proj/berzelius-2024-331/users/x_yiflu/.conda/envs/DiffuseIT/bin/python main.py \
    -p "$PROMPT" \
    -s "Person" \
    -i "$img" \
    --output_path "$RAW_OUT" \
    --use_range_restart \
    --use_noise_aug_all \
    --regularize_content \
    --use_ffhq \
    --iterations_num 1

  # move + rename the generated file to match input filename
  if [ -f "$RAW_OUT/output_i_0_b_0.png" ]; then
    mv "$RAW_OUT/output_i_0_b_0.png" "$OUT_DIR/$fname"
  else
    echo "WARN: no output for $img" >&2
  fi
done

echo "Done. Results in: $OUT_DIR"
