#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --job-name=ln-insid3-lora
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:00:00
#SBATCH --output=/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/logs/%x_fold%a_%A.out
#SBATCH --error=/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/logs/%x_fold%a_%A.err
#SBATCH --mail-user=chriscyyeung@gmail.com
#SBATCH --mail-type=END,FAIL

# LoRA-adapt INSID3's DINOv3 ViT-L backbone on one CV fold (array index = fold), then run
# INSID3 1-shot and 5-shot on that fold's validation patients with the merged backbone.
#   sbatch scripts/train_insid3_lora.sh                     # all folds
#   sbatch --array=0 scripts/train_insid3_lora.sh           # one fold
#   EXTRA="--steps 100 --eval-every 50 --eval-targets 30" INFER=0 OUT_NAME=insid3_lora_smoke \
#     sbatch --array=0 --time=2:00:00 --export=ALL scripts/train_insid3_lora.sh   # smoke: timing + memory only
# Outputs: external/INSID3/output/insid3_lora/fold_<k>/{train,pred_masks_1shot,pred_masks_5shot}
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
REPO=/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation
INSID3=$REPO/external/INSID3
CV=/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV
OUT=$INSID3/output/${OUT_NAME:-insid3_lora}/fold_$FOLD
EXTRA=${EXTRA:-}
INFER=${INFER:-1}

source "$INSID3/.venv/bin/activate"
cd "$INSID3"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Fold folders are symlinks into the full dataset: -L copies the real files to local scratch
echo "Copying fold $FOLD to \$SLURM_TMPDIR..."
mkdir -p "$SLURM_TMPDIR/cv"
cp -rL "$CV/fold_$FOLD" "$SLURM_TMPDIR/cv/"
echo "Copy complete."

python train_lora_lumpnav.py \
    --data-root "$SLURM_TMPDIR/cv" --fold "$FOLD" \
    --model-size large --image-size 1024 \
    --lora-last-blocks 12 --lora-targets qkv,proj --rank 16 --alpha 32 \
    --steps 2000 --episodes-per-step 4 --lr 2e-4 \
    --eval-every 250 --eval-targets 100 --eval-shots 1 5 \
    --grad-ckpt --num-workers 6 \
    --output-dir "$OUT/train" $EXTRA

if [ "$INFER" = 0 ]; then echo "INFER=0: skipping inference"; exit 0; fi
WEIGHTS=$(ls "$OUT"/train/dinov3_vitl16_lumpnav_lora_fold${FOLD}-*.pth)

for SHOTS in 1 5; do
    python inference_segmentation.py --dataset lumpnav --data-root "$SLURM_TMPDIR/cv" --fold "$FOLD" \
        --shots "$SHOTS" --model-size large --image-size 1024 --encoder-weights "$WEIGHTS" \
        --save-preds --output-dir "$OUT" --exp-name "${SHOTS}shot"
    # Drop the timestamp so compute_seg_metrics.py can use a {fold} template
    rm -rf "$OUT/pred_masks_${SHOTS}shot"
    mv "$(ls -d "$OUT/${SHOTS}shot_"* | tail -1)/pred_masks" "$OUT/pred_masks_${SHOTS}shot"
done
echo "Done fold $FOLD"
