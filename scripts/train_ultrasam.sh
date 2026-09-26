#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --job-name=ln-ultrasam-train-fold4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/logs/%x_%j.out
#SBATCH --error=/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/logs/%x_%j.err
#SBATCH --mail-user=chriscyyeung@gmail.com
#SBATCH --mail-type=ALL

# ---------------- DDP environment ----------------

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR

MASTER_PORT=$((10000 + SLURM_JOB_ID % 10000))
export MASTER_PORT

WORLD_SIZE=$SLURM_NTASKS
export WORLD_SIZE

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

srun bash -c '
echo "RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID HOST=$(hostname) MASTER=$MASTER_ADDR:$MASTER_PORT"
'

# ----------------------------
# Run training
# ----------------------------

module load gcc opencv

source /home/cyeung/projects/aip-medilab/cyeung/venv/bin/activate

# Copy dataset to fast local scratch before training
echo "Copying dataset to \$SLURM_TMPDIR..."
cp -r /home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV/fold_4 "$SLURM_TMPDIR/"
echo "Copy complete."

srun mim train mmdet configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --cfg-options data_root="$SLURM_TMPDIR" \
    --gpus 1 \
    --launcher pytorch \
    --work-dir ./work_dirs/UltraSAM_box_refine_fold4
