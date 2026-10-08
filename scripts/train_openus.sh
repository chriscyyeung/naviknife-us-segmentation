#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --job-name=ln-openus-train-fold4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:4
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

module load gcc cuda/12.9
module load gcc opencv

source /home/cyeung/projects/aip-medilab/cyeung/venv/bin/activate

mkdir -p /home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/output/OpenUS/fold_4

# Copy dataset to fast local scratch before training
echo "Copying dataset to \$SLURM_TMPDIR..."
cp -r /home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV/fold_4 "$SLURM_TMPDIR/"
echo "Copy complete."

srun python eval_segmentation.py \
    --arch vmamba_small \
    --dataset_name LumpNav \
    --data_root $SLURM_TMPDIR/fold_4 \
    --fold 4 \
    --num_workers 8 \
    --pretrained_vmamba True \
    --pretrained_weights /home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/OpenUS/pretrained/openus_cpt0150.pth \
    --output_dir /home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/output/OpenUS/fold_4 \
    --load_from checkpoint_teacher_seg_best.pth \
    --log_name OpenUSLumpnavF4
