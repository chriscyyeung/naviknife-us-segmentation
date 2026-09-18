#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --job-name=ln-openus-test-fold0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

# ----------------------------
# Run training
# ----------------------------

module load gcc opencv

source /home/cyeung/projects/aip-medilab/cyeung/venv/bin/activate

python test_segmentation.py \
    --arch vmamba_small \
    --dataset_name LumpNav \
    --data_root /home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV/fold_0 \
    --fold 0 \
    --num_workers 4 \
    --pretrained_vmamba True \
    --pretrained_weights /home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/OpenUS/pretrained/openus_cpt0150.pth \
    --output_dir /home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/output/OpenUS/fold_0