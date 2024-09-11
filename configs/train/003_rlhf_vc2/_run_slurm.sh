#!/bin/bash -l

##########################################################################################################
# slurm-related args
#SBTACH --job-name=cnn_unet_video_16node
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH -p project   #important and necessary
#SBATCH --gres=gpu:8
# SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=24:00:00 # must set the training time by default. 24h max...
#SBATCH --cpus-per-task=8
#SBATCH --output=srun_output/_%j/output.txt
#SBATCH --error=srun_output/_%j/error.txt
#SBATCH --signal=SIGUSR1@90 # reboot if the process is killed..

# debugging flags (optional)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "numnodes:"$nodes

nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export WANDB_MODE="offline"
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_DEBUG=TRACE
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="^docker0,lo,bond0"
export MASTER_PORT=12345
# export WORLD_SIZE=$SLURM_NTASKS
# export LOCAL_RANK=$SLURM_LOCALID
# export RANK=$SLURM_LOCALID
# export NODE_RANK=$SLURM_PROCID
export MASTER_ADDR=$head_node_ip
export WORK_DIR=../
export PYTHONPATH=$WORK_DIR


# echo "WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK, NODE_RANK=$NODE_RANK, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="overfit_t2v512_dpo"                            # experiment name 
CONFIG='configs/train/000_videocrafter2ft/config.yaml' # experiment config 
LOGDIR="./results_dpo"                                     # experiment saving directory

# run
python scripts/train.py \
-t --devices '8' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume False

##########################################################################################################
# code-related args
# export TOKENIZERS_PARALLELISM=false
# CKPT_RESUME="/project/llmsvgen/share/shared_ckpts/VideoCrafter/VideoCrafter2-Text2Video-512/model.ckpt" # ckpt for resume
# current_time=$(date +%Y%m%d%H%M%S) # for experiment name 

# EXPNAME="run_macvid_t2v512"                            # experiment name 
# CONFIG='configs/train/000_videocrafter2ft/config.yaml' # experiment config 
# LOGDIR="./results"                                     # experiment saving directory

# # run
# srun python scripts/train.py \
# -t --devices '0,1,2,3,4,5,6,7', \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True \
# --load_from_checkpoint $CKPT_RESUME 
