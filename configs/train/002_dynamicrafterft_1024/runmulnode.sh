export TOKENIZERS_PARALLELISM=false


# dependencies
SDCKPT="checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"
VC2CKPT="checkpoints/videocrafter/base_512_v2/model.ckpt"

# exp settings
EXPNAME="run_macvid_t2v512"                            # experiment name 
CONFIG='configs/train/000_videocrafter2ft/config.yaml' # experiment config 
RESROOT="results/train"                               # experiment saving directory

# gpu ddp settings
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HOST_GPU_NUM=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1

# run
current_time=$(date +%Y%m%d%H%M%S)
python -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=12352 --node_rank=$NODE_RANK \
scripts/train.py \
-t \
--name "$current_time"_$EXPNAME \
--base $CONFIG \
--logdir $RESROOT \
--sdckpt $SDCKPT \
--ckpt $VC2CKPT \
--devices $CUDA_VISIBLE_DEVICES \
lightning.trainer.num_nodes=$NUM_NODES \
--auto_resume True

