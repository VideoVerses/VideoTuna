export TOKENIZERS_PARALLELISM=false


# dependencies
SDCKPT="checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"
VC2CKPT="checkpoints/videocrafter/t2v_v2_512/model.ckpt"
# LORACKPT="checkpoints/lora/512/lora.ckpt"

# exp settings
EXPNAME="train_t2v_512_lora"                            # experiment name 
CONFIG='configs/001_videocrafter2/vc2_t2v_lora.yaml' # experiment config 
RESROOT="results/train"                               # experiment saving directory

# run
current_time=$(date +%Y%m%d%H%M%S)
python scripts/train.py \
-t \
--name "$current_time"_$EXPNAME \
--base $CONFIG \
--logdir $RESROOT \
--ckpt $VC2CKPT \
--devices '0,' \
lightning.trainer.num_nodes=1 \
--auto_resume 



