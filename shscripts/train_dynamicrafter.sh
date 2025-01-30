export TOKENIZERS_PARALLELISM=false


# dependencies
SDCKPT="checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"
# DCCKPT="checkpoints/dynamicrafter/i2v_576x1024/model.ckpt"
DCCKPT="checkpoints/dynamicrafter/i2v_576x1024/model_converted.ckpt"

EXPNAME="002_dynamicrafterft_1024"                            # experiment name 
CONFIG='configs/002_dynamicrafter/dc_i2v_1024.yaml' # experiment config 
RESROOT="results/train"                               # experiment saving directory

# run
current_time=$(date +%Y%m%d%H%M%S)
python scripts/train.py \
-t \
--name "$current_time"_$EXPNAME \
--base $CONFIG \
--logdir $RESROOT \
--sdckpt $SDCKPT \
--ckpt $DCCKPT \
--devices '0,' \
lightning.trainer.num_nodes=1 \
--auto_resume

