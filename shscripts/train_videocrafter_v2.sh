export TOKENIZERS_PARALLELISM=false

# dependencies
SDCKPT="checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"        # pretrained checkpoint of stablediffusion 2.1
VC2CKPT="checkpoints/videocrafter/t2v_v2_512/model_converted.ckpt"  # pretrained checkpoint of videocrafter2
CONFIG='configs/001_videocrafter2/vc2_t2v_320x512.yaml'             # experiment config: model+data+training

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="results/train"                                             # root directory for saving multiple experiments
EXPNAME="videocrafter2_320x512"                                     # experiment name 
CURRENT_TIME=$(date +%Y%m%d%H%M%S)                                  # current time

# run
python scripts/train.py \
-t \
--sdckpt $SDCKPT \
--ckpt $VC2CKPT \
--base $CONFIG \
--logdir $RESROOT \
--name ${CURRENT_TIME}_${EXPNAME} \
--devices '0,' \
lightning.trainer.num_nodes=1 \
--auto_resume
