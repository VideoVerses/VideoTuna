export TOKENIZERS_PARALLELISM=false

CKPT="checkpoints/wan/Wan2.1-I2V-14B-480P"
CONFIG='configs/008_wanvideo/wan2_1_i2v_14B_480P_lora.yaml'        

RESROOT="results/train"                                             
EXPNAME="train_wanvideo_i2v_lora"                                     
CURRENT_TIME=$(date +%Y%m%d%H%M%S)                                  


python scripts/train_new.py -t \
--ckpt $CKPT \
--base $CONFIG \
--logdir $RESROOT \
--name "$EXPNAME"_"$CURRENT_TIME" \
--devices 0, \
--auto_resume