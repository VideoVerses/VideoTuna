export TOKENIZERS_PARALLELISM=false

CKPT="checkpoints/wan_test/Wan2.1-T2V-14B"
CONFIG='configs/008_wanvideo/wan2_1_t2v_14B_lora.yaml'        

RESROOT="results/train"                                             
EXPNAME="train_wanvideo_t2v_lora"                                     
CURRENT_TIME=$(date +%Y%m%d%H%M%S)                                  


python scripts/train_new.py -t \
--ckpt $CKPT \
--base $CONFIG \
--logdir $RESROOT \
--name "$EXPNAME"_"$CURRENT_TIME" \
--devices 0, \
--auto_resume