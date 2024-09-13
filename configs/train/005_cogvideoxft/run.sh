export TOKENIZERS_PARALLELISM=false
# dependencies
COGCKPT="/hcogVideo/CogVideoX-2b-sat/transformer/1000/mp_rank_00_model_states.pt"
# exp settings
EXPNAME="correctness"                              # experiment name 
CONFIG='configs/train/003_cogvideoxft/config.yaml' # experiment config 
RESROOT="results/cogvideo_train"                   # experiment saving directory

# run
current_time=$(date +%Y%m%d%H%M%S)
python scripts/train.py \
    -t \
    --name "$current_time"_$EXPNAME \
    --base $CONFIG \
    --logdir $RESROOT \
    --ckpt $COGCKPT \
    --devices '0,' \
    lightning.trainer.num_nodes=1 \
    --auto_resume True
