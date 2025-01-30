export TOKENIZERS_PARALLELISM=false

# dependencies
CONFIG="configs/004_cogvideox/cogvideo5b-i2v.yaml"   # experiment config 

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="results/train"             # experiment saving directory
EXPNAME="cogvideox_i2v_5b"          # experiment name 
CURRENT_TIME=$(date +%Y%m%d%H%M%S)  # current time

# run
python scripts/train.py \
-t \
--base $CONFIG \
--logdir $RESROOT \
--name "$CURRENT_TIME"_$EXPNAME \
--devices '0,' \
lightning.trainer.num_nodes=1 \
--auto_resume