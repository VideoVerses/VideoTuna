export TOKENIZERS_PARALLELISM=false

current_time=$(date +%Y%m%d%H%M%S)

EXPNAME="run_macvid_t2v512"                            # experiment name 
CONFIG='configs/train/000_videocrafter2ft/config_opensora2B.yaml' # experiment config 
LOGDIR="./results"                                     # experiment saving directory

# run
# python scripts/train.py \
# -t --devices '0,1' \
# lightning.trainer.num_nodes=1 \
# --base $CONFIG \
# --name "$current_time"_$EXPNAME \
# --logdir $LOGDIR \
# --auto_resume True

python scripts/train.py \
-t --devices '0,' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume True