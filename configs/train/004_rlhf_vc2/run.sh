# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=4
current_time=$(date +%Y%m%d%H%M%S)

CONFIG='configs/train/004_rlhf_vc2/config.yaml' # experiment config 
LOGDIR="./results/reward-vc2-1th"   # experiment saving directory all should under subfolder so that won't be copied to codeversion
EXPNAME="lora_aes_chatgpt_instructions"

# ### run
python scripts/train.py \
-t --devices '0,' \
lightning.trainer.num_nodes=1 \
--base $CONFIG \
--name "$current_time"_$EXPNAME \
--logdir $LOGDIR \
--auto_resume True \
--gpu_num 1
