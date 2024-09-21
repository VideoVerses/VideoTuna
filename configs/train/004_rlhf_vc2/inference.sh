# lora evaluate code  
ckpt='/home/liurt/liurt_data/haoyu/DPO-videocrafter/init_ckpt/vc2/model.ckpt'
# ckpt='/home/liurt/liurt_data/haoyu/DPO-videocrafter/results/elon.ckpt'
# config='../VideoCrafter/configs/inference_t2v_512_v2.0.yaml'
config='configs/inference/lora_inference_t2v_512_v2.0.yaml'
prompt_file="/home/liurt/liurt_data/haoyu/cache_code/DPO-videocrafter/lvdm/models/rlhf_utils/assets/chatgpt_custom_instruments_unseen.txt"

res_dir="rlhf-visual-results"
name="lora_aes_chatgpt_instructions-3184-unseen"

lora_path="/home/liurt/liurt_data/haoyu/DPO-videocrafter/results/dpo-vc2-12th-lora/20240905140559_total_score_75P/checkpoints/trainstep_checkpoints/epoch=000000-step=000000470.ckpt"
#--------------------------base model the same; lora rank defaullt 4 --------------------------
lora_path="/home/liurt/liurt_data/haoyu/cache_code/DPO-videocrafter/results/reward-vc2-1th/lora_aes_chatgpt_instructions/checkpoints/trainstep_checkpoints/step-step=3184.ckpt"


python3 scripts/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28 \
--lora_ckpt $lora_path
