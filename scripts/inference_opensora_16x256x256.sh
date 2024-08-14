torchrun --standalone --nproc_per_node 1 scripts/inference_opensora.py \
configs/inference/opensora_16x256x256.py \
--save-dir samples/examples \
--prompt-path configs/inference/test_prompt.txt \
--ckpt-path /project/llmsvgen/share/opensora_ckpt/stage5/last