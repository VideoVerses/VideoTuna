
# Introduction
- This document provides instructions for fine-tuning the WanVideo T2V and I2V model.
- It supports both full fine-tuning and lora fine-tuning.

| Model    | Avg. Epoch Time (s) | Avg. Peak Memory (GB) |
| -------- | ------------------- | ---------------------- |
| i2v Full | 57.76               | 48.41                  |
| i2v LoRA | 45.19               | 38.45                  |
| t2v Full | 51.04               | 43.59                  |
| t2v LoRA | 40.89               | 37.79                  |

# Preliminary steps
  1) [Install the environment](#1prepare-environment)
  2) To use deepspeed Zero3 training, please review the following preparation steps.
```shell
poetry run install-deepspeed
```
  3) Download the example training data.
You can download manually from [this link](https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip), or download via `wget`:
```
wget https://huggingface.co/datasets/Yingqing/VideoTuna-Datasets/resolve/main/apply_lipstick.zip
cd data
unzip apply_lipstick.zip -d apply_lipstick
```
Make sure the data is putted at `data/apply_lipstick/metadata.csv`

  4) [Download the checkpoints](docs/CHECKPOINTS.md) and get the checkpoint
```
  $ ll checkpoints/wan/Wan2.1-T2V-14B
  $ ll checkpoints/wan/Wan2.1-I2V-14B-480P
```

# Steps of Simple Fine-tuning
**1. Full Fine-tuning of WanVideo Text-to-Video:**

**(1) Train:** Run this command to start training on the single GPU. 
```
bash shscripts/train_wanvideo_t2v_fullft.sh
```
or
```
poetry run train-wan2-1-t2v-fullft
```

The training results will be automatically saved at `results/train/train_wanvideo_t2v_fullft_${CURRENT_TIME}_${EXPNAME}`. The checkpoints will be save at `results/train/train_wanvideo_t2v_fullft_${CURRENT_TIME}_${EXPNAME}/checkpoints/only_trained_model/denoiser-$epoch-$step.ckpt` every 50 iteractions. Saving checkpoints is time consuming, you can increase every_n_train_steps in `configs/008_wanvideo/wan2_1_t2v_14B_fullft.yaml` callbacks section

**(2) Inference:**  Remember replace trained_ckpt
```
bash shscripts/inference_wanvideo_t2v_fullft.sh
```

**2. Lora Fine-tuning of WanVideo Text-to-Video:**  
**(1) Train:**

```
bash shscripts/train_wanvideo_t2v_lora.sh
```
The training results will be automatically saved at `results/train/train_wanvideo_t2v_lora_${CURRENT_TIME}_${EXPNAME}`. The checkpoints will be save at `results/train/train_wanvideo_t2v_lora_${CURRENT_TIME}_${EXPNAME}/checkpoints/only_trained_model/denoiser-$epoch-$step.ckpt` every 50 iteractions. Saving checkpoints is time consuming, you can increase every_n_train_steps in `configs/008_wanvideo/wan2_1_t2v_14B_lora.yaml` callbacks section

or
```
poetry run train-wan2-1-t2v-lora
```


**(2) Inference:** Remember replace trained_ckpt
```
bash shscripts/inference_wanvideo_t2v_lora.sh
```

**3. Full Fine-tuning of WanVideo Image-to-Video:**

**(1) Train:** Run this command to start training on the single GPU. 
```
bash shscripts/train_wanvideo_i2v_fullft.sh
```
or
```
poetry run train-wan2-1-i2v-fullft
```

The training results will be automatically saved at `results/train/train_wanvideo_i2v_fullft_${CURRENT_TIME}_${EXPNAME}`. The checkpoints will be save at `results/train/train_wanvideo_i2v_fullft_${CURRENT_TIME}_${EXPNAME}/checkpoints/only_trained_model/denoiser-$epoch-$step.ckpt` every 50 iteractions. Saving checkpoints is time consuming, you can increase every_n_train_steps in `configs/008_wanvideo/wan2_1_i2v_14B_480P_fullft.yaml` callbacks section

**(2) Inference:**  Remember replace trained_ckpt
```
bash shscripts/inference_wanvideo_i2v_fullft.sh
```

**4. Lora Fine-tuning of WanVideo Image-to-Video:**  
**(1) Train:**

```
bash shscripts/train_wanvideo_i2v_lora.sh
```
The training results will be automatically saved at `results/train/train_wanvideo_i2v_lora_${CURRENT_TIME}_${EXPNAME}`. The checkpoints will be save at `results/train/train_wanvideo_i2v_lora_${CURRENT_TIME}_${EXPNAME}/checkpoints/only_trained_model/denoiser-$epoch-$step.ckpt` every 50 iteractions. Saving checkpoints is time consuming, you can increase every_n_train_steps in `configs/008_wanvideo/wan2_1_i2v_14B_480P_lora.yaml` callbacks section

or
```
poetry run train-wan2-1-i2v-lora
```


**(2) Inference:** Remember replace trained_ckpt
```
bash shscripts/inference_wanvideo_i2v_lora.sh
```
