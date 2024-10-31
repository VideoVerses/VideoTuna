
# Prepare checkpoints

This document contains commands for preparing model checkpoints and the final checkpoint orgnization structure.

### Download checkpoints
Please run the following commands in your terminal to download the checkpoints for each model.
```
mkdir checkpoints

# ---------------------------- T2V ----------------------------

# ---- CogVideo (diffusers) ----
cd checkpoints/cogvideo
git clone https://huggingface.co/THUDM/CogVideoX-2b
git clone https://huggingface.co/THUDM/CogVideoX-5b
git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V


# ---- Open-Sora ----
mkdir -p checkpoints/open-sora/t2v_v10
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
#
mkdir -p checkpoints/open-sora/t2v_v11
cd checkpoints/open-sora/t2v_v11
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3
cd ../../..
#
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3


# ---- Videocrafter ----
mkdir checkpoints/videocrafter/

mkdir checkpoints/videocrafter/t2v_v2_512
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v2_512  # videocrafter2-t2v-512

mkdir checkpoints/videocrafter/t2v_v1_1024
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v1_1024 # videocrafter1-t2v-1024


# ---------------------------- I2V ----------------------------
# ---- Dynamicrafter ----
mkdir checkpoints/dynamicrafter/
mkdir checkpoints/dynamicrafter/i2v_576x1024

wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt -P checkpoints/dynamicrafter/i2v_576x1024  # dynamicrafter-i2v-1024

# ---- Videocrafter ----
mkdir checkpoints/videocrafter/
mkdir checkpoints/videocrafter/i2v_v1_512

wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt -P checkpoints/videocrafter/i2v_v1_512 # videocrafter1-i2v-512

# ---- Stable Diffusion checkpoint for VC2 Training ----
mkdir checkpoints/stablediffusion/
mkdir checkpoints/stablediffusion/v2-1_512-ema

wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt -P checkpoints/stablediffusion/v2-1_512-ema

```

### Checkpoint Orgnization Structure
After downloading, the model checkpoints should be placed as follows:  

```
VideoTuna/
    └── checkpoints/
        ├── cogvideo/
        │   └── CogVideoX-2b/   
        │   └── CogVideoX-5b/        
        │   └── CogVideoX-5b-I2V/        
        ├── dynamicrafter/
        │   └── i2v_576x1024/
        │       └── model.ckpt
        ├── videocrafter/
        │   ├── t2v_v2_512/
        │   │   └── model.ckpt
        │   ├── t2v_v1_1024/
        │   │   └── model.ckpt
        │   └── i2v_v1_512/
        │       └── model.ckpt
        └── open-sora/
            ├── t2v_v10/
            │   ├── OpenSora-v1-16x256x256.pth
            │   └── OpenSora-v1-HQ-16x512x512.pth
            ├── t2v_v11/
            │   ├── OpenSora-STDiT-v2-stage2/
            │   └── OpenSora-STDiT-v2-stage3/
            └── t2v_v12/
                ├── OpenSora-STDiT-v3/
                └── OpenSora-VAE-v1.2/
```

If you do not follow these locations, please modify the default checkpoint path argument during training/inference.
