<p align="center" width="50%">
<img src="assets/logo.jpg" alt="VideoTuna" style="width: 30%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna
Let's finetune video generation models!


## ⏰TODOs
- [x] inference vc, dc   
- [x] finetune & train vc2，dc   
- [x] opensora-train, inference  
- [x] flux inference, fine-tune  
- [x] cogvideo inference, fine-tune  
- [x] merge diffusion parts
- [x] add peft lora 
- [x] add RL for alignment 
- [ ] refactor vc, opensora, cogvideo and flux 
- [ ] add documents 
- [ ] add unit test support 
- [ ] svd, open-sora-plan
- [ ] Finish codebase V1.0.0
- [ ] Release demo gallery
- [ ] Release technical report

## 🔆 Updates
- [2024-09-XX] We make the VideoTuna V1.0.0 public!


## 🔆 Introduction
🤗🤗🤗 VideoTuna is an open-sourcing finetuning framework for text-to-video generation.

### Features
1. All in one framework: Inference and finetune state-of-the-art T2V models.
2. Continuous training
3. Fintuning: domain-specific.
4. Fintuning: enhanced language understanding.
5. Fintuning: enhancement.
6. Human preference alignment/Post-training: RLHF, DPO.

### Code Structure
```
VideoTuna/
    ├── assets
    ├── checkpoints  # put model checkpoints here
    ├── configs      # model and experimental configs
    ├── eval         # evaluation scripts
    ├── inputs       # input examples for testing 
    ├── scripts      # train and inference python scripts
    ├── shsripts     # train and inference shell scripts
    ├── src          # model-related source code
    ├── tests        # testing scripts
    ├── tools        # some tool scripts
```

### Checkpoint Structure
```
VideoTuna/
    └── checkpoints/
        ├── cogvideo/
        │   └── CogVideoX-2b/        
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

### Models

|T2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-2B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b)
|CogVideoX-5B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)
|Open-Sora 1.2|240p to 720p, 2~16s|[STDIT](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3), [VAE](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2)
|Open-Sora 1.1|144p & 240p & 480p, 0~15s|[Stage 2](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2)
|Open-Sora 1.1|144p to 720p, 0~15s|[Stage 3](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3)
|Open-Sora 1.0|512×512x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)
|VideoCrafter2|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-5B-I2V|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
|DynamiCrafter|576x1024x16|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|

* Note: H: height; W: width; L: length

## 🔆 Get started

### 1.Prepare environment
```
conda create --name videotuna python=3.10 -y
conda activate videotuna
pip install -r requirements.txt
git clone https://github.com/JingyeChen/SwissArmyTransformer
pip install -e SwissArmyTransformer/
rm -rf SwissArmyTransformer
git clone https://github.com/tgxs002/HPSv2.git
pip install -e HPSv2/
rm -rf HPSv2
```

### 2.Prepare checkpoints
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
after these commands, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/tree/main?tab=readme-ov-file#checkpoint-structure).

### 3.Inference state-of-the-art T2V/I2V models

- Inference a set of models **in one command**:

    <!-- ```bash todo.sh``` -->

|Task|Commands|
|:---------|:---------|
|T2V|`bash tools/video_comparison/compare.sh`|
|I2V|`bash todo.sh`|



- Inference one specific model:

Task|Models|Commands|
|:---------|:---------|:---------|
|T2V|cogvideo|`bash shscripts/inference_cogVideo_diffusers.sh`|
|T2V|open-sora|@yazhou|
|T2V|videocrafter-v2-320x512|`bash shscripts/inference_vc2_t2v_320x512.sh`|
|(not working) T2V|videocrafter-v1-576x1024|`bash shscripts/inference_vc1_t2v_576x1024.sh`|
|I2V|dynamicrafter|`bash shscripts/inference_dc_i2v_576x1024.sh`|
|I2V|videocrafter1|`bash shscripts/inference_vc1_i2v_320x512.sh`|
|T2I|flux|`bash shscripts/inference_flux_schnell.sh`|

For detailed inference settings please check [docs/inference.md](docs/inference.md).

### 4. Finetune T2V models
#### Lora finetuning

We support lora finetuning to make the model to learn new concepts/characters/styles.   
- Example config file: `configs/train/003_vc2_lora_ft/config.yaml`  
- Training lora based on VideoCrafter2: `bash scripts/train/003_vc2_lora_ft/run.sh`  
- Inference the trained models: `bash scripts/train/003_vc2_lora_ft/inference.sh`   

Please check [configs/train/003_vc2_lora_ft/README.md](configs/train/003_vc2_lora_ft/README.md) for details.   
<!-- 

(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
``` -->

#### Finetuning for enhanced langugage understanding


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance. 
Please check [eval/README.md](eval/README.md) for details.

### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details.



## Acknowledgement
We thank the following repos for sharing their awsome models and codes!
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): Overcoming Data Limitations for High-Quality Video Diffusion Models
* [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Open Diffusion Models for High-Quality Video Generation
* [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All
* [CogVideoX](https://github.com/THUDM/CogVideo): Text-to-Video Diffusion Models with An Expert Transformer
* [VADER](https://github.com/mihirp1998/VADER): Video Diffusion Alignment via Reward Gradients
* [VBench](https://github.com/Vchitect/VBench): Comprehensive Benchmark Suite for Video Generative Models

## 🍻 Contributors

## 📋 License

## 😊 Citation
```
To be updated...
```
