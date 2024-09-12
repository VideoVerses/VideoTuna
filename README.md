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
- [ ] merge diffusion parts
- [ ] refactor vc, opensora, cogvideo and flux 
- [ ] add peft lora 
- [ ] add RL for alignment 
- [ ] add documents 
- [ ] add unit test support 

## 🔆 Updates



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
VideoTuna
├── configs
│ ├── model_name_inf.yaml
│ └── model_name_train.yaml
├── checkpoints
├── docs
├── inputs
├── results
├── src
│ ├── dataset
│ ├── model-1
│ ├── model-2
│ └── model-N
├── scripts
│ ├── inference_xxx.py
│ └── train_xxx.py
├── shscripts
│ ├── inference_xxx.sh
│ └── train_xxx.sh
├── utils
└── test

```

### Models

|T2V-Models|Resolution|Checkpoints|
|:---------|:---------|:--------|
|Open-Sora 1.2|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|Open-Sora 1.1|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|Open-Sora 1.0|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|Open-Sora Plan 1.2.0|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|Open-Sora Plan 1.1.0|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|Open-Sora Plan 1.0.0|TODO|[TODO](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter2|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|Resolution|Checkpoints|
|:---------|:---------|:--------|
|DynamiCrafter|576x1024|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|640x1024|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|xG



## 🔆 Get started

### 1.Prepare environment
```
conda create --name videotuna python=3.10 -y
conda activate videotuna
pip install -r requirements.txt
git clone https://github.com/JingyeChen/SwissArmyTransformer
pip install -e SwissArmyTransformer/
rm -rf SwissArmyTransformer
```

### 2.Prepare checkpoints
```
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt   # videocrafter2-t2v-512
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt # videocrafter1-t2v-1024
wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt # videocrafter1-i2v-512
wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt   # dynamicrafter-i2v-1024

```

### 3.Inference state-of-the-art T2V models
#### (1) VideoCrafter
Before running the following scripts, make sure you download the checkpoint and put it at `checkpoints/videocrafter/base_512_v2/model.ckpt`.
```
bash scripts/inference_t2v_vc2.sh
```
#### (2) Open-Sora


#### (3) Cogvideo

#### (4) DynamiCrafter
#### (5) Flux 

### 4. Finetune T2V models
#### Finetuning for specific domains 

#### Finetuning for enhanced langugage understanding 

#### Finetuning for generative video enhancement

(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
```

### 5. Evaluation


### 6. Alignment


## 🍻 Contributors

## 📋 License

## 😊 Citation

