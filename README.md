[] logo

# VideoTuna
Let's finetune video generation models!

## 🔆 Updates


## 🔆 Introduction
🤗🤗🤗 VideoTuna is an open-sourcing finetuning framework for text-to-video generation.

### Features
1. All in one framework: Inference and finetune state-of-the-art T2V models.
2. T2V Pretraining.
1. Domain-specific finetuning.
1. Human preference alignment/Post-training: RLFH, DPO.
1. Post-processing: enhancement.

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
conda create -n videocrafter python=3.8
pip install -r requirements_vc.txt
conda activate videocrafter
```

### 2.Prepare checkpoints
```
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt   # videocrafter2-t2v-512
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt # videocrafter1-t2v-1024
wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt # videocrafter1-i2v-512
wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt   # dynamicrafter-i2v-1024

```

### 3.Inference state-of-the-art T2V models
#### (1) Cogvideo
#### (2) Open-Sora

#### (3) VideoCrafter
Before running the following scripts, make sure you download the checkpoint and put it at `checkpoints/videocrafter/base_512_v2/model.ckpt`.
```
bash scripts/inference_t2v_vc2.sh
```


### 4. Finetune T2V models
(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
```

### 5. Evaluation


## ⏳ TODOs
[x] inference vc, dc   
[x] finetune & train vc2，dc   
[x] opensora-train, inference  
[ ] dpo, lora  
[ ] flux inference, fine-tune  
[ ] cogvideo inference, fine-tune  
[ ] vae  
next:  
[ ] inference dc interp & loop  

## 🍻 Contributors

## 📋 License

## 😊 Citation

