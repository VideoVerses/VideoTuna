[] logo

# NewVid
[] Slogan

## TODOs
[] inference vc2
[] finetune vc2
[] train vc2

## Updates

## What we have
### Features
1. Continuous pretraining.
1. Domain-specific finetuning: human, cartoon, robotics, autonomous driving, etc.
1. Concept-specific finetuning: character, style, etc.
1. Human preference alignment post-training: RLFH, DPO.
1. Post-processing: enhancement.

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
|DynamiCrafter|TODO|[Hugging Face](TODO)
|VideoCrafter1|640x1024|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)
|VideoCrafter1|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)



## Get started

### New a environment
```
conda create -n videocrafter python=3.8
pip install -r requirements_vc.txt
conda activate videocrafter
```

### Prepare checkpoints
```
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt   # videocrafter2-512
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt # videocrafter1-1024
wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt   # dynamicrafter-1024
```

### New a video
Before running the following scripts, make sure you download the checkpoint and put it at `checkpoints/videocrafter/base_512_v2/model.ckpt`.
```
bash scripts/inference_t2v_vc2.sh
```

### New a model
1. Prepare data


2. Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
```

### Evaluation

## Contributors

## License

## Citation

