[] logo

# NewVid
[] Slogan

## TODOs
[] inference vc2
[] finetune vc2
[] train vc2

## Updates

## What we have
Features & Examples
Models

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

