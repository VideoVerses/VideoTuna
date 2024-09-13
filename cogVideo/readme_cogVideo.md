# CogVideo

# Diffusers

## Model

| T2V-Models   | Resolution | Checkpoints                                                         |
|--------------|------------|---------------------------------------------------------------------|
| CogVideoX-2b | 720x480    | [Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b/tree/main) |
 | CogVideoX-5b | 720x480    | [Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b/tree/main) |



## Get started

### Set up environment

```
conda create -n cogvideo python=3.10
conda activate cogvideo
```


### Install dependencies
```
cd cogVideo
pip install -r requirements_cogVideo.txt
pip install --upgrade opencv-python transformers diffusers # Must using diffusers>=0.30.0
```

### Prepare checkpoints
#### Go to the checkpoints folder
```
mkdir checkpoints/cogvideo
cd checkpoints/cogvideo
```

#### Download VAE model
```
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
rm vae.zip
```

#### Clone the T5 model
```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
rm -r CogVideoX-2b
```

#### CogVideoX-2b
Use the following command to clone the repository and download the checkpoints. 
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-2b
```
#### CogVideoX-5b
Use the following command to clone the repository and download the checkpoints.
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b
```

#### CogVideoX-2b-sat

```
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
rm transformer.zip
```

#### CogVideoX-5b-sat
Use the following command to clone the repository and download the checkpoints.
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b) to download the checkpoints.
```
mkdir CogVideoX-5b-sat
cd CogVideoX-5b-sat
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1"
mv 'index.html?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1' transformer.tar.gz
tar -xzvf transformer.tar.gz
rm transformer.tar.gz
```
The model structure should be as follows:
```
.
├── vae
├── t5-v1_1-xxl
├── CogVideoX-2b
├── CogVideoX-5b
├── CogVideoX-2b-sat
│   └── transformer
│       └── 1000 (or 1)
│           └── mp_rank_00_model_states.pt
│       └── latest
├── CogVideoX-5b-sat
│   └── transformer
│       └── 1000 (or 1)
│           └── mp_rank_00_model_states.pt
│       └── latest

```

### Generate video
Parameters:
- prompt (str): The description of the video to be generated.
- model_path (str): The path of the pre-trained model to be used.
- output_path (str): The path where the generated video will be saved.
- num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
- guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
- num_videos_per_prompt (int): Number of videos to generate per prompt.
- dtype (torch.dtype): The data type for computation (default is torch.float16).


### Example
```
python inference_cogVideo_diffusers.py --prompt "A video of a cat playing with a ball" --model_path /path/to/CogVideoX-2b --output_path output.mp4
```
OR
```
bash ./shscripts/inference_cogVideo_diffusers.sh
```
It will generate a video of a cat playing with a ball and save it to the file cogVideo/output.mp4, with the model CogVideoX-2b.

# SAT Inference
* Single GPU Inference for 2b version (FP16), the model need 18GB GPU memory. 
* Single GPU Inference for 5b version (BF16), the model need 26GB GPU memory.


## For 2b version
### Modify the file in ```./src/sat/configs/cogvideox_2b.yaml```.
- Change the model path:
```
model_dir: "{absolute_path/to/your/t5-v1_1-xxl}/t5-v1_1-xxl" # Absolute path to the t5-v1_1-xxl weights folder
```
- Change the ckpt path: 
```
ckpt_path: "{absolute_path/to/your/t5-v1_1-xxl}/CogVideoX-2b-sat/vae/3d-vae.pt" # Absolute path to the CogVideoX-2b-sat/vae/3d-vae.pt
```


### Modify the file in ```./src/sat/configs/inference.yaml```.
- Modify the path of transformer: 
```
load: "{absolute_path/to/your}/transformer" # Absolute path to the CogVideoX-2b-sat/transformer folder
```

## For 5b version
### Modify the file in ```./src/sat/configs/cogvideox_5b.yaml```.
- Change the model path: 
```
model_dir: "{absolute_path/to/your/t5-v1_1-xxl}/t5-v1_1-xxl" # Absolute path to the t5-v1_1-xxl weights folder
```
- Change the ckpt path: 
```
ckpt_path: "{absolute_path/to/your/t5-v1_1-xxl}/CogVideoX-5b-sat/vae/3d-vae.pt"
```


### Modify the file in ```./src/sat/configs/inference.yaml```.
- Modify the load path: 
```
load: "{absolute_path/to/your}/transformer" # Absolute path to the CogVideoX-5b-sat/transformer folder
```


#### Remarks
If you want to use your own prompt, you can modify the file in ```./src/sat/configs/test.txt```.
If multiple prompts is required, in which each line makes a prompt.

### Generate video
Go to file ```./src/sat```, and run the following command:
```
bash shscripts/inference_cogVideo_sat.sh
```


# SAT Finetune

The memory consumption per GPU:

|            | CogVideoX-2B | CogVideoX-5B | 
|------------|--------------|--------------|
| bs=1, LORA | 47GB         | 63GB         |
| bs=2, LORA | 61GB         | 80GB         |
| bs=1, SFT  | 62GB         | 75GB         |
- This method only finetune the transformer part.


### Preparing the dataset
The dataset should be prepared in the following format:
```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

#### Remarks
- Each video should have a corresponding label file, 
which contains the description of the video. 
- The file name should be the same as the video file name.
- For style finetune, the dataset should be larger than 50 videos and labels
with similar styles.

### Change the configs file
1. Modify the file in ```/path/to/src/sat/configs/sft.yaml```**(For LoRA and SFT)**.
```
  # checkpoint_activations: True ## Using gradient checkpointing (Both checkpoint_activations in the config file need to be set to True)
  model_parallel_size: 1 # Model parallel size
  experiment_name: lora-disney  # Experiment name (do not modify)
  mode: finetune # Mode (do not modify)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer model path
  no_load_rng: True # Whether to load random seed
  train_iters: 1000 # Training iterations
  eval_iters: 1 # Evaluation iterations
  eval_interval: 100    # Evaluation interval
  eval_batch_size: 1  # Evaluation batch size
  save: ckpts # Model save path
  save_interval: 100 # Model save interval
  log_interval: 20 # Log output interval
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # Training and validation datasets can be the same
  split: 1,0,0 # Training, validation, and test set ratio
  num_workers: 8 # Number of worker threads for data loader
  force_train: True # Allow missing keys when loading checkpoint (T5 and VAE are loaded separately)
  only_log_video_latents: True # Avoid memory overhead caused by VAE decode
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B set to False and for CogVideoX-5B set to True
    fp16:
      enabled: True  # For CogVideoX-2B set to True and for CogVideoX-5B set to False
```
**If you want to use Lora finetune, you need to also do the following steps.(Please ignore the following step if you want to use the full-parameter fine-tuning)**
2. Modify the file in ```/path/to/src/sat/configs/cogvideox_<model_parameters>_lora.yaml```**(For LoRA only)**.
```
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## Uncomment
  log_keys:
    - txt'

  lora_config: ## Uncomment
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### Modify the Run Script
Modify the file in ```./src/sat/finetune_single_gpu.sh```
```
run_cmd="torchrun --standalone --nproc_per_node=1 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

### Start Finetune
```
bash finetune_single_gpu.sh
```
The finetuned model will be saved in the path ```./src/sat/ckpts```.

### Evaluation
Modify the file in ```./shscripts/inference_cogVideo_sat.sh```
```
run_cmd="$environs python sample_video.py --base configs/cogvideox_2b_lora.yaml configs/inference.yaml --seed 42"
```
Modify the file in ```./src/sat/configs/inference.yaml```
```
load: "{your finetune model path}" # Finetune model path
# For example: load: "/disk4/juno/cogVideo/sat/ckpts/lora-disney-08-29-01-00"
```
Run the following command to evaluate the finetuned model.
- The ```inference_cogVideo_sat.sh``` is in the ```/path/to/shscripts``` folder. 
```
bash inference_cogVideo_sat.sh
```

## Remarks
When downloading the checkpoints, the file /text_encoder/model-00001-of-00002.safetensors and
/text_encoder/model-00002-of-00002.safetensors may not be successfully downloaded. You need to remove the
files and download them again using the following command:
```
cd CogVideoX-2b
cd text_encoder
rm model-00001-of-00002.safetensors
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/model-00001-of-00002.safetensors
rm model-00002-of-00002.safetensors
wget https://huggingface.co/THUDM/CogVideoX-2b/resolve/main/text_encoder/model-00002-of-00002.safetensors
```




