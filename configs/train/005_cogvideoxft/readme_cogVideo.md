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

#### Clone the T5 model
- If you already downloaded the CogVideoX-2b, please use the following command to clone the T5 model.
```
mkdir t5-v1_1-xxl
cp CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

- If you use CogVideoX-5b, please use the following command to clone the T5 model.
```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
rm -r CogVideoX-2b
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
```
mkdir CogVideoX-5b-sat
cd CogVideoX-5b-sat
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1"
mv 'index.html?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1' transformer.tar.gz
tar -xzvf transformer.tar.gz
mv 'CogVideoX-5B-transformer' transformer
rm transformer.tar.gz
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2Flatest&dl=1"
mv 'index.html?p=%2Flatest&dl=1' latest
```

[//]: # (The model structure should be as follows:)

[//]: # (```)

[//]: # (.)

[//]: # (├── vae)

[//]: # (├── t5-v1_1-xxl)

[//]: # (├── CogVideoX-2b)

[//]: # (├── CogVideoX-5b)

[//]: # (├── CogVideoX-2b-sat)

[//]: # (│   └── transformer)

[//]: # (│       └── 1000 &#40;or 1&#41;)

[//]: # (│           └── mp_rank_00_model_states.pt)

[//]: # (│       └── latest)

[//]: # (├── CogVideoX-5b-sat)

[//]: # (│   └── transformer)

[//]: # (│       └── 1000 &#40;or 1&#41;)

[//]: # (│           └── mp_rank_00_model_states.pt)

[//]: # (│       └── latest)

[//]: # ()
[//]: # (```)

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
It will generate a video of a cat playing with a ball and save it to the file ./output.mp4, with the model CogVideoX-2b.

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

### Modify the file in ```./src/sat/inference.sh```.
```
run_cmd="$environs python sample_video.py --base configs/cogvideox_2b.yaml configs/inference.yaml --seed $RANDOM"
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

### Modify the file in ```./src/sat/inference.sh```.
```
run_cmd="$environs python sample_video.py --base configs/cogvideox_5b.yaml configs/inference.yaml --seed $RANDOM"
```



#### Remarks
If you want to use your own prompt, you can modify the file in ```./src/sat/configs/test.txt```.
If multiple prompts is required, in which each line makes a prompt.

### Generate video
Go to file ```./src/sat```, and run the following command:
```
bash shscripts/inference_cogVideo_sat.sh
```



# VideoTuna Support: finetune with pytorch lightning 

this branch use pytorch lightning framework to reorganize cogvideo training and inference. 

## environment
please follow the requirenments at `requrienemnt_cogVideo.txt`
## inference
please refer to the scripts for configuration.
```shell
bash configs/train/005_cogvideoxft/inference.sh
```

or 

```shell
bash shscripts/inference_cogvideo_pl.sh
```
## train
The results has been tested to be correct on a tiny dataset. 
please refer to the scripts for configuration. 

```shell
bash configs/train/005_cogvideoxft/run.sh
```

## implementation details.

train and inference lanucher, all configs are saved to `configs/train/005_cogvideoxft`.


### dataset 

the dataset follow original cogvideo format which has follow foramts:


```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```



> Tips: To build a fake dataset, simply duplicate the videos label files.




