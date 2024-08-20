# CogVideo

## Model

| T2V-Models   | Resolution | Checkpoints                                                         |
|--------------|------------|---------------------------------------------------------------------|
| CogVideoX-2b | 720x480    | [Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b/tree/main) |

## Get started

## Set up environment

### For CUDA 12.x
```
conda create -n cogvideo python=3.10
conda activate cogvideo
```


### Install dependencies
```
pip install -r requirements.txt
pip install --upgrade opencv-python transformers diffusers # Must using diffusers>=0.30.0
```

## Prepare checkpoints

Use the following command to clone the repository and download the checkpoints. 
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-2b
```

### Remarks
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

## Generate video
Generates a video based on the given prompt and saves it to the specified path.

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
python inference_diffusers.py --prompt "A video of a cat playing with a ball" --model_path /path/to/CogVideoX-2b --output_path output.mp4
```
It will generate a video of a cat playing with a ball and save it to the file /cogVideo/output.mp4.