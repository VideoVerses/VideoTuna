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
conda create -n CogVideo python=3.10
conda activate cogvideo
```


### Install dependencies
```
cd cogVideo
pip install -r requirements_cogVideo.txt
pip install --upgrade opencv-python transformers diffusers # Must using diffusers>=0.30.0
```

### Prepare checkpoints

#### CogVideoX-2b
Use the following command to clone the repository and download the checkpoints. 
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-2b
```

#### Remarks
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

#### CogVideoX-5b
Use the following command to clone the repository and download the checkpoints.
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b
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
bash /path/to/shscripts/inference_cogVideo_diffusers.sh
```
It will generate a video of a cat playing with a ball and save it to the file /cogVideo/output.mp4, with the model CogVideoX-2b.

# SAT Inference
* Single GPU Inference for 2b version (FP16), the model need 18GB GPU memory. 
* Single GPU Inference for 5b version (BF16), the model need 26GB GPU memory.

## Get started

If you already use the inference by Diffusers, you can skip the following step
- Set up environment 
- Install the dependencies for CogVideo.

### Set up environment
```
conda create -n CogVideo python=3.10
conda activate CogVideo
```

### Install dependencies
1. Install the dependencies for the CogVideo.
```
cd cogVideo
pip install -r requirements_cogVideo.txt
```
2. Install the dependencies for the SAT model.
```
cd src/sat  # Go to the /path/to/src/sat
pip install -r requirements_cogVideo_sat.txt
```

### Prepare checkpoints
#### CogVideoX-2b-sat

```
cd cogVideo # Go to the /path/to/cogVideo
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```
### Clone the T5 model
```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

### Modify the file in ```/path/to/src/sat/configs/cogvideox_2b.yaml```.
```
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  log_keys:
    - txt

  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 30
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 1920
      adm_in_channels: 256
      num_attention_heads: 30

      transformer_args:
        checkpoint_activations: True ## using gradient checkpointing
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Basic3DPositionEmbeddingMixin
          params:
            text_length: 226
            height_interpolation: 1.875
            width_interpolation: 1.875

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "{absolute_path/to/your/t5-v1_1-xxl}/t5-v1_1-xxl" # Absolute path to the t5-v1_1-xxl weights folder
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "{absolute_path/to/your/t5-v1_1-xxl}/CogVideoX-2b-sat/vae/3d-vae.pt" # Absolute path to the CogVideoX-2b-sat/vae/3d-vae.pt folder
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 3.0

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### Modify the file in ```/path/to/src/sat/configs/inference.yaml```.
```
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # Absolute path to the CogVideoX-2b-sat/transformer folder
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt # You can choose txt for pure text input, or change to cli for command line input
  input_file: configs/test.txt # Pure text file, which can be edited. If use command line as prompt iuput, please change it to input_type: cli
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
#  bf16: True # For CogVideoX-5B
  output_dir: outputs/ # Change the output_dir if you want to save the results to your own directory
  force_inference: True
```

#### CogVideoX-5b-sat
Use the following command to clone the repository and download the checkpoints.
Or access the [Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b) to download the checkpoints.
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b
```

And you also need to download the model from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
Or run the following command:
```
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1"
mv 'index.html?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1' transformer.tar.gz
tar -zxvf transformer.tar.gz
```

The model structure should be as follows:
```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```
Next clone the T5 model
```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

Modify the file in ```/path/to/src/sat/configs/cogvideox_5b.yaml```.
```
model:
  scale_factor: 0.7 # different from cogvideox_2b_infer.yaml
  disable_first_stage_autocast: true
  log_keys:
    - txt
  
  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 1.0 # different from cogvideox_2b_infer.yaml

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 42 # different from cogvideox_2b_infer.yaml
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 3072 # different from cogvideox_2b_infer.yaml
      adm_in_channels: 256
      num_attention_heads: 48 # different from cogvideox_2b_infer.yaml

      transformer_args:
        checkpoint_activations: True
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Rotary3DPositionEmbeddingMixin # different from cogvideox_2b_infer.yaml
          params:
            hidden_size_head: 64
            text_length: 226

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "{absolute_path/to/your/t5-v1_1-xxl}/t5-v1_1-xxl" # Absolute path to the t5-v1_1-xxl weights folder
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "{absolute_path/to/your/t5-v1_1-xxl}/CogVideoX-5b-sat/vae/3d-vae.pt"
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 1.0 # different from cogvideox_2b_infer.yaml

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 1.0 # different from cogvideox_2b_infer.yaml

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### Modify the file in ```/path/to/src/sat/configs/inference.yaml```.
```
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # Absolute path to the CogVideoX-2b-sat/transformer folder
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt # You can choose txt for pure text input, or change to cli for command line input
  input_file: configs/test.txt # Pure text file, which can be edited
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
#  fp16: True # For CogVideoX-2B
  bf16: True # For CogVideoX-5B
  output_dir: outputs/ 
  force_inference: True
```

#### Remarks
If you want to use your own prompt, you can modify the file in ```/path/to/src/sat/configs/test.txt```.
If multiple prompts is required, in which each line makes a prompt.

### Generate video
Go to file ```/path/to/src/sat```, and run the following command:
```
bash shscripts/inference_cogVideo_sat.sh
```


# SAT Finetune(Lora)

[//]: # (- For Lora finetune, bs=1, it needs 47GB GPU memory.)

[//]: # (- For Lora finetune, bs=2, it needs 61GB GPU memory.)
The memory consumption per GPU:

|            | CogVideoX-2B | CogVideoX-5B | 
|------------|--------------|--------------|
| bs=1, LORA | 47GB         | 63GB         |
| bs=2, LORA | 61GB         | 80GB         |
| bs=1, SFT  | 62GB         | 75GB         |
- This method only finetune the transformer part.

## Get started

### Set up environment
```
conda create -n CogVideo python=3.10
conda activate CogVideo
```

### Install dependencies
1. Install the dependencies for the CogVideo.
```
cd cogVideo
pip install -r requirements_cogVideo.txt
```
2. Install the dependencies for the SAT model.
```
cd src/sat  # Go to the /path/to/src/sat
pip install -r requirements_cogVideo_sat.txt
```

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
1. Modify the file in ```/path/to/src/sat/configs/sft.yaml```.
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
If you want to use Lora finetune, you need to also do the following steps.(Please ignore the following step if you want to use the full-parameter fine-tuning)
2. Modify the file in ```/path/to/src/sat/configs/cogvideox_<model_parameters>_lora.yaml```
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
Modify the file in ```/path/to/src/sat/finetune_single_gpu.sh```
```
run_cmd="torchrun --standalone --nproc_per_node=1 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

### Start Finetune
```
bash finetune_single_gpu.sh
```
The finetuned model will be saved in the path ```/path/to/src/sat/ckpts```.

### Evaluation
Modify the file in ```/path/to/shscripts/inference_cogVideo_sat.sh```
```
run_cmd="$environs python sample_video.py --base configs/cogvideox_2b_lora.yaml configs/inference.yaml --seed 42"
```
Modify the file in ```/path/to/src/sat/configs/inference.yaml```
```
load: "{your finetune model path}" # Finetune model path
# For example: load: "/disk4/juno/cogVideo/sat/ckpts/lora-disney-08-29-01-00"
```
Run the following command to evaluate the finetuned model.
- The ```inference_cogVideo_sat.sh``` is in the ```/path/to/shscripts``` folder. 
```
bash inference_cogVideo_sat.sh
```



