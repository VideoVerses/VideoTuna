# Introduction
This document provides instructions for fine-tuning the HunyuanVideo model.

# Preliminary steps
1. Install the videotuna environment (see [here](https://github.com/VideoVerses/VideoTuna?tab=readme-ov-file#1prepare-environment))
2. Download the checkpoints for HunyuanVideo (see [here](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md))
3. Install deepspeed:
```shell
poetry run install-deepspeed
```

# Steps for Fine-tuning
### Lora Fine-tuning of HunyuanVideo Text-to-Video

1. Run the commands in the terminal to launch training.
    ```
    bash shscripts/train_hunyuanvideo_t2v_lora.sh
    ```
    NOTE: this script uses deepspeed for training.

2. After training, one additional checkpoints converting step is needed. The script is:
    ```shell
    tools/deepspeed_checkpoint_converter.py
    ```

3. Inference:
    ```
    bash shscripts/inference_hunyuanvideo_t2v_lora.sh
    ```
    - You need to provide the checkpoint path to the `ckpt` argument in the above shell script.  

    Note: 
    - The training and inference use the default model config from `configs/007_hunyuanvideo/hunyuanvideo_diffuser.yaml`




