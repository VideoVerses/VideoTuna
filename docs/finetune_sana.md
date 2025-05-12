
# Introduction
This document provides instructions for fine-tuning the Sana model.

# Preliminary steps
1. **Install the environment** (see [Installation](https://github.com/VideoVerses/VideoTuna/tree/main?tab=readme-ov-file#1prepare-environment)). 
2. **Download pretrained Sana model**. You can select model from [Official Sana Model Cards](https://github.com/NVlabs/Sana/blob/main/asset/docs/model_zoo.md). We use [Sana_1600M_1024px_BF16_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers) as an example.

    ```shell
    # download the model to your local path
    huggingface-cli download "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers" --local-dir ${your_local_path}
    ```


# Steps of Simple Fine-tuning
We use images in `inputs/t2i/sana/nezha` to train.  
1. Set the exp configs in the file `configs/010_sana/sana.yaml`
      
    **Necessary arguments that you need to modify to train different loras.**

    - `pretrained_model_name_or_path`: Name or path of the pre-trained model.
    - `instance_data_dir`: the image directory. set to `data/images/${DataName}`
    - `instance_prompt`: The prompt with identifier specifying the instance.
    - `validation_prompt`: the testing prompt for validation during training. It should contain the concept name used in training labels.
    - `output_dir`: the directory for saving trained lora models and intermediate results.  

    **Optional arguments that you may need to adjust to match more advanced requirements.**  

    <details>
      <summary>Click to view the introduction to these arguments</summary>

      - `cache_dir`: The directory where the downloaded models and datasets will be stored.
      - `num_validation_images`: Number of images that should be generated during validation with `validation_prompt`.
      - `validation_epochs`: Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.
      - `rank`: The rank of the LoRA models, the bigger, the more learnable parameters.
      - `seed`: Random seed for reproducibility.
      - `resolution`: Image resolution.
      - `train_batch_size`: Batch size (per device) for the training dataloader.
      - `sample_batch_size`: Batch size (per device) for sampling images.
      - `num_train_epochs`: Total number of training epochs (-1 means determined by steps).
      - `max_train_steps`: the total steps for training.
      - `checkpointing_steps`: the steps intersection for saving each LoRA checkpoint. 
      - `checkpoints_total_limit`: the total number of saved model checkpoints.
      - `resume_from_checkpoint`: Resume training from the latest checkpoint.
      - `gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
      - `gradient_checkpointing`: Whether to enable gradient checkpointing.
      - `learning_rate`: controls how much the model weights are adjusted per update, balancing convergence speed and stability.
      - `lr_scheduler`: Type of learning rate scheduler.
      - `lr_warmup_steps`: Number of warmup steps for learning rate.
      - `optimizer`: Type of optimizer.
      - `lora_layers`: The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. `to_k,to_q,to_v` will result in lora training of attention layers only.
      - `mixed_precision`: Type of mixed precision.
    </details>

3. Run the commands in the terminal to launch training.
    ```
    poetry run train-sana-lora
    ```
4. After training, run the commands in the terminal to inference your personalized videotuna models.
    ```
    poetry run inference-sana-lora \
    --prompt "nezha is riding a bike" \
    --lora_path ${lora_path} \
    --out_path ${out_path}
    ```
    - ${out_path} should be a file path like `image.jpg`  

    You can also inference multiple prompts by passing a txt file:
    ```
    poetry run inference-sana-lora \
    --prompt data/prompts/nezha.txt \
    --lora_path ${lora_path} \
    --out_path ${out_path}
    ```
    - ${out_path} should be a directory.

# Use your own dataset
If you want to build your own dataset, please organize your data as `inputs/t2i/sana/nezha`, which contains the training images and the corresponding text prompt files, as shown in the following directory structure.  
Then modify the `instance_data_dir` in `configs/010_sana/sana.yaml`.
```
your_own_data/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
    ├── ...
```
