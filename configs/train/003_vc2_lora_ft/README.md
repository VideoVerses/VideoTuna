# LoRA Finetuning Instructions


## Notes
- The LoRA module is designed to be irrelevant to model details as long as they are subclass of `nn.Module` or `pl.LightningModule`。 

- An example of LoRA finetuning based on VideoCrafter2 is placed in `configs/train/003_vc2_lora_ft`, where exists:

    1. `config.yaml`: model configs with lora-related args.
    2. `run.sh` : training script.
    3. `inference.sh` : inference script after lora training.

## Dependencies
We use `peft` package as the implementation of LoRA to ensure the most compatitableness.  
Install `peft`:

```shell
pip install peft
```
## Prepare Data
You can use your own dataset or download the [example dataset](https://huggingface.co/datasets/Yingqing/VideoTuna/blob/main/VideoTuna-TestData.zip) we provided for testing the finetuning code.
The example dataset can be downloaded via
```
wget https://huggingface.co/datasets/Yingqing/VideoTuna/resolve/main/VideoTuna-TestData.zip -P inputs/trainingdata/
```
The data size is `4.2G` which takes around `5 minitues` to download.
after downloaded, unzip the file via  
```
unzip inputs/trainingdata/VideoTuna-TestData.zip -d inputs/trainingdata
```

## Train
Before training, customize the `config.yaml` if necessary.  
Then run the following command to train:

```shell
bash scripts/train/003_vc2_lora_ft/run.sh
```
## Inference
The standard inference config is at: `configs/inference/vc2_t2v_512_lora.yaml`.   
If train config is modified, the inference config should be modified coresspondly. 

Specify Lora Checkpoint Path `LORACKPT` in `inference.sh`.  
Then run the following command to inference:  
```shell
bash scripts/train/003_vc2_lora_ft/inference.sh
```

## TODO
- [x] inject peft lora and config 
- [x] train and inference test 
- [x] push inject lora back to DDPM and test. 
- [ ] test on different foundation models. 
- [ ] more robust checkpoint IO function. 
- [ ] more flexible LoRA config and more types of LoRA. 