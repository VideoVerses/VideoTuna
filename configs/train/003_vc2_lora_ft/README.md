# LoRA Implementation Details

To ensure the most compatitableness, we directly use `peft` as the implementation of LoRA. 

```shell
pip install peft
```


The LoRA module is designed to be irrelevant to model details as long as they are subclass of `nn.Module` or `pl.LightningModule`。 

## To Use LoRA in VideoTune

An example of LoRA is placed in `configs/train/003_vc2_lora_ft`, where exists:

1. `config.yaml`: add lora args from the full finetune config. 
2. `run.sh` : train lora models with `lora.yaml`
3. `inference.sh` : test lora result trained by `run.sh`. 

### Train
custom `config.yaml` if necessary.

```shell
bash scripts/train/003_vc2_lora_ft/run.sh
```
### Inference
The standard inference config is at: `configs/inference/vc2_t2v_512_lora.yaml`. if train config is modified, the inference config should be modified coresspondly. 

Specify Lora Checkpoint Path in `inference.sh`. Refere to the script for details. 

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