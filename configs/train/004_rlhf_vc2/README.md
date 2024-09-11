# RLHF-Readme

# TODO

- [x] RLHF READEME
- [x] RLHF train and inference code 
- [x] RLHF visual results  on trained prompts
- [x] RLHF vbench results 
- [ ] Visual results comparison
- [ ] More reward models

# Reward Based Alignment

We follow VADER to use "hpsv2" as aesthetic reward model to finetune. 

# VideoTuna Support 

// environment configuration follow official RADER 

## Train VADER with VideoTuna

```shell
bash configs/configs/train/004_rlhf_vc2/run.sh
```

## Inference VADER with VideoTuna

```
bash configs/configs/train/004_rlhf_vc2/inference.sh
```

## Implementation Illustration

`lvdm/models/rlhf_utils` contains sources from [VADER](https://github.com/mihirp1998/VADER).

customized training code is in `lvdm/models/ddpm3d.RewardLVDMTrainer`。 



