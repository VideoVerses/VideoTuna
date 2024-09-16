# Alignment post-training
We use `hpsv2` as the aesthetic reward model following [VADER](https://github.com/mihirp1998/VADER).

# TODO

- [x] RLHF READEME
- [x] RLHF train and inference code 
- [x] RLHF visual results  on trained prompts
- [x] RLHF vbench results 
- [ ] Visual results comparison
- [ ] More reward models

<!-- # Reward Based Alignment

We follow VADER to use "hpsv2" as aesthetic reward model to finetune.  -->

# Start
## Installation
Please add the following dependencies to support alignment post-training based on the existing videotuna environment. 
```shell
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..

pip install decord==0.6.0
pip install kornia==0.7.3
pip install inflect==7.3.0
```

## Train

```shell
bash configs/train/004_rlhf_vc2/run.sh
```

## Inference

```
bash configs/train/004_rlhf_vc2/inference.sh
```

# Documentation
