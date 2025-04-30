<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/38efb5bc-723e-4012-aebd-f55723c593fb" alt="VideoTuna" style="width: 75%; min-width: 450px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna

![Version](https://img.shields.io/badge/version-0.1.0-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=VideoVerses.VideoTuna&left_color=green&right_color=red)  [![](https://dcbadge.limes.pink/api/server/AammaaR2?style=flat)](https://discord.gg/AammaaR2) <a href='https://github.com/user-attachments/assets/a48d57a3-4d89-482c-8181-e0bce4f750fd'><img src='https://badges.aleen42.com/src/wechat.svg'></a> [![Homepage](https://img.shields.io/badge/Homepage-VideoTuna-orange)](https://videoverses.github.io/videotuna/) [![GitHub](https://img.shields.io/github/stars/VideoVerses/VideoTuna?style=social)](https://github.com/VideoVerses/VideoTuna)


🤗🤗🤗 Videotuna is a useful codebase for text-to-video applications.  
🌟 VideoTuna is the first repo that integrates multiple AI video generation models including `text-to-video (T2V)`, `image-to-video (I2V)`, `text-to-image (T2I)`, and `video-to-video (V2V)` generation for model inference and finetuning (to the best of our knowledge).  
🌟 VideoTuna is the first repo that provides comprehensive pipelines in video generation, from fine-tuning to pre-training, continuous training, and post-training (alignment) (to the best of our knowledge).  



## 🔆 Features
![videotuna-pipeline-fig3](https://github.com/user-attachments/assets/625693d9-b5cf-4c00-8e84-20ea855c2445)
🌟 **All-in-one framework:** Inference and fine-tune various up-to-date pre-trained video generation models.  
🌟 **Continuous training:** Keep improving your model with new data.  
🌟 **Fine-tuning:** Adapt pre-trained models to specific domains.  
🌟 **Human preference alignment:** Leverage RLHF to align with human preferences.  
🌟 **Post-processing:** Enhance and rectify the videos with video-to-video enhancement model.  


## 🔆 Updates

- [2025-04-22] 🐟 Supported **inference** for `Wan2.1` and `Step Video` and **fine-tuning** for `HunyuanVideo T2V`, with a unified codebase architecture.
- [2025-02-03] 🐟 Supported automatic code formatting via [PR#27](https://github.com/VideoVerses/VideoTuna/pull/27). Thanks [@samidarko](https://github.com/samidarko)!
- [2025-02-01] 🐟 Migrated to [Poetry](https://python-poetry.org) for streamlined dependency and script management ([PR#25](https://github.com/VideoVerses/VideoTuna/pull/25)). Thanks [@samidarko](https://github.com/samidarko)!
- [2025-01-20] 🐟 Supported **fine-tuning** for `Flux-T2I`.
- [2025-01-01] 🐟 Released **training** for `VideoVAE+` in the [VideoVAEPlus repo](https://github.com/VideoVerses/VideoVAEPlus).
- [2025-01-01] 🐟 Supported **inference** for `Hunyuan Video` and `Mochi`.
- [2024-12-24] 🐟 Released `VideoVAE+`: a SOTA Video VAE model—now available in [this repo](https://github.com/VideoVerses/VideoVAEPlus)! Achieves better video reconstruction than NVIDIA’s [`Cosmos-Tokenizer`](https://github.com/NVIDIA/Cosmos-Tokenizer).
- [2024-12-01] 🐟 Supported **inference** for `CogVideoX-1.5-T2V&I2V` and `Video-to-Video Enhancement` from ModelScope.
- [2024-12-01] 🐟 Supported **fine-tuning** for `CogVideoX`.
- [2024-11-01] 🐟 🎉 Released **VideoTuna v0.1.0**!  
  Initial support includes inference for `VideoCrafter1-T2V&I2V`, `VideoCrafter2-T2V`, `DynamiCrafter-I2V`, `OpenSora-T2V`, `CogVideoX-1-2B-T2V`, `CogVideoX-1-T2V`, `Flux-T2I`, and training/fine-tuning of `VideoCrafter`, `DynamiCrafter`, and `Open-Sora`.

## 🔆 Get started

### 1.Prepare environment

#### (1) If you use Linux and Conda (Recommend)
``` shell
conda create -n videotuna python=3.10 -y
conda activate videotuna
pip install poetry
poetry install
```
- ↑ It takes around 3 minitues.

**Optional: Flash-attn installation**

Hunyuan model uses it to reduce memory usage and speed up inference. If it is not installed, the model will run in normal mode. Install the `flash-attn` via:
``` shell
poetry run install-flash-attn 
```
- ↑ It takes 1 minitue.

**Optional: Video-to-video enhancement**
```
poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
- If this command ↑ get stucked, kill and re-run it will solve the issue.


#### (2) If you use Linux and Poetry (without Conda):
<details>
  <summary>Click to check instructions</summary>
  <br>

  Install Poetry: https://python-poetry.org/docs/#installation  
  Then:

  ``` shell
  poetry config virtualenvs.in-project true # optional but recommended, will ensure the virtual env is created in the project root
  poetry config virtualenvs.create true # enable this argument to ensure the virtual env is created in the project root
  poetry env use python3.10 # will create the virtual env, check with `ls -l .venv`.
  poetry env activate # optional because Poetry commands (e.g. `poetry install` or `poetry run <command>`) will always automatically load the virtual env.
  poetry install
  ```

  **Optional: Flash-attn installation**

  Hunyuan model uses it to reduce memory usage and speed up inference. If it is not installed, the model will run in normal mode. Install the `flash-attn` via:
  ``` shell
  poetry run install-flash-attn
  ```
  
  **Optional: Video-to-video enhancement**
  ```
  poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
  ```
  - If this command ↑ get stucked, kill and re-run it will solve the issue.

</details>



#### (3) If you use MacOS
<details>
  <summary>Click to check instructions</summary>
  <br>

  On MacOS with Apple Silicon chip use [docker compose](https://docs.docker.com/compose/) because some dependencies are not supporting arm64 (e.g. `bitsandbytes`, `decord`, `xformers`).

  First build:

  ```shell
  docker compose build videotuna
  ```

  To preserve the project's files permissions set those env variables:

  ```shell
  export HOST_UID=$(id -u)
  export HOST_GID=$(id -g)
  ```

  Install dependencies:

  ```shell
  docker compose run --remove-orphans videotuna poetry env use /usr/local/bin/python
  docker compose run --remove-orphans videotuna poetry run python -m pip install --upgrade pip setuptools wheel
  docker compose run --remove-orphans videotuna poetry install
  docker compose run --remove-orphans videotuna poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
  ```

  Note: installing swissarmytransformer might hang. Just try again and it should work.

  Add a dependency:

  ```shell
  docker compose run --remove-orphans videotuna poetry add wheel
  ```

  Check dependencies:

  ```shell
  docker compose run --remove-orphans videotuna poetry run pip freeze
  ```

  Run Poetry commands:

  ```shell
  docker compose run --remove-orphans videotuna poetry run format
  ```

  Start a terminal:

  ```shell
  docker compose run -it --remove-orphans videotuna bash
  ```
</details>

### 2.Prepare checkpoints

- Please follow [docs/checkpoints.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/checkpoints.md) to download model checkpoints.  
- After downloading, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/blob/main/docs/checkpoints.md#checkpoint-orgnization-structure).

### 3.Inference state-of-the-art T2V/I2V/T2I models


Run the following commands to inference models:
It will automatically perform T2V/T2I based on prompts in `inputs/t2v/prompts.txt`, 
and I2V based on images and prompts in `inputs/i2v/576x1024`.  

**T2V**
Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|T2V|HunyuanVideo|`poetry run inference-hunyuan-t2v`|129|720x1280|32min|60G|
|T2V|WanVideo|`poetry run inference-wanvideo-t2v-720p`|81|720x1280|32min|70G|
|T2V|StepVideo|`poetry run inference-stepvideo-t2v-544x992`|51|544x992|8min|61G|
|T2V|Mochi|`poetry run inference-mochi`|84|480x848|2min|26G|
|T2V|CogVideoX-5b|`poetry run inference-cogvideo-t2v-diffusers`|49|480x720|2min|3G|
|T2V|CogVideoX-2b|`poetry run inference-cogvideo-t2v-diffusers`|49|480x720|2min|3G|
|T2V|Open Sora V1.0|`poetry run inference-opensora-v10-16x256x256`|16|256x256|11s|24G|
|T2V|VideoCrafter-V2-320x512|`poetry run inference-vc2-t2v-320x512`|16|320x512|26s|11G|
|T2V|VideoCrafter-V1-576x1024|`poetry run inference-vc1-t2v-576x1024`|16|576x1024|2min|15G|

---


**I2V**


Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|I2V|WanVideo|`poetry run inference-wanvideo-i2v-720p `|81|720x1280|28min|77G|
|I2V|HunyuanVideo|`poetry run inference-hunyuan-i2v-720p`|129|720x1280|29min|43G|
|I2V|CogVideoX-5b-I2V|`poetry run inference-cogvideox-15-5b-i2v`|49|480x720|5min|5G|
|I2V|DynamiCrafter|`poetry run inference-dc-i2v-576x1024`|16|576x1024|2min|53G|
|I2V|VideoCrafter-V1|`poetry run inference-vc1-i2v-320x512`|16|320x512|26s|11G|


---

**T2I**

Task|Model|Command|Length (#Frames)|Resolution|Inference Time|GPU Memory (GB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|T2I|Flux-dev|`poetry run inference-flux-dev`|1|768x1360|4s|37G|
|T2I|Flux-dev|`poetry run inference-flux-dev --enable_vae_tiling --enable_sequential_cpu_offload`|1|768x1360|4.2min|2G|
|T2I|Flux-schnell|`poetry run inference-flux-schnell`|1|768x1360|1s|37G|
|T2I|Flux-schnell|`poetry run inference-flux-schnell --enable_vae_tiling --enable_sequential_cpu_offload`|1|768x1360|24s|2G|

### 4. Finetune T2V models
#### (1) Prepare dataset
Please follow the [docs/datasets.md](docs/datasets.md) to try provided toydataset or build your own datasets.

#### (2) Fine-tune
All  training commands were tested on H800 80G GPUs.  
**T2V**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|T2V|Wan Video|Lora Fine-tune|`poetry run train-wan2-1-t2v-lora`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|T2V|Wan Video|Full Fine-tune|`poetry run train-wan2-1-t2v-fullft`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|T2V|Hunyuan Video|Lora Fine-tune|`poetry run train-hunyuan-t2v-lora`|[docs/finetune_hunyuanvideo.md](docs/finetune_hunyuanvideo.md)|2|
|T2V|CogvideoX|Lora Fine-tune|`poetry run train-cogvideox-t2v-lora`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|1|
|T2V|CogvideoX|Full Fine-tune|`poetry run train-cogvideox-t2v-fullft`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|4|
|T2V|Open-Sora v1.0|Full Fine-tune|`poetry run train-opensorav10`|-|1|
|T2V|VideoCrafter|Lora Fine-tune|`poetry run train-videocrafter-lora`|[docs/finetune_videocrafter.md](docs/finetune_videocrafter.md)|1|
|T2V|VideoCrafter|Full Fine-tune|`poetry run train-videocrafter-v2`|[docs/finetune_videocrafter.md](docs/finetune_videocrafter.md)|1|

---

**I2V**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|I2V|Wan Video|Lora Fine-tune|`poetry run train-wan2-1-i2v-lora`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|I2V|Wan Video|Full Fine-tune|`poetry run train-wan2-1-i2v-fullft`|[docs/finetune_wan.md](docs/finetune_wan.md)|1|
|I2V|CogvideoX|Lora Fine-tune|`poetry run train-cogvideox-i2v-lora`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|1|
|I2V|CogvideoX|Full Fine-tune|`poetry run train-cogvideox-i2v-fullft`|[docs/finetune_cogvideox.md](docs/finetune_cogvideox.md)|4|

---

**T2I**

|Task|Model|Mode|Command|More Details|#GPUs|
|:----|:---------|:---------------|:-----------------------------------------|:----------------------------|:------|
|T2I|Flux|Lora Fine-tune|`poetry run train-flux-lora`|[docs/finetune_flux.md](docs/finetune_flux.md)|1|


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance.
Please check [eval/README.md](docs/evaluation.md) for details.

<!-- ### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details. -->

## Contribute

## Git hooks

Git hooks are handled with [pre-commit](https://pre-commit.com) library.

### Hooks installation

Run the following command to install hooks on `commit`. They will check formatting, linting and types.

```shell
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

### Running the hooks without commiting

```shell
poetry run pre-commit run --all-files
```

## Acknowledgement
We thank the following repos for sharing their awesome models and codes!

* [Wan2.1](https://github.com/Wan-Video/Wan2.1): Wan: Open and Advanced Large-Scale Video Generative Models.
* [HunyuanVideo](https://github.com/Tencent/HunyuanVideo): A Systematic Framework For Large Video Generation Model.
* [Step-Video](https://github.com/stepfun-ai/Step-Video-T2V): A text-to-video pre-trained model with 30 billion parameters and the capability to generate videos up to 204 frames.
* [Mochi](https://www.genmo.ai/blog): A new SOTA in open-source video generation models
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): Overcoming Data Limitations for High-Quality Video Diffusion Models
* [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Open Diffusion Models for High-Quality Video Generation
* [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All
* [CogVideoX](https://github.com/THUDM/CogVideo): Text-to-Video Diffusion Models with An Expert Transformer
* [VADER](https://github.com/mihirp1998/VADER): Video Diffusion Alignment via Reward Gradients
* [VBench](https://github.com/Vchitect/VBench): Comprehensive Benchmark Suite for Video Generative Models
* [Flux](https://github.com/black-forest-labs/flux): Text-to-image models from Black Forest Labs.
* [SimpleTuner](https://github.com/bghira/SimpleTuner): A fine-tuning kit for text-to-image generation.




## Some Resources
* [LLMs-Meet-MM-Generation](https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation): A paper collection of utilizing LLMs for multimodal generation (image, video, 3D and audio).
* [MMTrail](https://github.com/litwellchi/MMTrail): A multimodal trailer video dataset with language and music descriptions.
* [Seeing-and-Hearing](https://github.com/yzxing87/Seeing-and-Hearing): A versatile framework for Joint VA generation, V2A, A2V, and I2A.
* [Self-Cascade](https://github.com/GuoLanqing/Self-Cascade): A Self-Cascade model for higher-resolution image and video generation.
* [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter) and [HiPrompt](https://liuxinyv.github.io/HiPrompt/): Free method for higher-resolution image and video generation.
* [FreeTraj](https://github.com/arthur-qiu/FreeTraj) and [FreeNoise](https://github.com/AILab-CVC/FreeNoise): Free method for video trajectory control and longer-video generation.
* [Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji), [Follow-Your-Click](https://github.com/mayuelala/FollowYourClick), and [Follow-Your-Pose](https://follow-your-pose.github.io/): Follow family for controllable video generation.
* [Animate-A-Story](https://github.com/AILab-CVC/Animate-A-Story): A framework for storytelling video generation.
* [LVDM](https://github.com/YingqingHe/LVDM): Latent Video Diffusion Model for long video generation and text-to-video generation.



## 🍻 Contributors

<a href="https://github.com/VideoVerses/VideoTuna/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=VideoVerses/VideoTuna" />
</a>

## 📋 License
Please follow [CC-BY-NC-ND](./LICENSE). If you want a license authorization, please contact the project leads Yingqing He (yhebm@connect.ust.hk) and Yazhou Xing (yxingag@connect.ust.hk).

## 😊 Citation

```bibtex
@software{videotuna,
  author = {Yingqing He and Yazhou Xing and Zhefan Rao and Haoyu Wu and Zhaoyang Liu and Jingye Chen and Pengjun Fang and Jiajun Li and Liya Ji and Runtao Liu and Xiaowei Chi and Yang Fei and Guocheng Shao and Yue Ma and Qifeng Chen},
  title = {VideoTuna: A Powerful Toolkit for Video Generation with Model Fine-Tuning and Post-Training},
  month = {Nov},
  year = {2024},
  url = {https://github.com/VideoVerses/VideoTuna}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VideoVerses/VideoTuna&type=Date)](https://star-history.com/#VideoVerses/VideoTuna&Date)
