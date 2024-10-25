<p align="center" width="50%">
<img src="assets/logo.jpg" alt="VideoTuna" style="width: 30%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna
Let's finetune video generation models!


## ⏰TODOs
- [x] inference vc, dc   
- [x] finetune & train vc2，dc   
- [x] opensora-train, inference  
- [x] flux inference, fine-tune  
- [x] cogvideo inference, fine-tune  
- [x] merge diffusion parts
- [x] add peft lora 
- [x] add RL for alignment 
- [ ] refactor vc, opensora, cogvideo and flux 
- [x] add documents 
- [ ] add unit test support 
- [ ] svd, open-sora-plan
- [ ] Finish codebase V0.1.0
- [ ] Release demo gallery
- [ ] Release technical report

## 🔆 Updates
- [2024-09-XX] We make the VideoTuna V0.1.0 public!


## Demo
### VAE

<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtview7.gif"><img src="assets/demos/fkview7.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkview7.gif"><img src="assets/demos/fkview7.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Groud Truth: Las Vegas, Fremont Street Walking Tour"</td>
    <td style="text-align:center;" width="320">"Reconstruction: Las Vegas, Fremont Street Walking Tour"</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtview9.gif"><img src="assets/demos/fkview9.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkview9.gif"><img src="assets/demos/fkview9.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Ground Truth: view in a gardon"</td>
    <td style="text-align:center;" width="320">"Reconstruction: view in a gardon."</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtportrait5.gif"><img src="assets/demos/gtportrait5.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkportrait5.gif"><img src="assets/demos/fkportrait5.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Ground Truth: A moman blink in the car."</td>
    <td style="text-align:center;" width="320">"Reconstruction: A moman blink in the car."</td>
  </tr>

  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtmotion4.gif"><img src="assets/demos/gtmotion4.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkmotion4.gif"><img src="assets/demos/fkmotion4.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Ground Truth: Driving on the seaside road."</td>
    <td style="text-align:center;" width="320">"Reconstruction: Driving on the seaside road."</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtanimal2.gif"><img src="assets/demos/gtanimal2.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkanminal2.gif"><img src="assets/demos/fkanminal2.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Ground Truth: A bird on the tree."</td>
    <td style="text-align:center;" width="320">"Reconstruction: A bird on the tree."</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/gtcloseshot1.gif"><img src="assets/demos/gtcloseshot1.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/fkcloseshot1.gif"><img src="assets/demos/fkcloseshot1.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">"Ground Truth: A closeshot of a camera."</td>
    <td style="text-align:center;" width="320">"Reconstruction: A closeshot of a camera."</td>
  </tr>
</table>

<style>
  .video-grid {
    display: flex;
    flex-direction: column;
    gap: 20px; /* Space between each row */
  }

  .video-row {
    display: flex;
    justify-content: space-between;
  }

  .video-container {
    width: 48%; /* Make the videos take equal width with space between */
    text-align: center;
  }

  video {
    width: 100%;
  }
</style>




### Face domain

### Storytelling


## 🔆 Introduction
🤗🤗🤗 VideoTuna is an open-sourcing finetuning framework for text-to-video generation.

### Features
1. All in one framework: Inference and finetune state-of-the-art T2V models.
2. Continuous training
3. Fintuning: domain-specific.
4. Fintuning: enhanced language understanding.
5. Fintuning: enhancement.
6. Human preference alignment/Post-training: RLHF, DPO.

### Code Structure
```
VideoTuna/
    ├── assets
    ├── checkpoints  # put model checkpoints here
    ├── configs      # model and experimental configs
    ├── data         # data processing scripts and dataset files
    ├── docs         # documentations
    ├── eval         # evaluation scripts
    ├── inputs       # input examples for testing 
    ├── scripts      # train and inference python scripts
    ├── shsripts     # train and inference shell scripts
    ├── src          # model-related source code
    ├── tests        # testing scripts
    ├── tools        # some tool scripts
```


### Supported Models

|T2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-2B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b)
|CogVideoX-5B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)
|Open-Sora 1.2|240p to 720p, 2~16s|[STDIT](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3), [VAE](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2)
|Open-Sora 1.1|144p & 240p & 480p, 0~15s|[Stage 2](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2)
|Open-Sora 1.1|144p to 720p, 0~15s|[Stage 3](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3)
|Open-Sora 1.0|512×512x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)
|VideoCrafter2|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-5B-I2V|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
|DynamiCrafter|576x1024x16|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|

* Note: H: height; W: width; L: length


## 🔆 Get started

### 1.Prepare environment
```
conda create --name videotuna python=3.10 -y
conda activate videotuna
pip install -r requirements.txt
git clone https://github.com/JingyeChen/SwissArmyTransformer
pip install -e SwissArmyTransformer/
rm -rf SwissArmyTransformer
git clone https://github.com/tgxs002/HPSv2.git
pip install -e HPSv2/
rm -rf HPSv2
```

### 2.Prepare checkpoints

Please follow [docs/CHECKPOINTS.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md) to download model checkpoints.  
After downloading, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md#checkpoint-orgnization-structure).

### 3.Inference state-of-the-art T2V/I2V models

- Inference a set of models **in one command**:

    <!-- ```bash todo.sh``` -->

|Task|Commands|
|:---------|:---------|
|T2V|`bash tools/video_comparison/compare.sh`|
|I2V|`TODO`|



- Inference one specific model:

Task|Models|Commands|
|:---------|:---------|:---------|
|T2V|cogvideo|`bash shscripts/inference_cogVideo_diffusers.sh`|
|T2V|open-sora|@yazhou|
|T2V|videocrafter-v2-320x512|`bash shscripts/inference_vc2_t2v_320x512.sh`|
|T2V|videocrafter-v1-576x1024|`bash shscripts/inference_vc1_t2v_576x1024.sh`|
|I2V|dynamicrafter|`bash shscripts/inference_dc_i2v_576x1024.sh`|
|I2V|videocrafter1|`bash shscripts/inference_vc1_i2v_320x512.sh`|
|T2I|flux|`bash shscripts/inference_flux_schnell.sh`|

For detailed inference settings please check [docs/inference.md](docs/inference.md).

### 4. Finetune T2V models
#### Lora finetuning

We support lora finetuning to make the model to learn new concepts/characters/styles.   
- Example config file: `configs/train/003_vc2_lora_ft/config.yaml`  
- Training lora based on VideoCrafter2: `bash scripts/train/003_vc2_lora_ft/run.sh`  
- Inference the trained models: `bash scripts/train/003_vc2_lora_ft/inference.sh`   

Please check [configs/train/003_vc2_lora_ft/README.md](configs/train/003_vc2_lora_ft/README.md) for details.   
<!-- 

(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
``` -->

#### Finetuning for enhanced langugage understanding


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance. 
Please check [eval/README.md](eval/README.md) for details.

### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details.



## Acknowledgement
We thank the following repos for sharing their awsome models and codes!
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): Overcoming Data Limitations for High-Quality Video Diffusion Models
* [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Open Diffusion Models for High-Quality Video Generation
* [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All
* [CogVideoX](https://github.com/THUDM/CogVideo): Text-to-Video Diffusion Models with An Expert Transformer
* [VADER](https://github.com/mihirp1998/VADER): Video Diffusion Alignment via Reward Gradients
* [VBench](https://github.com/Vchitect/VBench): Comprehensive Benchmark Suite for Video Generative Models

## 🍻 Contributors

## 📋 License

## 😊 Citation
```
To be updated...
```
