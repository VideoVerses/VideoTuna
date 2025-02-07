<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/38efb5bc-723e-4012-aebd-f55723c593fb" alt="VideoTuna" style="width: 75%; min-width: 450px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna

![Version](https://img.shields.io/badge/version-0.1.0-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=VideoVerses.VideoTuna&left_color=green&right_color=red)  [![](https://dcbadge.limes.pink/api/server/AammaaR2?style=flat)](https://discord.gg/AammaaR2) <a href='https://github.com/user-attachments/assets/a48d57a3-4d89-482c-8181-e0bce4f750fd'><img src='https://badges.aleen42.com/src/wechat.svg'></a> [![Homepage](https://img.shields.io/badge/Homepage-VideoTuna-orange)](https://videoverses.github.io/videotuna/) [![GitHub](https://img.shields.io/github/stars/VideoVerses/VideoTuna?style=social)](https://github.com/VideoVerses/VideoTuna)


🤗🤗🤗 Videotuna is a useful codebase for text-to-video applications.   
🌟 VideoTuna is the first repo that integrates multiple AI video generation models including `text-to-video (T2V)`, `image-to-video (I2V)`, `text-to-image (T2I)`, and `video-to-video (V2V)` generation for model inference and finetuning (to the best of our knowledge).   
🌟 VideoTuna is the first repo that provides comprehensive pipelines in video generation, from fine-tuning to pre-training, continuous training, and post-training (alignment) (to the best of our knowledge).   
🌟 An Emotion Control I2V model will be released soon.  


## Features
🌟 **All-in-one framework:** Inference and fine-tune up-to-date video generation models.  
🌟 **Pre-training:** Build your own foundational text-to-video model.  
🌟 **Continuous training:** Keep improving your model with new data.  
🌟 **Domain-specific fine-tuning:** Adapt models to your specific scenario.  
🌟 **Concept-specific fine-tuning:** Teach your models with unique concepts.  
🌟 **Enhanced language understanding:** Improve model comprehension through continuous training.  
🌟 **Post-processing:** Enhance the videos with video-to-video enhancement model.  
🌟 **Post-training/Human preference alignment:** Post-training with RLHF for more attractive results.  


## 🔆 Updates
- [2025-02-03] 🐟 We update automatic code formatting from [PR#27](https://github.com/VideoVerses/VideoTuna/pull/27). Thanks [samidarko](https://github.com/samidarko)!
- [2025-02-01] 🐟 We update [Poetry](https://python-poetry.org) migration for better dependency management and script automation from [PR#25](https://github.com/VideoVerses/VideoTuna/pull/25). Thanks [samidarko](https://github.com/samidarko)!
- [2025-01-20] 🐟 We update the **fine-tuning** of `Flux-T2I`. Thanks VideoTuna team!
- [2025-01-01] 🐟 We update the **training** of `VideoVAE+` in [this repo](https://github.com/VideoVerses/VideoVAEPlus). Thanks VideoTuna team!
- [2025-01-01] 🐟 We update the **inference** of `Hunyuan Video` and `Mochi`. Thanks VideoTuna team!
- [2024-12-24] 🐟 We release a SOTA Video VAE model `VideoVAE+` in [this repo](https://github.com/VideoVerses/VideoVAEPlus)! Better video reconstruction than Nvidia's [`Cosmos-Tokenizer`](https://github.com/NVIDIA/Cosmos-Tokenizer). Thanks VideoTuna team!
- [2024-12-01] 🐟 We update the **inference** of `CogVideoX-1.5-T2V&I2V`, `Video-to-Video Enhancement` from ModelScope, and **fine-tuning** of `CogVideoX-1`. Thanks VideoTuna team!
- [2024-11-01] 🐟 We make the VideoTuna V0.1.0 public! It supports inference of `VideoCrafter1-T2V&I2V`, `VideoCrafter2-T2V`, `DynamiCrafter-I2V`, `OpenSora-T2V`, `CogVideoX-1-2B-T2V`, `CogVideoX-1-T2V`, `Flux-T2I`, as well as training and finetuning of part of these models. Thanks VideoTuna team!



## Application Demonstration
### Model Inference and Comparison

![combined_video_29_A_mountain_biker_racing_down_a_trail__dust_flying_behind](https://github.com/user-attachments/assets/f8249049-e0d8-47b9-a5b3-511994779cb1)
![combined_video_22_Fireworks_exploding_over_a_historic_river__reflections_twinkling_in_the_water](https://github.com/user-attachments/assets/868c02fc-1e44-4636-b4e7-d9f2287bc89f)
<!-- ![combined_video_20_Waves_crashing_against_a_rocky_shore_under_a_stormy_sky__spray_misting_the_air](https://github.com/user-attachments/assets/ab04d3c6-2d5d-40e5-be64-5d8373f12402)
![combined_video_17_A_butterfly_landing_delicately_on_a_wildflower_in_a_vibrant_meadow](https://github.com/user-attachments/assets/247212e5-0d5a-4f93-b47f-ee9c8ba945fb)
![combined_video_12_Sunlight_piercing_through_a_dense_canopy_in_a_tropical_rainforest__illuminating_a_](https://github.com/user-attachments/assets/f66551ca-7d18-4c73-9656-3d2757ea4fb5)
![combined_video_3_Divers_observing_a_group_of_tuna_as_they_navigate_through_a_vibrant_coral_reef_teem](https://github.com/user-attachments/assets/6c084832-5a0d-42ac-b7b8-1d914b8a35dc) -->





### Video VAE+
Video VAE+ can accurately compress and reconstruct the input videos with fine details.

<table class="center">
  
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/0efcbf80-0074-4421-810f-79a1f1733ed3"><img src="https://github.com/user-attachments/assets/0efcbf80-0074-4421-810f-79a1f1733ed3" width="320"></a></td>
    <td><a href="https://github.com/user-attachments/assets/4adf29f2-d413-49b1-bccc-48adfd64a4da"><img src="https://github.com/user-attachments/assets/4adf29f2-d413-49b1-bccc-48adfd64a4da" width="320"></a></td>
  </tr>  
  
</table>

### Emotion Control I2V

<table class="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a1562c70-d97c-4324-bb11-47db2b83f443" width="240" alt="Image 1"></td>
    <td><img src="https://github.com/user-attachments/assets/4e873f4c-ca56-4549-aaa1-ef24032ae96b" width="240" alt="Image 3"></td>
  </tr>
  <tr>
    <td style="text-align: center;">Input 1</td>
    <td style="text-align: center;">Input 2</td>
  </tr>
</table>

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/972dde7a-fa88-479a-a47e-71d3650b1826"><img src="https://github.com/user-attachments/assets/972dde7a-fa88-479a-a47e-71d3650b1826" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/3c216090-9ad1-4911-b990-179b45314d3e"><img src="https://github.com/user-attachments/assets/3c216090-9ad1-4911-b990-179b45314d3e" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/2e2fb78e-2f39-47bd-acaf-3cfbce83b162"><img src="https://github.com/user-attachments/assets/2e2fb78e-2f39-47bd-acaf-3cfbce83b162" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/f2f55021-4e0d-43a7-9f57-3c94b772f573"><img src="https://github.com/user-attachments/assets/f2f55021-4e0d-43a7-9f57-3c94b772f573" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/600a2f6c-7a8f-4304-bdc3-5f0d65d4fb83"><img src="https://github.com/user-attachments/assets/600a2f6c-7a8f-4304-bdc3-5f0d65d4fb83" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/8ad7c7d8-6492-4435-9436-168f90429be3"><img src="https://github.com/user-attachments/assets/8ad7c7d8-6492-4435-9436-168f90429be3" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/f55a2b6c-ce10-4716-a001-f747c0da17a4"><img src="https://github.com/user-attachments/assets/f55a2b6c-ce10-4716-a001-f747c0da17a4" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/bb3620af-322d-43bc-9060-c7ce9fc32672"><img src="https://github.com/user-attachments/assets/bb3620af-322d-43bc-9060-c7ce9fc32672" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/46a39738-89c2-43fe-9f98-f0ac0d26e39b"><img src="https://github.com/user-attachments/assets/46a39738-89c2-43fe-9f98-f0ac0d26e39b" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/2d3d6e0d-2034-4341-8ca3-42e0cda2704f"><img src="https://github.com/user-attachments/assets/2d3d6e0d-2034-4341-8ca3-42e0cda2704f" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/331f23e1-f441-46f7-98a6-b25a684780f3"><img src="https://github.com/user-attachments/assets/331f23e1-f441-46f7-98a6-b25a684780f3" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/542f8535-634a-4f82-ae2a-39f988c6bc55"><img src="https://github.com/user-attachments/assets/542f8535-634a-4f82-ae2a-39f988c6bc55" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


### Character-Consistent Storytelling Video Generation

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/27aee539-f2bf-467a-8da5-22f506713aa0"><img src="https://github.com/user-attachments/assets/27aee539-f2bf-467a-8da5-22f506713aa0" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/fef5b694-6e1f-42f6-a5a1-7f0b856f3678"><img src="https://github.com/user-attachments/assets/fef5b694-6e1f-42f6-a5a1-7f0b856f3678" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/91408eb8-264d-4d3f-9098-0bfe06022467"><img src="https://github.com/user-attachments/assets/91408eb8-264d-4d3f-9098-0bfe06022467" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/0b822858-da6f-4e6f-82dd-5f243e2feccc"><img src="https://github.com/user-attachments/assets/0b822858-da6f-4e6f-82dd-5f243e2feccc" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/75f69de2-9e5c-48d7-ae55-b28084772836"><img src="https://github.com/user-attachments/assets/75f69de2-9e5c-48d7-ae55-b28084772836" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The picture shows a cozy room with a little girl telling her travel story to her teddybear beside the bed.</td>
    <td style="text-align:center;">As night falls, teddybear sits by the window, his eyes sparkling with longing for the distant place</td>
    <td style="text-align:center;">Teddybear was in a corner of the room, making a small backpack out of old cloth strips, with a map, a compass and dry food next to it.</td>
    <td style="text-align:center;">The first rays of sunlight in the morning came through the window, and teddybear quietly opened the door and embarked on his adventure.</td>
    <td style="text-align:center;">In the forest, the sun shines through the treetops, and teddybear moves among various animals and communicates with them.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/3ae06dbf-f41f-4e7f-b384-fca3abf2c0aa"><img src="https://github.com/user-attachments/assets/3ae06dbf-f41f-4e7f-b384-fca3abf2c0aa" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/09a2fdb9-3e84-40a5-a729-075876b10412"><img src="https://github.com/user-attachments/assets/09a2fdb9-3e84-40a5-a729-075876b10412" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/a382aa30-895d-4476-8f86-b668fe153c16"><img src="https://github.com/user-attachments/assets/a382aa30-895d-4476-8f86-b668fe153c16" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/c906b20e-1576-4a96-9c77-b67a940dcce8"><img src="https://github.com/user-attachments/assets/c906b20e-1576-4a96-9c77-b67a940dcce8" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/04f553d2-8ac5-4085-8c1b-59bddf9deb41"><img src="https://github.com/user-attachments/assets/04f553d2-8ac5-4085-8c1b-59bddf9deb41" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Teddybear leaves his mark on the edge of a clear lake, surrounded by exotic flowers, and the picture is full of mystery and exploration.</td>
    <td style="text-align:center;">Teddybear climbs the rugged mountain road, the weather is changeable, but he is determined.</td>
    <td style="text-align:center;">The picture switches to the top of the mountain, where teddybear stands in the glow of the sunrise, with a magnificent mountain view in the background.</td>
    <td style="text-align:center;">On the way home, teddybear helps a wounded bird, the picture is warm and touching.</td>
    <td style="text-align:center;">Teddybear sits by the little girl's bed and tells her his adventure story, and the little girl is fascinated.</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/14c5f6f1-7830-46fc-9ebd-8611f339d8ab"><img src="https://github.com/user-attachments/assets/14c5f6f1-7830-46fc-9ebd-8611f339d8ab" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/705eb5c4-b084-4752-a47b-ab20d206b9ee"><img src="https://github.com/user-attachments/assets/705eb5c4-b084-4752-a47b-ab20d206b9ee" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/39a814b6-8692-41f5-819d-eecfd6085b03"><img src="https://github.com/user-attachments/assets/39a814b6-8692-41f5-819d-eecfd6085b03" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/e2b7297d-d0bb-482e-9c16-f01f9a56fbb0"><img src="https://github.com/user-attachments/assets/e2b7297d-d0bb-482e-9c16-f01f9a56fbb0" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/3afc9aa8-92f5-4e27-bfed-950300645748"><img src="https://github.com/user-attachments/assets/3afc9aa8-92f5-4e27-bfed-950300645748" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The scene shows a peaceful village, with moonlight shining on the roofs and streets, creating a peaceful atmosphere.</td>
    <td style="text-align:center;">cat sits by the window, her eyes twinkling in the night, reflecting her special connection with the moon and stars.</td>
    <td style="text-align:center;">Villagers gather in the center of the village for the annual Moon Festival celebration, with lanterns and colored lights adorning the night sky.</td>
    <td style="text-align:center;">cat feels the call of the moon, and her beard trembles with the excitement in her heart.</td>
    <td style="text-align:center;">cat quietly leaves her home in the night and embarks on a path illuminated by the silver moonlight.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/b055930e-4f97-4872-bc07-2e9cb5641d7d"><img src="https://github.com/user-attachments/assets/b055930e-4f97-4872-bc07-2e9cb5641d7d" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/ece35515-295d-4655-a5c5-cca95fd11e92"><img src="https://github.com/user-attachments/assets/ece35515-295d-4655-a5c5-cca95fd11e92" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/804e32f1-75ae-4a0b-b3c0-1d13c6e2d987"><img src="https://github.com/user-attachments/assets/804e32f1-75ae-4a0b-b3c0-1d13c6e2d987" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/e09d8134-7991-4585-a09a-b00283ab6a56"><img src="https://github.com/user-attachments/assets/e09d8134-7991-4585-a09a-b00283ab6a56" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/4c722b97-79d9-4281-8451-b31eb3393c3a"><img src="https://github.com/user-attachments/assets/4c722b97-79d9-4281-8451-b31eb3393c3a" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">A group of forest elves dance around glowing mushrooms, their costumes and movements full of magic and vitality.</td>
    <td style="text-align:center;">cat joins the celebration and dances with the elves, the picture is full of joy and freedom.</td>
    <td style="text-align:center;">A wise old owl reveals the secret power of the moon to cat and the light of the moon in the picture becomes brighter.</td>
    <td style="text-align:center;">cat closes her eyes in the moonlight, puts her hands together, and makes a wish, surrounded by the light of stars and the moon.</td>
    <td style="text-align:center;">cat feels the surge of power, and her eyes become more determined.</td>
  </tr>
</table>



<!-- ## ⏰ TODOs
- [ ] More demo and applications
- [ ] More functionalities such as control modules. (Suggestions are welcome!) -->


## 🔆 Information

### Code Structure
```
VideoTuna/
    ├── assets       # put images for readme
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
|HunyuanVideo|720x1280x129|[Hugging Face](https://huggingface.co/tencent/HunyuanVideo)
|Mochi|848x480, 3s|[Hugging Face](https://huggingface.co/genmo/mochi-1-preview)
|CogVideoX-2B|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b)
|CogVideoX-5B|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)
|Open-Sora 1.0|512×512x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)
|VideoCrafter2|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-5B-I2V|480x720x49|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
|DynamiCrafter|576x1024x16|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|

* Note: H: height; W: width; L: length
<!-- |Open-Sora 1.2|240p to 720p, 2~16s|[STDIT](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3), [VAE](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2) -->
<!-- |Open-Sora 1.1|144p & 240p & 480p, 0~15s|[Stage 2](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) -->
<!-- |Open-Sora 1.1|144p to 720p, 0~15s|[Stage 3](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3) -->

Please check [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) to download all the model checkpoints.

## 🔆 Get started

### 1.Prepare environment
``` shell
conda create -n videotuna python=3.10 -y
conda activate videotuna
pip install poetry
poetry install
poetry run pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
**Flash-attn installation (Optional)**

Hunyuan model uses it to reduce memory usage and speed up inference. If it is not installed, the model will run in normal mode.
``` shell
poetry run install-flash-attn 
```
### 2.Prepare checkpoints

Please follow [docs/CHECKPOINTS.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md) to download model checkpoints.  
After downloading, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md#checkpoint-orgnization-structure).

### 3.Inference state-of-the-art T2V/I2V/T2I models

- Inference a set of text-to-video models **in one command**: `bash tools/video_comparison/compare.sh`
  - The default mode is to run all models, e.g., `inference_methods="videocrafter2;dynamicrafter;cogvideo—t2v;cogvideo—i2v;opensora"`
  - If the users want to inference specific models, modify the `inference_methods` variable in `compare.sh`, and list the desired models separated by semicolons.
  - Also specify the input directory via the `input_dir` variable. This directory should contain a `prompts.txt` file, where each line corresponds to a prompt for the video generation. The default `input_dir` is `inputs/t2v`
- Inference a set of image-to-video models **in one command**: `bash tools/video_comparison/compare_i2v.sh`




<!-- |Task|Commands|
|:---------|:---------|
|T2V|`bash tools/video_comparison/compare.sh`|
|I2V|`TODO`|
 -->


- Inference a specific model, run the corresponding commands as follows:

Task|Model|Command|Length (#frames)|Resolution|Inference Time (s)|GPU Memory (GiB)|
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
|T2V|HunyuanVideo|`poetry run inference-hunyuan`|129|720x1280|1920|59.15|
|T2V|Mochi|`poetry run inference-mochi`|84|480x848|109.0|26|
|I2V|CogVideoX-5b-I2V|`poetry run inference-cogvideox-15-5b-i2v`|49|480x720|310.4|4.78|
|T2V|CogVideoX-2b|`poetry run inference-cogvideo-t2v-diffusers`|49|480x720|107.6|2.32|
|T2V|Open Sora V1.0|`poetry run inference-opensora-v10-16x256x256`|16|256x256|11.2|23.99|
|T2V|VideoCrafter-V2-320x512|`poetry run inference-vc2-t2v-320x512`|16|320x512|26.4|10.03|
|T2V|VideoCrafter-V1-576x1024|`poetry run inference-vc1-t2v-576x1024`|16|576x1024|91.4|14.57|
|I2V|DynamiCrafter|`poetry run inference-dc-i2v-576x1024`|16|576x1024|101.7|52.23|
|I2V|VideoCrafter-V1|`poetry run inference-vc1-i2v-320x512`|16|320x512|26.4|10.03|
|T2I|Flux-dev|`poetry run inference-flux-dev`|1|768x1360|238.1|1.18|
|T2I|Flux-schnell|`poetry run inference-flux-schnell`|1|768x1360|5.4|1.20|

**Flux-dev:** Trained using guidance distillation, it requires 40 to 50 steps to generate high-quality images.

**Flux-schnell:** Trained using latent adversarial diffusion distillation, it can generate high-quality images in only 1 to 4 steps.
### 4. Finetune T2V models
#### 4.1 Prepare dataset
Please follow the [docs/datasets.md](docs/datasets.md) to try provided toydataset or build your own datasets.

#### 4.2 Fine-tune

#### 1. VideoCrafter2 Full Fine-tuning
Before started, we assume you have finished the following two preliminary steps:
  1) [Install the environment](#1prepare-environment)
  2) [Prepare the dataset   ](#41-prepare-dataset)
  3) [Download the checkpoints](docs/CHECKPOINTS.md) and get these two checkpoints
```
  ll checkpoints/videocrafter/t2v_v2_512/model.ckpt
  ll checkpoints/stablediffusion/v2-1_512-ema/model.ckpt
```


First, run this command to convert the VC2 checkpoint as we make minor modifications on the keys of the state dict of the checkpoint. The converted checkpoint will be automatically save at `checkpoints/videocrafter/t2v_v2_512/model_converted.ckpt`.    
```
python tools/convert_checkpoint.py --input_path checkpoints/videocrafter/t2v_v2_512/model.ckpt
```


Second, run this command to start training on the single GPU. The training results will be automatically saved at `results/train/${CURRENT_TIME}_${EXPNAME}`    
```
poetry run train-videocrafter-v2
```





#### 2. VideoCrafter2 Lora Fine-tuning

We support lora finetuning to make the model to learn new concepts/characters/styles.   
- Example config file: `configs/001_videocrafter2/vc2_t2v_lora.yaml`  
- Training lora based on VideoCrafter2: `bash shscripts/train_videocrafter_lora.sh`  
- Inference the trained models: `bash shscripts/inference_vc2_t2v_320x512_lora.sh`   

#### 3. Open-Sora Fine-tuning
We support open-sora finetuning, you can simply run the following commands:
``` shell
# finetune the Open-Sora v1.0
poetry run train-opensorav10
```

#### 4. FLUX Lora Fine-tuning
We support flux lora finetuning, you can simply run the following commands:
``` shell
# finetune the Flux-Lora
poetry run train-flux-lora

# inference the lora model
poetry run inference-flux-lora
```
If you want to build your own dataset, please organize your data as `inputs/t2i/flux/plushie_teddybear`, which contains the training images and the corresponding text prompt files, as shown in the following directory structure. Then modify the `instance_data_dir` in`configs/006_flux/multidatabackend.json`.
```
owndata/
    ├── img1.jpg
    ├── img2.jpg  
    ├── img3.jpg           
    ├── ...
    ├── prompt1.txt      # prompt of img1.jpg
    ├── prompt2.txt      # prompt of img2.jpg
    ├── prompt3.txt      # prompt of img3.jpg
    ├── ...
``` 

<!-- Please check [configs/train/003_vc2_lora_ft/README.md](configs/train/003_vc2_lora_ft/README.md) for details.    -->
<!-- 

(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
``` -->

<!-- #### Finetuning for enhanced langugage understanding -->


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance. 
Please check [eval/README.md](docs/evaluation.md) for details.

<!-- ### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details. -->


## Acknowledgement
We thank the following repos for sharing their awesome models and codes!
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
