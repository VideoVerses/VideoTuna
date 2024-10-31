<p align="center" width="50%">
<img src="assets/logo.jpg" alt="VideoTuna" style="width: 30%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna

![Version](https://img.shields.io/badge/version-0.01-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=VideoVerses.VideoTuna&left_color=green&right_color=red) [![GitHub](https://img.shields.io/github/stars/VideoVerses/VideoTuna?style=social)](https://github.com/VideoVerses/VideoTuna) [![](https://dcbadge.limes.pink/api/server/AammaaR2?style=flat)](https://discord.gg/AammaaR2) <a href='assets/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>



🤗🤗🤗 VideoTuna is an open-sourcing finetuning framework for text-to-video generation.


## Features
1. All in one framework: Inference and finetune state-of-the-art T2V models.
2. Continuous training.
3. Fintuning: domain-specific.
4. Fintuning: enhanced language understanding.
5. Fintuning: enhancement.
6. Human preference alignment/Post-training: RLHF, DPO.


## Demo
### VAE

<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtview7.gif"><img src="assets/demos/vae/fkview7.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkview7.gif"><img src="assets/demos/vae/fkview7.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Groud Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtview9.gif"><img src="assets/demos/vae/fkview9.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkview9.gif"><img src="assets/demos/vae/fkview9.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtface.gif"><img src="assets/demos/vae/gtface.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkface.gif"><img src="assets/demos/vae/fkface.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>

  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtmotion4.gif"><img src="assets/demos/vae/gtmotion4.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkmotion4.gif"><img src="assets/demos/vae/fkmotion4.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtanimal2.gif"><img src="assets/demos/vae/gtanimal2.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkanimal2.gif"><img src="assets/demos/vae/fkanimal2.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtcloseshot1.gif"><img src="assets/demos/vae/gtcloseshot1.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkcloseshot1.gif"><img src="assets/demos/vae/fkcloseshot1.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
</table>

### Face domain

<table class="center">
  <tr>
    <td><img src="assets/demos/face_i2v/zcCWO3QOguA_0.png" width="240" alt="Image 1"></td>
    <td><img src="assets/demos/face_i2v/YJJbE-w2qzA_0.png" width="240" alt="Image 2"></td>
    <td><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0.png" width="240" alt="Image 3"></td>
  </tr>
  <tr>
    <td style="text-align: center;">Input 1</td>
    <td style="text-align: center;">Input 2</td>
    <td style="text-align: center;">Input 3</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_anger.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_anger.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_disgust.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_disgust.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_fear.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_fear.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_happy.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_happy.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_sad.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_sad.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_surprise.gif"><img src="assets/demos/face_i2v/zcCWO3QOguA_0_sample0/zcCWO3QOguA_0_sample0_surprise.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_anger.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_anger.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_disgust.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_disgust.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_fear.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_fear.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_happy.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_happy.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_sad.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_sad.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_surprise.gif"><img src="assets/demos/face_i2v/YJJbE-w2qzA_0_sample0/YJJbE-w2qzA_0_sample0_surprise.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_anger.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_anger.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_disgust.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_disgust.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_fear.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_fear.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_happy.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_happy.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_sad.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_sad.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_surprise.gif"><img src="assets/demos/face_i2v/-ZxtmDbqDRc_0_sample0/-ZxtmDbqDRc_0_sample0_surprise.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


### Storytelling


<table class="center">
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/0_9.gif"><img src="assets/demos/story/bear/0_9.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/1_4.gif"><img src="assets/demos/story/bear/1_4.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/2_5.gif"><img src="assets/demos/story/bear/2_5.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/3_6.gif"><img src="assets/demos/story/bear/3_6.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/4_4.gif"><img src="assets/demos/story/bear/4_4.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The picture shows a cozy room with a little girl telling her travel story to her teddybear beside the bed.</td>
    <td style="text-align:center;">As night falls, teddybear sits by the window, his eyes sparkling with longing for the distant place</td>
    <td style="text-align:center;">Teddybear was in a corner of the room, making a small backpack out of old cloth strips, with a map, a compass and dry food next to it.</td>
    <td style="text-align:center;">The first rays of sunlight in the morning came through the window, and teddybear quietly opened the door and embarked on his adventure.</td>
    <td style="text-align:center;">In the forest, the sun shines through the treetops, and teddybear moves among various animals and communicates with them.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/5_5.gif"><img src="assets/demos/story/bear/5_5.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/6_2.gif"><img src="assets/demos/story/bear/6_2.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/7_4.gif"><img src="assets/demos/story/bear/7_4.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/8_4.gif"><img src="assets/demos/story/bear/8_4.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/bear/10_5.gif"><img src="assets/demos/story/bear/10_5.gif" width="240"></a></td>
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
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/0_2.gif"><img src="assets/demos/story/cat/0_2.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/1_3.gif"><img src="assets/demos/story/cat/1_3.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/2_4.gif"><img src="assets/demos/story/cat/2_4.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/3_7.gif"><img src="assets/demos/story/cat/3_7.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/4_3.gif"><img src="assets/demos/story/cat/4_3.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The scene shows a peaceful village, with moonlight shining on the roofs and streets, creating a peaceful atmosphere.</td>
    <td style="text-align:center;">cat sits by the window, her eyes twinkling in the night, reflecting her special connection with the moon and stars.</td>
    <td style="text-align:center;">Villagers gather in the center of the village for the annual Moon Festival celebration, with lanterns and colored lights adorning the night sky.</td>
    <td style="text-align:center;">cat feels the call of the moon, and her beard trembles with the excitement in her heart.</td>
    <td style="text-align:center;">cat quietly leaves her home in the night and embarks on a path illuminated by the silver moonlight.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/8_8.gif"><img src="assets/demos/story/cat/8_8.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/9_2.gif"><img src="assets/demos/story/cat/9_2.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/11_0.gif"><img src="assets/demos/story/cat/11_0.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/12_1.gif"><img src="assets/demos/story/cat/12_1.gif" width="240"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/story/cat/15_9.gif"><img src="assets/demos/story/cat/15_9.gif" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">A group of forest elves dance around glowing mushrooms, their costumes and movements full of magic and vitality.</td>
    <td style="text-align:center;">cat joins the celebration and dances with the elves, the picture is full of joy and freedom.</td>
    <td style="text-align:center;">A wise old owl reveals the secret power of the moon to cat and the light of the moon in the picture becomes brighter.</td>
    <td style="text-align:center;">cat closes her eyes in the moonlight, puts her hands together, and makes a wish, surrounded by the light of stars and the moon.</td>
    <td style="text-align:center;">cat feels the surge of power, and her eyes become more determined.</td>
  </tr>
</table>



## ⏰TODOs
- [ ] More demo and applications
- [ ] More functionalities

## 🔆 Updates
- [2024-10-31] We make the VideoTuna V0.1.0 public!


## 🔆 Information

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
cd ./HPSv2
pip install -e .
cd ..
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

<a href="https://github.com/VideoVerses/VideoTuna/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=VideoVerses/VideoTuna" />
</a>

## 📋 License

## 😊 Citation
```
To be updated...
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VideoVerses/VideoTuna&type=Date)](https://star-history.com/#VideoVerses/VideoTuna&Date)
