# Usage: 1. cd ./docker
# Usage: 2. docker build -t videotuna:v1.0 .
# Usage: 3. docker run --gpus all -it videotuna:v1.0 /bin/bash



FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/videotuna/.local/bin:${PATH}"

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y
RUN apt install -y aria2 git git-lfs unzip ffmpeg python3-tk

# ---------------------------- Set up python env ----------------------------
RUN git clone https://github.com/VideoVerses/VideoTuna.git &&\
    cd VideoTuna && \
    pip install -r requirements.txt && \
    git clone https://github.com/JingyeChen/SwissArmyTransformer && \
    pip install -e SwissArmyTransformer/ && \
    git clone https://github.com/tgxs002/HPSv2.git && \
    cd ./HPSv2 && \
    pip install -e . && \
    cd .. && \
    pip install ffmpeg

# ---------------------------- For add check points ----------------------------
# RUN mkdir checkpoints

# ---------------------------- T2V ----------------------------

# # ---- CogVideo (diffusers) ----
# RUN mkdir -p checkpoints/cogvideo; cd checkpoints/cogvideo
# RUN git clone https://huggingface.co/THUDM/CogVideoX-2b         # This are checkpoints for CogVideoX T2V-2B
# # RUN git clone https://huggingface.co/THUDM/CogVideoX-5b         # This are checkpoints for CogVideoX T2V-5B
# # RUN git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V     # This are checkpoints for CogVideoX I2V-5B
# # RUN git clone https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT  # This are checkpoints for CogVideoX 1.5-5B (both T2V and I2V)


# # ---- Open-Sora ----
# RUN mkdir -p checkpoints/open-sora/t2v_v10
# # RUN wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth -P checkpoints/open-sora/t2v_v10/
# # RUN wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
# RUN wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/mai/n/OpenSora-v1-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
# #
# RUN mkdir -p checkpoints/open-sora/t2v_v11
# RUN cd checkpoints/open-sora/t2v_v11
# RUN git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2
# RUN git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3
# RUN cd ../../..
# #
# RUN mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3
# RUN mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
# RUN wget https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
# RUN wget https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3


# # ---- Videocrafter ----
# RUN mkdir checkpoints/videocrafter/

# RUN mkdir checkpoints/videocrafter/t2v_v2_512
# RUN wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v2_512  # videocrafter2-t2v-512

# RUN mkdir checkpoints/videocrafter/t2v_v1_1024
# RUN wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v1_1024 # videocrafter1-t2v-1024


# # ---------------------------- I2V ----------------------------
# # ---- Dynamicrafter ----
# RUN mkdir checkpoints/dynamicrafter/
# RUN mkdir checkpoints/dynamicrafter/i2v_576x1024

# RUN wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt -P checkpoints/dynamicrafter/i2v_576x1024  # dynamicrafter-i2v-1024

# # ---- Videocrafter ----
# RUN mkdir -p checkpoints/videocrafter/i2v_v1_512

# RUN wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt -P checkpoints/videocrafter/i2v_v1_512 # videocrafter1-i2v-512

# # ---- Stable Diffusion checkpoint for VC2 Training ----
# RUN mkdir -p checkpoints/stablediffusion/v2-1_512-ema
# RUN wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -P checkpoints/stablediffusion/v2-1_512-ema
