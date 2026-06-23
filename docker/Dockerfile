# Usage: 1. cd ./docker
# Usage: 2. docker build -t privtune:latest .
# Usage: 3. docker run --gpus all -it privtune:latest /bin/bash
#
# Legacy VideoTuna all-in-one image — prefer Poetry install from repo root
# (see README.md) for PrivTune domain training.

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/privtune/.local/bin:${PATH}"

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y
RUN apt install -y aria2 git git-lfs unzip ffmpeg python3-tk

# PrivTune: clone repo and install via Poetry (CUDA + training).
# Checkpoint download: see docs/checkpoints.md (Wan 2.1 local, Flux/Wan2.2 from HF hub).
