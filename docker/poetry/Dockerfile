FROM python:3.10-bookworm

ARG UID=1000
ARG GID=1000

WORKDIR /opt/VideoTuna/

RUN groupadd -g "${GID}" videotuna \
    && useradd -m -u "${UID}" -s /usr/bin/bash -g videotuna videotuna \
    && chown -R videotuna:videotuna /opt/VideoTuna/ \
    && chmod -R 755 /opt/VideoTuna/ \
    && pip install pipx

USER videotuna

WORKDIR /opt/VideoTuna/

ENV PATH="/home/videotuna/.local/bin:${PATH}"

RUN pipx ensurepath \
    && pipx install poetry \
    && poetry config virtualenvs.in-project true
