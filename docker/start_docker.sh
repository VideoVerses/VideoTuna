cd ./docker
docker build -t privtune:v1.0 -t videotuna:v1.0 .
docker run --gpus all -it privtune:v1.0 /bin/bash

# if you want to use your local path:
docker run  -v /path/to/VideoTuna/:/content/local_VideoTuna --gpus all -it privtune:v1.0 /bin/bash
