#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python src/sat/sample_video.py --base src/sat/configs/cogvideox_2b.yaml src/sat/configs/inference.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"