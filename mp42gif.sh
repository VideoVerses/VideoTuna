#!/bin/bash

# 设置输入和输出目录
input_dir="/Users/haoyu/VideoTuna/assets/demos"
output_dir="/Users/haoyu/VideoTuna/assets/demos"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 遍历所有 MP4 文件
for mp4_file in "$input_dir"/*.mp4; do
    # 获取文件名（去掉扩展名）
    filename=$(basename "$mp4_file" .mp4)
    
    # 设置输出 GIF 文件的路径
    output_gif="$output_dir/$filename.gif"
    
    # 使用 ffmpeg 转换 MP4 为 GIF
    ffmpeg -i "$mp4_file" -vf "fps=10,scale=320:-1:flags=lanczos" -c:v gif "$output_gif"
    
    echo "Converted $mp4_file to $output_gif"
done
