import os

# 设置输入和输出目录
input_dir = "/Users/haoyu/VideoTuna/assets"
output_dir = "/Users/haoyu/VideoTuna/assets"

import cv2
from PIL import Image

def convert_mp4_to_gif(mp4_file, gif_file):
    # 使用 OpenCV 读取视频
    cap = cv2.VideoCapture(mp4_file)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB 格式并添加到帧列表
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))

    cap.release()
    
    # 保存为 GIF
    frames[0].save(gif_file, save_all=True, append_images=frames[1:], duration=100, loop=0)



# 设置输入和输出目录
input_dir = "/Users/haoyu/VideoTuna/assets/demos"
output_dir = "/Users/haoyu/VideoTuna/assets/demos"

# 遍历输入目录中的所有 MP4 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        mp4_path = os.path.join(input_dir, filename)
        gif_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.gif")
        convert_mp4_to_gif(mp4_path, gif_path)
        print(f"Converted {mp4_path} to {gif_path}")
