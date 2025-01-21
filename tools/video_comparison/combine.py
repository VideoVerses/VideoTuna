import argparse
import glob
import os

import numpy as np
from moviepy.editor import VideoFileClip, clips_array
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description="Check the input directory")
parser.add_argument(
    "--input_dir", type=str, help="The input should be a directory", required=True
)
parser.add_argument(
    "--save_dir", type=str, help="The directory of saving results", required=True
)
parser.add_argument(
    "--unified_height", type=int, help="The height of the unified video", default=320
)
args = parser.parse_args()

methods = glob.glob(f"{args.save_dir}/*/*")
print(f"methods: {methods}")
num_of_videos = len(os.listdir(methods[0]))
print(f"number of videos: {num_of_videos}")
videos = [i for i in os.listdir(methods[0]) if i.endswith(".mp4")]

prompts = open(f"{args.input_dir}/prompts.txt", "r").readlines()


def add_text_to_frame(frame, text="hi", position=(0, 0)):
    # print(f'index {index}')
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("tools/video_comparison/Arial.ttf", size=24)
    draw.text(position, text, font=font, fill="white")
    return np.array(img)


for video_index in range(num_of_videos):

    video_paths = []
    for method in methods:
        video_path = sorted(os.listdir(method))[video_index]
        video_paths.append(f"{method}/{video_path}")

    clips = [VideoFileClip(video_path) for video_path in video_paths]
    max_fps = max([clip.fps for clip in clips])
    max_duration = max([clip.duration for clip in clips])
    clips = [clip.set_end(max_duration).set_fps(max_fps) for clip in clips]

    clips = [clip.resize(height=args.unified_height) for clip in clips]

    clips_with_name = []
    for index, clip in enumerate(clips):
        method = methods[index].split("/")[-1]
        print(f"tackling {index} {method}")
        clip = clip.fl_image(
            lambda frame, method=method: add_text_to_frame(
                frame, text=method, position=(0, 0)
            )
        )
        clips_with_name.append(clip)
    clips = clips_with_name

    final_clip = clips_array([clips])

    prompt_str = prompts[video_index].strip().replace(" ", "_")
    output_path = f"{args.save_dir}/combined_video_{video_index}_{prompt_str}.mp4"
    final_clip.write_videofile(output_path, codec="libx264", fps=max_fps)
    print(f"video {video_index} has been saved to {output_path}")
