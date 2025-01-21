import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline

from videotuna.utils.inference_utils import load_inputs_v2v


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/Video-to-Video",
        help="Checkpoint path of the model",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="A input directory containing videos and prompts for video-to-video enhancement",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results saving directory"
    )
    return parser


# prepare arguments, model, pipeline.
args = get_parser().parse_args()
model = Model.from_pretrained(args.ckpt_path)
pipe = pipeline(
    task="video-to-video", model=model, model_revision="v1.1.0", device="cuda:0"
)
print(f"Successfully loaded model from {args.ckpt_path}")

os.makedirs(args.output_dir, exist_ok=True)

# load input prompts, video paths, video filenames
prompt_list, video_filepaths, video_filenames = load_inputs_v2v(
    input_dir=args.input_dir
)

# video-to-video enhancement
for i, (prompt, videofilepath, videofilename) in enumerate(
    zip(prompt_list, video_filepaths, video_filenames)
):
    print(f"[{i}:03d] input path: {videofilepath}")
    print(f"[{i}:03d] input name: {videofilename}")
    print(f"[{i}:03d] prompt: {prompt}")
    p_input = {"video_path": videofilepath, "text": prompt}
    output_video_path = pipe(
        p_input, output_video=os.path.join(args.output_dir, videofilename)
    )[OutputKeys.OUTPUT_VIDEO]
    print(f"Successfully processed {videofilename} and saved to {output_video_path}")
