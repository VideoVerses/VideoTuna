import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from pydantic import Field
from pydantic_core import ValidationError
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict, SettingsError

from videotuna.utils.inference_utils import load_inputs_v2v


class Settings(BaseSettings, cli_parse_args=True, cli_prog_name="inference_v2v_ms"):
    ckpt_path: str = Field(
        "checkpoints/Video-to-Video", description="Checkpoint path of the model"
    )
    input_dir: str = Field(
        ...,
        description="A input directory containing videos and prompts for video-to-video enhancement",
    )
    output_dir: str = Field(..., description="Results saving directory")


def inference_v2v_ms(settings: Settings):
    # prepare arguments, model, pipeline.
    model = Model.from_pretrained(settings.ckpt_path)
    pipe = pipeline(
        task="video-to-video", model=model, model_revision="v1.1.0", device="cuda:0"
    )
    print(f"Successfully loaded model from {settings.ckpt_path}")

    os.makedirs(settings.output_dir, exist_ok=True)

    # load input prompts, video paths, video filenames
    prompt_list, video_filepaths, video_filenames = load_inputs_v2v(
        input_dir=settings.input_dir
    )

    # video-to-video enhancement
    for i, (prompt, video_filepath, video_filename) in enumerate(
        zip(prompt_list, video_filepaths, video_filenames)
    ):
        print(f"[{i}:03d] input path: {video_filepath}")
        print(f"[{i}:03d] input name: {video_filename}")
        print(f"[{i}:03d] prompt: {prompt}")
        p_input = {"video_path": video_filepath, "text": prompt}
        output_video_path = pipe(
            p_input, output_video=os.path.join(settings.output_dir, video_filename)
        )[OutputKeys.OUTPUT_VIDEO]
        print(
            f"Successfully processed {video_filename} and saved to {output_video_path}"
        )


if __name__ == "__main__":
    try:
        settings = CliApp.run(
            Settings,
        )
        inference_v2v_ms(settings)
    except SystemExit as e:
        print(e)
    except ValidationError as e:
        print(e)
        print("Use --help for more info")
