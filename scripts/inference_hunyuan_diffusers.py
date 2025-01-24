import torch
import argparse
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video


# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for Hunyuan Video Inference")
    parser.add_argument('--ckpt-path', type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument('--lora-path', type=str, required=True, help="Path to the LoRA weights")
    parser.add_argument('--lora-weight', type=float, required=True, help="Weight for the LoRA model")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt for the video generation")
    parser.add_argument('--video-size', type=int, nargs=2, required=True,
                        help="Height and width of the generated video")
    parser.add_argument('--video-frame-length', type=int, required=True, help="Number of frames in the video")
    parser.add_argument('--video-fps', type=int, required=True, help="Frames per second for the output video")
    parser.add_argument('--infer-steps', type=int, required=True, help="Number of inference steps")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the output video")

    return parser.parse_args()


# Main function
def main():
    # Parse arguments
    args = parse_args()

    # Load model and pipeline
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.ckpt_path, subfolder="transformer",
                                                                 torch_dtype=torch.bfloat16)
    pipe = HunyuanVideoPipeline.from_pretrained(args.ckpt_path, transformer=transformer, torch_dtype=torch.float16)

    # Load LoRA weights
    pipe.load_lora_weights(args.lora_path, adapter_name="hunyuanvideo-lora")
    pipe.set_adapters(["hunyuanvideo-lora"], [args.lora_weight])

    # Enable tiling and move to GPU
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    # Generate video frames
    output = pipe(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        num_frames=args.video_frame_length,
        num_inference_steps=args.infer_steps,
    ).frames[0]

    # Export to video
    export_to_video(output, args.output_path, fps=args.video_fps)


if __name__ == "__main__":
    main()
