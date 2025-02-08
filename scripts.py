"""
Poetry commands
"""

import os
import subprocess
import sys
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d%H%M%S")


def install_flash_attn():
    """
    Install the flash attention package
    """
    command_install_cuda_nvcc = [
        "conda",
        "install",
        "-c",
        "nvidia",
        "cuda-nvcc",
        "-y",
    ] + sys.argv[1:]
    command_install_flash_attn = [
        "pip",
        "install",
        "flash-attn==2.7.3",
        "--no-build-isolation",
    ]
    result_nvcc = subprocess.run(command_install_cuda_nvcc, check=False)
    if result_nvcc.returncode != 0:
        exit(result_nvcc.returncode)

    result_flash = subprocess.run(command_install_flash_attn, check=False)
    exit(result_flash.returncode)


def code_format(check=False):
    """
    Run the code formatting
    """
    commands = [["isort", "."], ["black", "."]]
    return_code = 0

    for command in commands:
        if check:
            command.append("--check")
        process = subprocess.run(command, check=False)
        if process.returncode > 0:
            return_code = process.returncode
            break

    exit(return_code)


def code_format_check():
    """
    Check the code formatting (useful with CI)
    """
    code_format(check=True)


def lint():
    """
    Run the linter
    """
    result = subprocess.run(
        ["ruff", "check", "videotuna", "tests"] + sys.argv[1:], check=False
    )
    exit(result.returncode)


def test():  # pragma: no cover
    """
    Run all unittests
    """
    os.environ["ENV"] = "test"
    result = subprocess.run(["pytest", "."] + sys.argv[1:], check=False)
    exit(result.returncode)


def coverage_report():
    """
    Run all unittests with coverage
    """
    os.environ["ENV"] = "test"
    result = subprocess.run(
        ["coverage", "run", "-m", "pytest", "--junitxml", "report.xml"], check=False
    )
    if result.returncode > 0:
        exit(result.returncode)
    result = subprocess.run(["coverage", "report", "-m"], check=False)
    exit(result.returncode)


def type_check():
    """
    Run the type checking
    """
    result = subprocess.run(["mypy", "videotuna", "tests"], check=False)
    exit(result.returncode)


def inference_cogvideo_i2v_diffusers():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_cogVideo_diffusers.py",
            "--generate_type",
            "i2v",
            "--model_input",
            "inputs/i2v/576x1024",
            "--model_path",
            "checkpoints/cogvideo/CogVideoX-5b-I2V",
            "--output_path",
            "results/cogvideo-test-i2v",
            "--num_inference_steps",
            "50",
            "--guidance_scale",
            "3.5",
            "--num_videos_per_prompt",
            "1",
            "--dtype",
            "float16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_i2v_lora():
    config = "configs/004_cogvideox/cogvideo5b-i2v.yaml"
    ckpt = "results/train/cogvideox_i2v_5b/{YOUR_CKPT_PATH}.ckpt"
    prompt_dir = "{YOUR_PROMPT_DIR}"

    savedir = f"results/inference/i2v/cogvideox-i2v-lora-{current_time}"

    result = subprocess.run(
        [
            "python3",
            "scripts/inference_cogvideo.py",
            "--config",
            config,
            "--ckpt_path",
            ckpt,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "480",
            "--width",
            "720",
            "--fps",
            "16",
            "--seed",
            "6666",
            "--mode",
            "i2v",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_lora():
    config = "configs/004_cogvideox/cogvideo2b.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = f"results/t2v/{current_time}-cogvideo"
    ckpt = "{YOUR_CKPT_PATH}"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_cogvideo.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "480",
            "--width",
            "720",
            "--fps",
            "16",
            "--seed",
            "6666",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_t2v_diffusers():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_cogVideo_diffusers.py",
            "--model_input",
            "A cat playing with a ball",
            "--model_path",
            "checkpoints/cogvideo/CogVideoX-2b",
            "--output_path",
            "results/output.mp4",
            "--num_inference_steps",
            "50",
            "--guidance_scale",
            "3.5",
            "--num_videos_per_prompt",
            "1",
            "--dtype",
            "float16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideox1_5_5b_i2v():
    load_transformer = "checkpoints/cogvideo/CogVideoX1.5-5B-SAT/transformer_i2v"
    input_file = "inputs/i2v/576x1024/test_prompts.txt"
    output_dir = "results/i2v/"
    base = "configs/005_cogvideox1.5/cogvideox1.5_5b.yaml"
    image_folder = "inputs/i2v/576x1024/"

    result = subprocess.run(
        [
            "python",
            "scripts/inference_cogVideo_sat_refactor.py",
            "--load_transformer",
            load_transformer,
            "--input_file",
            input_file,
            "--output_dir",
            output_dir,
            "--base",
            base,
            "--mode_type",
            "i2v",
            "--sampling_num_frames",
            "22",
            "--image_folder",
            image_folder,
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideox1_5_5b_t2v():
    load_transformer = "checkpoints/cogvideo/CogVideoX1.5-5B-SAT/transformer_t2v"
    input_file = "inputs/t2v/prompts.txt"
    output_dir = "results/t2v/"
    base = "configs/005_cogvideox1.5/cogvideox1.5_5b.yaml"

    result = subprocess.run(
        [
            "python",
            "scripts/inference_cogVideo_sat_refactor.py",
            "--load_transformer",
            load_transformer,
            "--input_file",
            input_file,
            "--output_dir",
            output_dir,
            "--base",
            base,
            "--mode_type",
            "t2v",
            "--sampling_num_frames",
            "22",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_dc_i2v_576x1024():
    ckpt = "checkpoints/dynamicrafter/i2v_576x1024/model.ckpt"
    config = "configs/002_dynamicrafter/dc_i2v_1024.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/dc-i2v-576x1024"

    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--mode",
            "i2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "576",
            "--width",
            "1024",
            "--fps",
            "10",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_flux_schnell():
    prompt = "inputs/t2v/prompts.txt"
    width = 1360
    height = 768

    command_schnell = [
        "python",
        "scripts/inference_flux.py",
        "--model_type",
        "schnell",
        "--prompt",
        prompt,
        "--out_path",
        "results/flux-schnell/",
        "--width",
        str(width),
        "--height",
        str(height),
        "--num_inference_steps",
        "4",
        "--guidance_scale",
        "0.",
    ] + sys.argv[1:]

    result_schnell = subprocess.run(command_schnell, check=False)
    exit(result_schnell.returncode)


def inference_flux_dev():
    prompt = "inputs/t2v/prompts.txt"
    width = 1360
    height = 768

    command_dev = [
        "python",
        "scripts/inference_flux.py",
        "--model_type",
        "dev",
        "--prompt",
        prompt,
        "--out_path",
        "results/flux-dev/",
        "--width",
        str(width),
        "--height",
        str(height),
        "--num_inference_steps",
        "50",
        "--guidance_scale",
        "0.",
    ] + sys.argv[1:]

    result_dev = subprocess.run(command_dev, check=False)
    exit(result_dev.returncode)


def inference_flux_lora():
    os.environ["lora_ckpt"] = "{YOUR_CORA_CKPT_PATH}"
    result = subprocess.run(
        [
            "python",
            "scripts/inference_flux_lora.py",
            "--model_type",
            "dev",
            "--prompt",
            "inputs/t2v/prompts.txt",
            "--out_path",
            "results/t2i/flux-lora/",
            "--lora_path",
            os.environ["lora_ckpt"],
            "--width",
            "1360",
            "--height",
            "768",
            "--num_inference_steps",
            "50",
            "--guidance_scale",
            "3.5",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_hunyuan.py",
            "--video-size",
            "544",
            "960",
            "--video-length",
            "129",
            "--infer-steps",
            "50",
            "--prompt",
            "A cat walks on the grass, realistic style.",
            "--flow-reverse",
            "--use-cpu-offload",
            "--save-path",
            "./results/hunyuan",
            "--model-base",
            "./checkpoints/hunyuan",
            "--dit-weight",
            "./checkpoints/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "--seed",
            "43",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_mochi():
    ckpt = "checkpoints/mochi-1-preview"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = "results/t2v/mochi2"
    height = 480
    width = 848
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_mochi.py",
            "--ckpt_path",
            ckpt,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            str(height),
            "--width",
            str(width),
            "--fps",
            "28",
            "--seed",
            "124",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_opensora_v10_16x256x256():
    ckpt = "checkpoints/open-sora/t2v_v10/OpenSora-v1-HQ-16x256x256.pth"
    config = "configs/003_opensora/opensorav10_256x256.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = f"results/t2v/{current_time}-opensorav10-HQ-16x256x256"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--seed",
            "123",
            "--mode",
            "t2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--savedir",
            res_dir,
            "--n_samples",
            "3",
            "--bs",
            "2",
            "--height",
            "256",
            "--width",
            "256",
            "--unconditional_guidance_scale",
            "7.0",
            "--ddim_steps",
            "50",
            "--ddim_eta",
            "1.0",
            "--prompt_file",
            prompt_file,
            "--fps",
            "8",
            "--frames",
            "16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_v2v_ms():
    input_dir = "inputs/v2v/001"
    output_dir = f"results/v2v/{current_time}-v2v-modelscope-001"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_v2v_ms.py",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc1_i2v_320x512():
    ckpt = "checkpoints/videocrafter/i2v_v1_512/model.ckpt"
    config = "configs/000_videocrafter/vc1_i2v_512.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/i2v/vc1-i2v-320x512"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--mode",
            "i2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "320",
            "--width",
            "512",
            "--fps",
            "8",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc1_t2v_576x1024():
    ckpt = "checkpoints/videocrafter/t2v_v1_1024/model.ckpt"
    config = "configs/000_videocrafter/vc1_t2v_1024.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = "results/t2v/videocrafter1-576x1024"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            res_dir,
            "--bs",
            "1",
            "--height",
            "576",
            "--width",
            "1024",
            "--fps",
            "28",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc2_t2v_320x512():
    # Dependencies
    ckpt = "checkpoints/videocrafter/t2v_v2_512/model.ckpt"
    config = "configs/001_videocrafter2/vc2_t2v_320x512.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = f"results/t2v/{current_time}-videocrafter2"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "320",
            "--width",
            "512",
            "--fps",
            "28",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc2_t2v_320x512_lora():
    # Dependencies
    ckpt = "checkpoints/videocrafter/t2v_v2_512/model.ckpt"
    config = "configs/001_videocrafter2/vc2_t2v_lora.yaml"
    lorackpt = "YOUR_LORA_CKPT"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = "results/train/003_vc2_lora_ft"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--seed",
            "123",
            "--mode",
            "t2v",
            "--ckpt_path",
            ckpt,
            "--lorackpt",
            lorackpt,
            "--config",
            config,
            "--savedir",
            res_dir,
            "--n_samples",
            "1",
            "--bs",
            "1",
            "--height",
            "320",
            "--width",
            "512",
            "--unconditional_guidance_scale",
            "12.0",
            "--ddim_steps",
            "50",
            "--ddim_eta",
            "1.0",
            "--prompt_file",
            prompt_file,
            "--fps",
            "28",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_i2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo5b-i2v.yaml"  # Experiment config

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_i2v_5b"  # Experiment name

    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_t2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo2b.yaml"  # Experiment config

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_t2v_5b"  # Experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_dynamicrafter():
    # Dependencies
    sdckpt = "checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"
    dcckpt = "checkpoints/dynamicrafter/i2v_576x1024/model_converted.ckpt"

    # Experiment settings
    expname = "002_dynamicrafterft_1024"  # Experiment name
    config = "configs/002_dynamicrafter/dc_i2v_1024.yaml"  # Experiment config
    resroot = "results/train"  # Experiment saving directory
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--name",
            f"{current_time}_{expname}",
            "--base",
            config,
            "--logdir",
            resroot,
            "--sdckpt",
            sdckpt,
            "--ckpt",
            dcckpt,
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_flux_lora():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CONFIG_PATH"] = "configs/006_flux/config"
    os.environ["DATACONFIG_PATH"] = "configs/006_flux/multidatabackend"
    os.environ["CONFIG_BACKEND"] = "json"
    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--mixed_precision=bf16",
            "--num_processes=1",
            "--num_machines=1",
            "scripts/train_flux_lora.py",
            "--config_path",
            f"{os.environ['CONFIG_PATH']}.{os.environ['CONFIG_BACKEND']}",
            "--data_config_path",
            f"{os.environ['DATACONFIG_PATH']}.{os.environ['CONFIG_BACKEND']}",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_opensorav10():
    # Experiment settings
    expname = "run_macvid_t2v512"  # Experiment name
    config = "configs/003_opensora/opensorav10_256x256.yaml"  # Experiment config
    logdir = "./results"  # Experiment saving directory
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--base",
            config,
            "--name",
            f"{current_time}_{expname}",
            "--logdir",
            logdir,
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_videocrafter_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    vc2_ckpt = "checkpoints/videocrafter/t2v_v2_512/model.ckpt"

    # Experiment settings
    expname = "train_t2v_512_lora"  # Experiment name
    config = "configs/001_videocrafter2/vc2_t2v_lora.yaml"  # Experiment config
    resroot = "results/train"  # Experiment saving directory

    # Generate current time
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--name",
            f"{current_time}_{expname}",
            "--base",
            config,
            "--logdir",
            resroot,
            "--ckpt",
            vc2_ckpt,
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_videocrafter_v2():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    sdckpt = "checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"  # pretrained checkpoint of stablediffusion 2.1
    vc2_ckpt = "checkpoints/videocrafter/t2v_v2_512/model_converted.ckpt"  # pretrained checkpoint of videocrafter2
    config = "configs/001_videocrafter2/vc2_t2v_320x512.yaml"  # experiment config: model+data+training

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "videocrafter2_320x512"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--sdckpt",
            sdckpt,
            "--ckpt",
            vc2_ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)
