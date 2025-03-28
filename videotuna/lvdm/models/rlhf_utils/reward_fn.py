# adapted from VADER  https://github.com/mihirp1998/VADER
import argparse
import os
import sys

import torch

sys.path.insert(
    1, os.path.join(sys.path[0], "..", "..")
)  # setting path to get Core and assets

import hpsv2
import torchvision
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from lvdm.models.rlhf_utils.actpred_scorer import ActPredScorer
from lvdm.models.rlhf_utils.aesthetic_scorer import AestheticScorerDiff
from lvdm.models.rlhf_utils.compression_scorer import JpegCompressionScorer
from lvdm.models.rlhf_utils.weather_scorer import WeatherScorer
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

# import ipdb
# st = ipdb.set_trace


def create_output_folders(output_dir, run_name):
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    return out_dir


# to convert string to boolean in argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=20230211, help="seed for seed_everything"
    )
    parser.add_argument(
        "--mode",
        default="base",
        type=str,
        help="which kind of inference mode: {'base', 'i2v'}",
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="steps of ddim if positive, otherwise use DDPM",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="image height, in pixel space"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="image width, in pixel space"
    )
    parser.add_argument(
        "--frames", type=int, default=-1, help="frames num to inference"
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="prompt classifier-free guidance",
    )
    parser.add_argument(
        "--unconditional_guidance_scale_temporal",
        type=float,
        default=None,
        help="temporal consistency guidance",
    )
    ## for conditional i2v only
    parser.add_argument(
        "--cond_input", type=str, default=None, help="data dir of conditional input"
    )
    ## for training
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="batch size for validation"
    )
    parser.add_argument(
        "--num_val_runs",
        type=int,
        default=1,
        help="total number of validation samples = num_val_runs * num_gpus * num_val_batch",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="aesthetic",
        help="reward function: 'aesthetic', 'hps', 'aesthetic_hps', 'pick_score', 'rainy', 'snowy', 'objectDetection', 'actpred', 'compression'",
    )
    parser.add_argument(
        "--compression_model_path",
        type=str,
        default="../pretrained_models/compression_reward.pt",
        help="compression model path",
    )  # The compression model is used only when reward_fn is 'compression'
    # The "book." is for grounding-dino model . Remember to add "." at the end of the object name for grounding-dino model.
    # But for yolos model, do not add "." at the end of the object name. Instead, you should set the object name to "book" for example.
    parser.add_argument(
        "--target_object",
        type=str,
        default="book",
        help="target object for object detection reward function",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="yolos-base",
        help="object detection model",
        choices=[
            "yolos-base",
            "yolos-tiny",
            "grounding-dino-base",
            "grounding-dino-tiny",
        ],
    )
    parser.add_argument(
        "--hps_version", type=str, default="v2.1", help="hps version: 'v2.0', 'v2.1'"
    )
    parser.add_argument(
        "--prompt_fn", type=str, default="hps_custom", help="prompt function"
    )
    parser.add_argument(
        "--nouns_file", type=str, default="simple_animals.txt", help="nouns file"
    )
    parser.add_argument(
        "--activities_file", type=str, default="activities.txt", help="activities file"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=10000, help="max training steps"
    )
    parser.add_argument(
        "--backprop_mode",
        type=str,
        default="last",
        help="backpropagation mode: 'last', 'rand', 'specific'",
    )  # backprop_mode != None also means training mode for batch_ddim_sampling
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        help="mixed precision training: 'no', 'fp8', 'fp16', 'bf16'",
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        default="wandb",
        help="logger type: 'wandb', 'tensorboard'",
    )
    parser.add_argument(
        "--project_dir", type=str, default="./project_dir", help="project directory"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help="The frequency of validation, e.g., 1 means validate every 1*accelerator.num_processes steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1,
        help="The frequency of checkpointing",
    )
    parser.add_argument(
        "--use_wandb", type=str2bool, default=True, help="use wandb for logging"
    )
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")
    parser.add_argument("--debug", type=str2bool, default=False, help="debug mode")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="max gradient norm"
    )
    parser.add_argument(
        "--use_AdamW8bit", type=str2bool, default=False, help="use AdamW8bit optimizer"
    )
    parser.add_argument(
        "--is_sample_preview",
        type=str2bool,
        default=True,
        help="sample preview during training",
    )
    parser.add_argument(
        "--decode_frame",
        type=str,
        default="-1",
        help="decode frame: '-1', 'fml', 'all', 'alt'",
    )  # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
    parser.add_argument(
        "--inference_only", type=str2bool, default=False, help="only do inference"
    )
    parser.add_argument(
        "--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path"
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")

    return parser


def aesthetic_loss_fn(
    aesthetic_target=None, grad_scale=0, device=None, torch_dtype=None
):
    """
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        device: torch.device, the device to run the model.
        torch_dtype: torch.dtype, the data type of the model.

    Returns:
        loss_fn: function, the loss function of the aesthetic reward function.
    """
    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None:  # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss.mean() * grad_scale, rewards.mean()

    return loss_fn


def hps_loss_fn(inference_dtype=None, device=None, hps_version="v2.0"):
    """
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of the HPS reward function.
    """
    model_name = "ViT-H-14"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        "laion2B-s32B-b79K",
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    tokenizer = get_tokenizer(model_name)

    if (
        hps_version == "v2.0"
    ):  # if there is a error, please download the model manually and set the path
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:  # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    def loss_fn(im_pix, prompts):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = (
            outputs["image_features"],
            outputs["text_features"],
        )
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return loss.mean(), scores.mean()

    return loss_fn


def aesthetic_hps_loss_fn(
    aesthetic_target=None,
    grad_scale=0,
    inference_dtype=None,
    device=None,
    hps_version="v2.0",
):
    """
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of a combination of aesthetic and HPS reward function.
    """
    # HPS
    model_name = "ViT-H-14"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        "laion2B-s32B-b79K",
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    # tokenizer = get_tokenizer(model_name)

    if (
        hps_version == "v2.0"
    ):  # if there is a error, please download the model manually and set the path
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:  # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    # Aesthetic
    scorer = AestheticScorerDiff(dtype=inference_dtype).to(
        device, dtype=inference_dtype
    )
    scorer.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):
        # Aesthetic
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

        aesthetic_rewards = scorer(im_pix)
        if aesthetic_target is None:  # default maximization
            aesthetic_loss = -1 * aesthetic_rewards
        else:
            # using L1 to keep on same scale
            aesthetic_loss = abs(aesthetic_rewards - aesthetic_target)
        aesthetic_loss = aesthetic_loss.mean() * grad_scale
        aesthetic_rewards = aesthetic_rewards.mean()

        # HPS
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(im_pix, caption)
        image_features, text_features = (
            outputs["image_features"],
            outputs["text_features"],
        )
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        hps_loss = abs(1.0 - scores)
        hps_loss = hps_loss.mean()
        hps_rewards = scores.mean()

        loss = (
            1.5 * aesthetic_loss + hps_loss
        ) / 2  # 1.5 is a hyperparameter. Set it to 1.5 because experimentally hps_loss is 1.5 times larger than aesthetic_loss
        rewards = (
            aesthetic_rewards + 15 * hps_rewards
        ) / 2  # 15 is a hyperparameter. Set it to 15 because experimentally aesthetic_rewards is 15 times larger than hps_reward
        return loss, rewards

    return loss_fn


def pick_score_loss_fn(inference_dtype=None, device=None):
    """
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.

    Returns:
        loss_fn: function, the loss function of the PickScore reward function.
    """
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(
        processor_name_or_path, torch_dtype=inference_dtype
    )
    model = (
        AutoModel.from_pretrained(
            model_pretrained_name_or_path, torch_dtype=inference_dtype
        )
        .eval()
        .to(device)
    )
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):  # im_pix_un: b,c,h,w
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)

        # reproduce the pick_score preprocessing
        im_pix = im_pix * 255  # b,c,h,w

        if im_pix.shape[2] < im_pix.shape[3]:
            height = 224
            width = (
                im_pix.shape[3] * height // im_pix.shape[2]
            )  # keep the aspect ratio, so the width is w * 224/h
        else:
            width = 224
            height = (
                im_pix.shape[2] * width // im_pix.shape[3]
            )  # keep the aspect ratio, so the height is h * 224/w

        # interpolation and antialiasing should be the same as below
        im_pix = torchvision.transforms.Resize(
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )(im_pix)
        im_pix = im_pix.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)
        # crop the center 224x224
        startx = width // 2 - (224 // 2)
        starty = height // 2 - (224 // 2)
        im_pix = im_pix[:, starty : starty + 224, startx : startx + 224, :]
        # do rescale and normalize as CLIP
        im_pix = im_pix * 0.00392156862745098  # rescale factor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        im_pix = (im_pix - mean) / std
        im_pix = im_pix.permute(0, 3, 1, 2)  # BHWC -> BCHW

        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        # embed
        image_embs = model.get_image_features(pixel_values=im_pix)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        loss = abs(1.0 - scores / 100.0)
        return loss.mean(), scores.mean()

    return loss_fn


def weather_loss_fn(
    inference_dtype=None, device=None, weather="rainy", target=None, grad_scale=0
):
    """
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        weather: str, the weather condition. It is "rainy" or "snowy" in this experiment.
        target: float, the target value of the weather score. It is 1.0 in this experiment.
        grad_scale: float, the scale of the gradient. It is 1 in this experiment.

    Returns:
        loss_fn: function, the loss function of the weather reward function.
    """
    if weather == "rainy":
        reward_model_path = "../pretrained_models/rainy_reward.pt"
    elif weather == "snowy":
        reward_model_path = "../pretrained_models/snowy_reward.pt"
    else:
        raise NotImplementedError
    scorer = WeatherScorer(dtype=inference_dtype, model_path=reward_model_path).to(
        device, dtype=inference_dtype
    )
    scorer.requires_grad_(False)
    scorer.eval()

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un + 1) / 2).clamp(0, 1)  # from [-1, 1] to [0, 1]
        rewards = scorer(im_pix)

        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)

        return loss.mean() * grad_scale, rewards.mean()

    return loss_fn


def objectDetection_loss_fn(
    inference_dtype=None,
    device=None,
    targetObject="dog.",
    model_name="grounding-dino-base",
):
    """
    This reward function is used to remove the target object from the generated video.
    We use yolo-s-tiny model to detect the target object in the generated video.

    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        targetObject: str, the object to detect. It is "dog" in this experiment.

    Returns:
        loss_fn: function, the loss function of the object detection reward function.
    """
    if model_name == "yolos-base":
        image_processor = AutoImageProcessor.from_pretrained(
            "hustvl/yolos-base", torch_dtype=inference_dtype
        )
        model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-base", torch_dtype=inference_dtype
        ).to(device)
        # check if "." in the targetObject name for yolos model
        if "." in targetObject:
            raise ValueError(
                "The targetObject name should not contain '.' for yolos-base model."
            )
    elif model_name == "yolos-tiny":
        image_processor = AutoImageProcessor.from_pretrained(
            "hustvl/yolos-tiny", torch_dtype=inference_dtype
        )
        model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny", torch_dtype=inference_dtype
        ).to(device)
        # check if "." in the targetObject name for yolos model
        if "." in targetObject:
            raise ValueError(
                "The targetObject name should not contain '.' for yolos-tiny model."
            )
    elif model_name == "grounding-dino-base":
        image_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base", torch_dtype=inference_dtype
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", torch_dtype=inference_dtype
        ).to(device)
        # check if "." in the targetObject name for grounding-dino model
        if "." not in targetObject:
            raise ValueError(
                "The targetObject name should contain '.' for grounding-dino-base model."
            )
    elif model_name == "grounding-dino-tiny":
        image_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", torch_dtype=inference_dtype
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", torch_dtype=inference_dtype
        ).to(device)
        # check if "." in the targetObject name for grounding-dino model
        if "." not in targetObject:
            raise ValueError(
                "The targetObject name should contain '.' for grounding-dino-tiny model."
            )
    else:
        raise NotImplementedError

    model.requires_grad_(False)
    model.eval()

    def loss_fn(im_pix_un):  # im_pix_un: b,c,h,w
        images = ((im_pix_un / 2) + 0.5).clamp(0.0, 1.0)

        # reproduce the yolo preprocessing
        height = 512
        width = (
            512 * images.shape[3] // images.shape[2]
        )  # keep the aspect ratio, so the width is 512 * w/h
        images = torchvision.transforms.Resize((height, width), antialias=False)(images)
        images = images.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        image_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        images = (images - image_mean) / image_std
        normalized_image = images.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Process images
        if model_name == "yolos-base" or model_name == "yolos-tiny":
            outputs = model(pixel_values=normalized_image)
        else:  # grounding-dino model
            inputs = image_processor(text=targetObject, return_tensors="pt").to(device)
            outputs = model(pixel_values=normalized_image, input_ids=inputs.input_ids)

        # Get target sizes for each image
        target_sizes = torch.tensor(
            [normalized_image[0].shape[1:]] * normalized_image.shape[0]
        ).to(device)

        # Post-process results for each image
        if model_name == "yolos-base" or model_name == "yolos-tiny":
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=target_sizes
            )
        else:  # grounding-dino model
            results = image_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=target_sizes,
            )

        sum_avg_scores = 0
        for i, result in enumerate(results):
            if model_name == "yolos-base" or model_name == "yolos-tiny":
                id = model.config.label2id[targetObject]
                # get index of targetObject's label
                index = torch.where(result["labels"] == id)
                if len(index[0]) == 0:  # index: ([],[]) so index[0] is the first list
                    sum_avg_scores = torch.sum(
                        outputs.logits - outputs.logits
                    )  # set sum_avg_scores to 0
                    continue
                scores = result["scores"][index]
            else:  # grounding-dino model
                if result["scores"].shape[0] == 0:
                    sum_avg_scores = torch.sum(
                        outputs.last_hidden_state - outputs.last_hidden_state
                    )  # set sum_avg_scores to 0
                    continue
                scores = result["scores"]
            sum_avg_scores = sum_avg_scores + (torch.sum(scores) / scores.shape[0])

        loss = sum_avg_scores / len(results)
        reward = 1 - loss

        return loss, reward

    return loss_fn


def compression_loss_fn(
    inference_dtype=None, device=None, target=None, grad_scale=0, model_path=None
):
    """
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        model_path: str, the path of the compression model.

    Returns:
        loss_fn: function, the loss function of the compression reward function.
    """
    scorer = JpegCompressionScorer(dtype=inference_dtype, model_path=model_path).to(
        device, dtype=inference_dtype
    )
    scorer.requires_grad_(False)
    scorer.eval()

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un + 1) / 2).clamp(0, 1)
        rewards = scorer(im_pix)

        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)
        return loss.mean() * grad_scale, rewards.mean()

    return loss_fn


def actpred_loss_fn(inference_dtype=None, device=None, num_frames=14, target_size=224):
    scorer = ActPredScorer(device=device, num_frames=num_frames, dtype=inference_dtype)
    scorer.requires_grad_(False)

    def preprocess_img(img):
        img = ((img / 2) + 0.5).clamp(0, 1)
        img = torchvision.transforms.Resize((target_size, target_size), antialias=True)(
            img
        )
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(img)
        return img

    def loss_fn(vid, target_action_label):
        vid = torch.cat([preprocess_img(img).unsqueeze(0) for img in vid])[None]
        return scorer.get_loss_and_score(vid, target_action_label)

    return loss_fn


def should_sample(global_step, validation_steps, is_sample_preview):
    return (
        global_step % validation_steps == 0 or global_step == 1
    ) and is_sample_preview
