import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.instance_data_root = (
            Path(instance_data_root) if instance_data_root is not None else None
        )
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.image_to_video = image_to_video

        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = (
                self._load_dataset_from_hub()
            )
        else:
            self.instance_prompts, self.instance_video_paths = (
                self._load_dataset_from_local_path()
            )

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.instance_videos = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        if self.image_to_video:
            image = self.instance_videos[index][:1].clone()
            return {
                "instance_prompt": self.id_token + self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
                "instance_image": image,
            }
        else:
            return {
                "instance_prompt": self.id_token + self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
            }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [
            Path(self.instance_data_root, filepath)
            for filepath in dataset["train"][video_column]
        ]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [
                line.strip() for line in file.readlines() if len(line.strip()) > 0
            ]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip())
                for line in file.readlines()
                if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(
                uri=filename.as_posix(), width=self.width, height=self.height
            )
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(
                    range(
                        start_frame,
                        end_frame,
                        (end_frame - start_frame) // self.max_num_frames,
                    )
                )
                frames = video_reader.get_batch(indices)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            frames = frames.float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
        return videos
