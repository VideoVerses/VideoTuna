import json
import os

import pytorch_lightning as pl
from accelerate.logging import get_logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler

from videotuna.third_party.flux.data_backend.factory import configure_multi_databackend
from videotuna.third_party.flux.training.state_tracker import StateTracker

logger = get_logger(
    "SimpleTuner", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)


class ModelData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/disk1/xuelei/SimpleTuner_flux/SimpleTuner/VideoTuna-internal/SimpleTuner/datasets/pseudo-camera-10k/train",
        batch_size=1,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.images = []

    def init_data_backend(self):

        try:
            self.init_clear_backend_cache()
            self._send_webhook_msg(
                message="Configuring data backends... (this may take a while!)"
            )
            self._send_webhook_raw(
                structured_data={"message": "Configuring data backends."},
                message_type="init_data_backend_begin",
            )
            configure_multi_databackend(
                self.config,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
            )
            self._send_webhook_raw(
                structured_data={"message": "Completed configuring data backends."},
                message_type="init_data_backend_completed",
            )
        except Exception as e:
            import traceback

            logger.error(f"{e}, traceback: {traceback.format_exc()}")
            self._send_webhook_msg(
                message=f"Failed to load data backends: {e}",
                message_level="critical",
            )
            self._send_webhook_raw(
                structured_data={
                    "message": f"Failed to load data backends: {e}",
                    "status": "error",
                },
                message_type="fatal_error",
            )

            raise e

        self.init_validation_prompts()
        # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
        # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
        collected_data_backend_str = list(StateTracker.get_data_backends().keys())
        if self.config.push_to_hub and self.accelerator.is_main_process:
            self.hub_manager.collected_data_backend_str = collected_data_backend_str
            self.hub_manager.set_validation_prompts(
                self.validation_prompts, self.validation_shortnames
            )
            logger.debug(f"Collected validation prompts: {self.validation_prompts}")
        self._recalculate_training_steps()
        logger.info(
            f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_msg(
            message=f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_raw(
            structured_data={
                "message": f"Collected the following data backends: {collected_data_backend_str}"
            },
            message_type="init_data_backend",
        )
        self.accelerator.wait_for_everyone()

    def create_dataset(self):
        print("creating dataset...")
        self.images = [
            os.path.join(self.data_dir, image) for image in os.listdir(self.data_dir)
        ]

        print("dataset created!")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            self.train_set, _ = train_test_split(self.images, test_size=0.1)
        if stage is None or stage == "test":
            _, self.test_set = train_test_split(self.images, test_size=0.1)

    def train_dataloader(self):
        # train_sampler = DistributedSampler(self.train_set, shuffle=True)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        # test_sampler = DistributedSampler(self.test_set, shuffle=True)

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
        )
