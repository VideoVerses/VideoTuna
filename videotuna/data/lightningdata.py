from abc import abstractmethod
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.video_io import init_video_worker


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    init_video_worker()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[
            worker_id * split_size : (worker_id + 1) * split_size
        ]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _default_pin_memory(pin_memory):
    if pin_memory is not None:
        return pin_memory
    return torch.cuda.is_available()


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
        img_loader=None,
        train_img=None,
        test_max_n_samples=None,
        pin_memory=None,
        persistent_workers=None,
        prefetch_factor=2,
        drop_last=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = 4 if num_workers is None else num_workers
        self.pin_memory = _default_pin_memory(pin_memory)
        if persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        else:
            self.persistent_workers = persistent_workers and self.num_workers > 0
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.drop_last = drop_last
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        # train 2 dataset
        # if img_loader is not None:
        #     img_data = instantiate_from_config(img_loader)
        #     img_data.setup()
        if train_img is not None:
            if train_img["params"]["batch_size"] == -1:
                train_img["params"]["batch_size"] = (
                    batch_size * train["params"]["video_length"]
                )
                print(
                    "Set train_img batch_size to {}".format(
                        train_img["params"]["batch_size"]
                    )
                )
            img_data = instantiate_from_config(train_img)
            self.img_loader = img_data.train_dataloader()
        else:
            self.img_loader = None
        self.wrap = wrap
        self.test_max_n_samples = test_max_n_samples
        self.collate_fn = None

    def prepare_data(self):
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config(data_cfg)
        pass

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _resolve_worker_init_fn(self, dataset):
        if isinstance(dataset, Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            return worker_init_fn
        if self.num_workers > 0:
            return worker_init_fn
        return None

    def _build_dataloader(self, dataset, shuffle=False):
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=self._resolve_worker_init_fn(dataset),
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(
            self.datasets["train"], Txt2ImgIterableBaseDataset
        )
        loader = self._build_dataloader(
            self.datasets["train"],
            shuffle=False if is_iterable_dataset else True,
        )
        if self.img_loader is not None:
            return {"loader_video": loader, "loader_img": self.img_loader}
        return loader

    def _val_dataloader(self, shuffle=False):
        return self._build_dataloader(self.datasets["validation"], shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        try:
            is_iterable_dataset = isinstance(
                self.datasets["train"], Txt2ImgIterableBaseDataset
            )
        except Exception:
            is_iterable_dataset = isinstance(
                self.datasets["test"], Txt2ImgIterableBaseDataset
            )

        shuffle = shuffle and (not is_iterable_dataset)
        if self.test_max_n_samples is not None:
            dataset = torch.utils.data.Subset(
                self.datasets["test"], list(range(self.test_max_n_samples))
            )
        else:
            dataset = self.datasets["test"]
        return self._build_dataloader(dataset, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return self._build_dataloader(self.datasets["predict"], shuffle=shuffle)
