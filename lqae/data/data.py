import threading
from io import BytesIO
from queue import Queue

import gcsfs
import h5py
import numpy as np
import skimage.io
import torch
import torchvision
import transformers
from ml_collections import ConfigDict
from PIL import Image
from skimage.color import gray2rgb, rgba2rgb
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


class ImageTextDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_only = False
        config.tokenize = True
        config.tokenizer = "bert-base-uncased"
        config.tokenizer_max_length = 64

        config.transform_type = "pretrain"
        config.image_size = 256

        config.image_normalization = "imagenet"
        config.custom_image_mean = ""
        config.custom_image_std = ""

        config.random_drop_text = 0.0
        config.deterministic_drop_text = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.image_normalization == "imagenet":
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == "cc12m":
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == "none":
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == "custom":
            self.image_mean = tuple(
                float(x) for x in self.config.custom_image_mean.split("-")
            )
            self.image_std = tuple(
                float(x) for x in self.config.custom_image_std.split("-")
            )
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError("Unsupported image normalization mode!")

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "finetune":
            # Use Kaiming's finetune processing
            self.transform = create_transform(
                input_size=self.config.image_size,
                is_training=True,
                color_jitter=True,
                auto_augment=None,
                interpolation="bicubic",
                re_prob=0,
                re_mode=0,
                re_count="const",
                mean=self.image_mean,
                std=self.image_std,
            )
        elif self.config.transform_type == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "resize_only":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")

        if self.config.tokenize:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.tokenizer
            )

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["jpg"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def drop_text(self, raw_index):
        deterministic_drop = (
            float(raw_index % 100) / 100.0 < self.config.deterministic_drop_text
        )
        random_drop = np.random.rand() < self.config.random_drop_text
        return deterministic_drop or random_drop

    def __getitem__(self, raw_index):
        index = self.process_index(raw_index)
        with BytesIO(self.h5_file["jpg"][index]) as fin:
            image = skimage.io.imread(fin)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)
        if self.config.image_only:
            return image

        with BytesIO(self.h5_file["caption"][index]) as fin:
            caption = fin.read().decode("utf-8")

        if not self.config.tokenize:
            return image, caption

        if len(caption) == 0 or self.drop_text(raw_index):
            tokenized_caption = np.zeros(
                self.config.tokenizer_max_length, dtype=np.int32
            )
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            return image, tokenized_caption, padding_mask

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )

        if encoded_caption["input_ids"][0].size == 0:  # Empty token
            tokenized_caption = np.zeros(
                self.config.tokenizer_max_length, dtype=np.int32
            )
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_caption = encoded_caption["input_ids"][0]
            padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)

        return image, tokenized_caption, padding_mask

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def text_length(self):
        return self.config.tokenizer_max_length


class ImageNetDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.partition = "train"
        config.image_only = True

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_normalization = "none"
        config.transform_type = "pretrain"
        config.image_size = 256

        config.autoaug = "rand-m9-mstd0.5-inc1"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.image_normalization == "imagenet":
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == "cc12m":
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == "none":
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == "custom":
            self.image_mean = tuple(
                float(x) for x in self.config.custom_image_mean.split("-")
            )
            self.image_std = tuple(
                float(x) for x in self.config.custom_image_std.split("-")
            )
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError("Unsupported image normalization mode!")

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "finetune":
            # Use Kaiming's finetune processing
            self.transform = create_transform(
                input_size=self.config.image_size,
                is_training=True,
                color_jitter=True,
                auto_augment=self.config.autoaug,
                interpolation="bicubic",
                re_prob=0,
                re_mode=0,
                re_count="const",
                mean=self.image_mean,
                std=self.image_std,
            )
        elif self.config.transform_type == "plain_finetune":
            # Use supervised training processing of ViT from "Better plain ViT baselines for ImageNet-1k" https://arxiv.org/abs/2205.01580
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "linear_prob":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["{}_jpg".format(self.config.partition)].shape[0]
            - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, index):
        index = self.process_index(index)
        with BytesIO(
            self.h5_file["{}_jpg".format(self.config.partition)][index]
        ) as fin:
            image = skimage.io.imread(fin)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)

        if self.config.image_only:
            return image

        label = self.h5_file["{}_labels".format(self.config.partition)][index]

        return image, label

    def num_classes(self):
        return 1000
