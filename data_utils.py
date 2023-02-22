import os
import cv2
import numpy as np
import pandas as pd
from typing import Union, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader


class IWildCam19Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform: Union[A.Compose, None] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.ds_type = 'train' if train else 'test'
        self.df = pd.read_csv(os.path.join(data_dir, f"{self.ds_type}.csv"))

        # map between class-index and class-name
        self.idx2str = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal',
                        6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote',
                        12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',
                        18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}

        self.str2idx = {'empty': 0, 'deer': 1, 'moose': 2, 'squirrel': 3, 'rodent': 4, 'small_mammal': 5,
                        'elk': 6, 'pronghorn_antelope': 7, 'rabbit': 8, 'bighorn_sheep': 9, 'fox': 10, 'coyote': 11,
                        'black_bear': 12, 'raccoon': 13, 'skunk': 14, 'wolf': 15, 'bobcat': 16, 'cat': 17,
                        'dog': 18, 'opossum': 19, 'bison': 20, 'mountain_goat': 21, 'mountain_lion': 22}

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        if self.ds_type == 'train':
            label = df_row.category_id
        else:
            label = -1
        img_path = os.path.join(self.data_dir, self.ds_type, df_row.file_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  #.astype(np.float32)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


class IWildCam19DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: EasyDict):
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.val_split_size = data_cfg.val_split_size
        self.data_dir = data_cfg.dir
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers
        self.transform = self.parse_cfg_transform(data_cfg)

    def setup(self, stage: Optional[str] = None):
        base_ds = IWildCam19Dataset(data_dir=self.data_dir, train=True, transform=self.transform)
        num_val = int(self.val_split_size * len(base_ds))
        num_train = len(base_ds) - num_val
        self.train_set, self.val_set = torch.utils.data.random_split(base_ds, lengths=[num_train, num_val])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        test_ds = IWildCam19Dataset(data_dir=self.data_dir, train=False, transform=self.transform)
        return DataLoader(dataset=test_ds, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    @staticmethod
    def parse_cfg_transform(yml_cfg: Union[dict, EasyDict]) -> Union[A.Compose, None]:
        """Parse YAML config from plain text to albumentations transform

        :param yml_cfg: YAML config dictionary
        :return: albumentations composed transform object
        """
        augs = yml_cfg.get('augmentations', None)

        aug_list = [A.Resize(width=32, height=32)]  #
        if augs is not None:
            for aug, params in augs.items():
                aug_func = getattr(A, aug)
                if params is not None:
                    aug_list.append(aug_func(**params))
                else:
                    aug_list.append(aug_func(p=0.5))
        aug_list.append(A.ToFloat(max_value=255.0))
        aug_list.append(ToTensorV2())
        augmentations = A.Compose(aug_list)

        return augmentations
