import os
import sys
import argparse
from pathlib import Path

import numpy as np
from datetime import datetime

import wandb
import torch
from easydict import EasyDict
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

user = os.getlogin()
sys.path.append(f"/home/{user}/workspace")
from shared_utils.io import yaml_read  # noqa: E402
from iwildcam19_project.models import build_model, AnimalClassifier
from iwildcam19_project.data_utils import IWildCam19DataModule


def load_from_wandb(wandb_id: str) -> (pl.LightningModule, torch.utils.data.DataLoader):
    wandb_path = f"bio-vision-lab/iwildcam19/runs/{wandb_id}"
    api = wandb.Api()
    run = api.run(wandb_path)
    cfg = EasyDict(run.config)
    run_name = run.name  # .replace(' ', '_')
    ckpts_dir = Path(run.config['ckpt_path'])
    latest_ckpt = sorted([x for x in (ckpts_dir / run_name).glob('*')])[-1]
    net = build_model(name=cfg.model.name, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    model = AnimalClassifier.load_from_checkpoint(latest_ckpt, net=net, criterion=nn.CrossEntropyLoss(),
                                                  optimizer_params=cfg.optimizer,scheduler_params=cfg.scheduler)
    dm = IWildCam19DataModule(cfg.data)
    dl = dm.predict_dataloader()
    return model, dl
