import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import wandb
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight

user = os.getlogin()
sys.path.append(f"/home/{user}/workspace")
from shared_utils.io import yaml_read  # noqa: E402
from iwildcam19_project.models import build_model, AnimalClassifier
from iwildcam19_project.data_utils import IWildCam19DataModule, IWildCam19Dataset


# for reproducibility
torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, help="path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    assert os.path.exists(args.cfg), f"cfg file {args.cfg} does not exist!"
    cfg = yaml_read(args.cfg, easy_dict=True)

    data_module = IWildCam19DataModule(cfg.data)

    net = build_model(name=cfg.model.name, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    if cfg.loss.weighted:
        train_df = pd.read_csv(os.path.join(cfg.data.dir,'train.csv'))
        y = train_df.category_id.values
        cls_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        weights = torch.ones(23)
        for idx, weight in zip(np.unique(y), cls_weights):
            weights[idx] = weight
        loss = nn.CrossEntropyLoss(weight=weights)
    else:
        loss = nn.CrossEntropyLoss()
    model = AnimalClassifier(net=net, criterion=loss, optimizer_params=cfg.optimizer,
                             scheduler_params=cfg.scheduler)

    if args.debug:
        trainer = pl.Trainer(fast_dev_run=True)
    else:
        # define pl loggers and callbacks
        time_suffix = datetime.now().strftime(r"%y%m%d_%H%M%S")
        wandb_logger = WandbLogger(project="iwildcam19", name=f"{cfg.name}_{time_suffix}")
        wandb.config.update(cfg)
        ckpt_clbk = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.ckpt_path, f"{cfg.name}_{time_suffix}"),
                                                 filename=cfg.name + '_epoch{epoch:02d}', auto_insert_metric_name=False,
                                                 save_top_k=-1, every_n_epochs=10)
        lr_clbk = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[cfg.trainer.gpu],
            max_epochs=cfg.trainer.epochs,
            callbacks=[ckpt_clbk, lr_clbk],
            logger=wandb_logger
        )
    trainer.fit(model=model, datamodule=data_module)
