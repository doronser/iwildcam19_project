import os
import sys
import torch
import torchvision
import torch.nn as nn
from typing import Literal
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from torchmetrics import Precision, Recall


from shared_utils.models import BaseModel


class AnimalClassifier(BaseModel):
    def __init__(self, net: nn.Module, criterion, optimizer_params=None, scheduler_params=None):
        super().__init__(net, criterion, optimizer_params, scheduler_params)
        self.f1 = MulticlassF1Score(num_classes=23, average='macro')
        self.my_precision = Precision(num_classes=23, task='multiclass', average='macro', top_k=1)
        self.my_recall = Recall(num_classes=23, task='multiclass', average='macro', top_k=1)

    def infer_batch(self, batch):
        img, labels = batch
        scores = self.net(img)
        return scores, labels

    def training_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss = self.criterion(scores, labels)
        prc = self.my_precision(scores, labels)
        rcl = self.my_recall(scores, labels)
        f1 = self.f1(scores, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(dict(train_precision=prc, train_recall=rcl, train_f1=f1))
        return loss

    def validation_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss = self.criterion(scores, labels)
        prc = self.my_precision(scores, labels)
        rcl = self.my_recall(scores, labels)
        f1 = self.f1(scores, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log_dict(dict(val_precision=prc, val_recall=rcl, val_f1=f1), on_epoch=True, on_step=False)
        return loss


def build_model(name: Literal['convnext', 'mobilenet_v3', 'efficientnet_v2'], pretrained=False, num_classes=23,
                ckpt=None):
    assert name in ['convnext', 'mobilenet_v3', 'efficientnet_v2'], "invalid model name!"

    if name == 'convnext':
        model_func = torchvision.models.convnext_tiny
    elif name == 'mobilenet_v3':
        model_func = torchvision.models.convnext_tiny
    else:  # efficientnet_v2
        model_func = torchvision.models.efficientnet_v2_s

    if pretrained:
        model = model_func(weights='DEFAULT')
    else:
        model = model_func()

    # replace last layer with to get the correct output size
    last_layer = model.classifier[-1]
    model.classifier[-1] = torch.nn.Linear(in_features=last_layer.in_features, out_features=num_classes)
    return model

