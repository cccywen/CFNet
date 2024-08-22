import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from torchstat import stat
from thop import profile
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from geoseg.models.BANet import BANet
from geoseg.models.MANFPNUnet5 import PoolUNet
from geoseg.models.ABCNet import ABCNet




import torch

if __name__ == '__main__':
    # Model
    print('==> Building model..')
    modelPath = "/home/caoyiwen/slns/CFNet/model_weights/vaihingen/tm2-512-e90/tm2-512-e90.ckpt"
    # model = torch.load(modelPath, map_location="cpu")
    model = ABCNet(n_classes=6)

    dummy_input = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
