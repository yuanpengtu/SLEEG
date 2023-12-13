"""
Random conversion used for tensors
"""

import torch
import numpy as np
import random

import pycocotools.mask as maskutil
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from functools import partial
from mmcv import Config
from typing import Union, Tuple
from copy import deepcopy

MAJOR_VERSION, MINOR_VERSION = (int(v) for v in torch.__version__.split(".")[:2])
TENSOR_TRANS_AVAILABLE = (MAJOR_VERSION >= 1) and (MINOR_VERSION >= 7)

class TensorTransform:

    def __init__(self, transform = None):
        self.transform = transform

    def __call__(self, tensor):

        if self.transform is None or random.randint(0, 1):
            return tensor
        else:
            return self.transform(tensor)


def build_tensor_transform_from_cfg(cfgs):

    assert isinstance(cfgs, (list, tuple))

    transforms = []
    transform_list = cfgs

    if TENSOR_TRANS_AVAILABLE:
        for cfg in transform_list:
            arg = deepcopy(cfg)
            transform_name = arg.pop("type")
            try:
                transform = getattr(T, transform_name)(**arg)
                transforms.append(transform)
            except AttributeError:
                continue
    else:
        for cfg in transform_list:
            arg = deepcopy(cfg)
            transform_name = arg.pop("type")
            transform_func = getattr(TF, transform_name.lower())
            transform = partial(transform_func, **arg)
            transforms.append(transform)

    transform = T.RandomApply(transforms)

    return TensorTransform(transform)