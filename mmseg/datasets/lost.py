import os
import random
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

#from config.config import config
from .base_dataset import BaseDataset
from .img_utils import generate_random_crop_pos, random_crop_pad_to_shape


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std
    return img


def random_mirror(img, gt=None):
    if random.random() >= 0.2:
        img = cv2.flip(img, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, gt


def random_scale(img, gt=None, scales=None):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale


def SemanticEdgeDetector(gt):
    id255 = np.where(gt == 255)
    no255_gt = np.array(gt)
    no255_gt[id255] = 0
    cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
    edge_radius = 7
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_radius, edge_radius))
    cgt = cv2.dilate(cgt, edge_kernel)
    # print(cgt.max(), cgt.min())
    cgt[cgt > 0] = 1
    return cgt


class TrainPre(object):
    def __init__(self, img_mean, img_std,
                 augment=True):
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = augment

    def __call__(self, img, gt=None):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255
        if not self.augment:
            return normalize(img, self.img_mean, self.img_std), None, None, None

        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)
        if gt is not None:
            cgt = SemanticEdgeDetector(gt)
        else:
            cgt = None

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            p_cgt, _ = random_crop_pad_to_shape(cgt, crop_pos, crop_size, 255)
        else:
            p_gt = None
            p_cgt = None

        p_img = p_img.transpose(2, 0, 1)
        extra_dict = {}

        return p_img, p_gt, p_cgt, extra_dict


class ValPre(object):
    def __call__(self, img, gt):
        # gt = gt - 1
        extra_dict = {}
        return img, gt, None, extra_dict


def get_mix_loader(engine, collate_fn=None, augment=True, cs_root=None,
                   coco_root=None):
    train_preprocess = TrainPre(config.image_mean, config.image_std, augment=augment)

    train_dataset = CityscapesCocoMix(split='train', preprocess=train_preprocess, cs_root=cs_root,
                                      coco_root=coco_root, subsampling_factor=0.1)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    void_ind = train_dataset.void_ind

    return train_loader, train_sampler, void_ind


def get_test_loader(dataset, eval_source, config_file):
    data_setting = {'img_root': config_file.img_root_folder,
                    'gt_root': config_file.gt_root_folder,
                    'train_source': config_file.train_source,
                    'eval_source': eval_source}

    val_preprocess = ValPre()

    test_dataset = dataset(data_setting, 'val', val_preprocess)

    return test_dataset

@DATASETS.register_module()
class LostAndFound(Dataset):
    LostAndFoundClass = namedtuple('LostAndFoundClass', ['name', 'id', 'train_id', 'category_name',
                                                         'category_id', 'color'])

    labels = [
        LostAndFoundClass('unlabeled', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('ego vehicle', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('rectification border', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('out of roi', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('background', 0, 255, 'Counter hypotheses', 1, (0, 0, 0)),
        LostAndFoundClass('free', 1, 1, 'Counter hypotheses', 1, (128, 64, 128)),
        LostAndFoundClass('Crate (black)', 2, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - stacked)', 3, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - upright)', 4, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray)', 5, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - stacked) ', 6, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - upright)', 7, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Bumper', 8, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 1', 9, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue)', 10, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue - small)', 11, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green)', 12, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green - small)', 13, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Exhaust Pipe', 14, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Headlight', 15, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Euro Pallet', 16, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon', 17, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (large)', 18, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (white)', 19, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Rearview mirror', 20, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Tire', 21, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Ball', 22, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bicycle', 23, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (black)', 24, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (white)', 25, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Kid dummy', 26, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby car (gray)', 27, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (red)', 28, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (yellow)', 29, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 2', 30, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Marker Pole (lying)', 31, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Plastic bag (bloated)', 32, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Post (red - lying)', 33, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Post Stand', 34, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Styrofoam', 35, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Timber (small)', 36, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Timber (squared)', 37, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wheel Cap', 38, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wood (thin)', 39, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Kid (walking)', 40, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (on a bobby car)', 41, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (small bobby)', 42, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (crawling)', 43, 2, 'Humans', 6, (0, 0, 142)),
    ]

    train_id_in = 1
    train_id_out = 2
    num_eval_classes = 19

    def __init__(self, split='test', root="./datasets/", transform=None):
        assert os.path.exists(root), "lost&found valid not exists"
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split  # ['test', 'train']
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        self.annotations = []  # list of all ground truth LabelIds images

        for root, _, filenames in os.walk(os.path.join(root, 'leftImg8bit', self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    city = '_'.join(filename.split('_')[:-3])
                    self.images.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    target_root = os.path.join(self.root, 'gtCoarse', self.split)
                    self.targets.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelTrainIds.png'))
                    self.annotations.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelIds.png'))

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

