# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np

from collections import deque
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models.builder import HEADS, MODELS
from mmseg.models.segmentors import EncoderDecoder

from mmcv.utils.parrots_wrapper import _BatchNorm

from .uncertainty import get_estimator
from copy import deepcopy

from torchvision.transforms import Resize

@MODELS.register_module()
class BaseOODEncoderDecoder(EncoderDecoder):

    def __init__(
            self,
            uncertain_cfg=dict(type="maxlogit", probability=False),
            **kwargs
    ):
        super(BaseOODEncoderDecoder, self).__init__(**kwargs)
        self.uncertain_cfg = uncertain_cfg

    def _get_ood_segmentation_score(
            self,
            img,
            feats,
            seg_logit,
            img_metas
    ):

        ood_score = seg_logit.new_zeros((seg_logit.size(0), 1) + seg_logit.size()[2:])
        if self.uncertain_cfg.get("type", None) is not None:

            uncertain_cfg = deepcopy(self.uncertain_cfg)
            method_name = uncertain_cfg.pop("type")
            func = get_estimator(method_name)
            if func is None:
                raise NameError(f"Uncertainty estimation method {method_name} not found")

            ood_score = func(seg_logit, **uncertain_cfg)
            ood_score = ood_score.unsqueeze(1)

        return ood_score

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_logit, ood_logit = seg_logit[:, :self.num_classes], seg_logit[:, -1]
        seg_pred = seg_logit.argmax(dim=1)

        seg_pred = seg_pred.cpu().numpy()
        ood_logit = ood_logit.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        ood_pred = list(ood_logit)

        if self.test_cfg.get("with_ood", False):
            return seg_pred, ood_pred
        else:
            return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)
        seg_logit, ood_logit = seg_logit[:, :self.num_classes], seg_logit[:, -1]
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        ood_logit = ood_logit.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        ood_pred = list(ood_logit)

        if self.test_cfg.get("with_ood", False):
            return seg_pred, ood_pred
        else:
            return seg_pred

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        seg_out = out
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        out_ood = self._get_ood_segmentation_score(
            img, x, seg_out, img_metas
        )

        out_ood = resize(
            input=out_ood,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        out = torch.cat([out, out_ood], dim=1)

        return out

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        def flip_output(out):

            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    out = out.flip(dims=(3,))
                elif flip_direction == 'vertical':
                    out = out.flip(dims=(2,))

            return out

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        output = F.softmax(seg_logit[:, :self.num_classes], dim=1)
        output = flip_output(output)

        ood_logit = seg_logit[:, -1:]
        ood_logit = flip_output(ood_logit)
        output = torch.cat([output, ood_logit], dim=1)

        return output


@MODELS.register_module()
class PatchEncoderDecoder(BaseOODEncoderDecoder):
    """Variational segmentors
    """

    DEFAULT_PATCH_CFG = {
        "aspect" : (0.8, 1.2),
        "area" : (0.05, 0.08),
        "patch_index" : 1,
        "noise_ratio" : 0.5
    }

    def __init__(self,
                 patch_head,
                 patch_cfg = None,
                 freeze_segment = True,
                 patch_queue_size = 10,
                 **kwargs):
        super(PatchEncoderDecoder, self).__init__(**kwargs)
        self.patch_head = HEADS.build(patch_head)
        self.freeze_semgnet = freeze_segment
        self.patch_cfg = patch_cfg if patch_cfg is not None else dict()
        
        for required_key, default_val in self.DEFAULT_PATCH_CFG.items():
            if required_key not in self.patch_cfg:
                self.patch_cfg[required_key] = default_val

        

        self.patch_queue = deque(maxlen=patch_queue_size)
        self.merge_weight = self.test_cfg.get("merge_weight", 2.0)

        self.segment_part = ['backbone', 'decode_head', 'auxiliary_head']
    def _freeze_segment_net(self):

        for name in self.segment_part:
            mod = getattr(self, name)
            if isinstance(mod, nn.Module):
                for mod_name, param in mod.named_parameters():
                    param.requires_grad = False

                for m in mod.modules():
                    if isinstance(m, _BatchNorm):
                        m.eval()

    def train(self, mode=True):
        super(PatchEncoderDecoder, self).train(mode)
        if mode and self.freeze_semgnet:
            self._freeze_segment_net()

    def add_patch(self, img):

        batchsize = img.size(0)
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)

            aspect = np.random.uniform(*self.patch_cfg["aspect"])
            area = np.random.uniform(*self.patch_cfg["area"]) * (h * w)

            wc = int(math.sqrt(aspect * area))
            hc = int(math.sqrt(area / aspect))
            wc = min(w, wc)
            hc = min(h, hc)

            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)

            patch = img[i, :, y:y+hc, x:x+wc].clone().detach()
            self.patch_queue.append(patch)
    def add_patch_new(self, img):

        batchsize = img.size(0)
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)

            aspect = np.random.uniform(*self.patch_cfg["aspect"])
            area = np.random.uniform(*self.patch_cfg["area"]) * (h * w)
            wc = int(math.sqrt(aspect * area))
            hc = int(math.sqrt(area / aspect))
            wc = min(w, wc)
            hc = min(h, hc)
            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)
            patch = img[i, :, y:y+hc, x:x+wc].clone().detach()
            self.patch_queue.append(patch)

            aspect = np.random.uniform(*self.patch_cfg["aspect"])
            area = np.random.uniform(*self.patch_cfg["area"]) * (h * w)
            wc = int(math.sqrt(aspect * area))
            hc = int(math.sqrt(area / aspect))
            wc = min(w, wc)
            hc = min(h, hc)
            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)
            patch = img[i, :, y:y+hc, x:x+wc].clone().detach()
            self.patch_queue.append(patch)


    def render_patch(self, img, gt_semantic_seg):

        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)

            patch = random.choice(self.patch_queue).clone()
            if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                patch[...] = torch.randn(patch.size()).to(img.device)

            hc, wc = patch.size(1), patch.size(2)
            hc, wc = min(hc, h), min(wc, w)
            patch = patch[:, :hc, :wc]

            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)
            
            lam = np.random.beta(1.0, 1.0)
           
            img[i, :, y:y+hc, x:x+wc] = lam * patch + (1-lam) * img[i, :, y:y+hc, x:x+wc]
            gt_semantic_seg[i, :, y:y+hc, x:x+wc] = self.decode_head.ignore_index
            ood_seg[i, :, y:y+hc, x:x+wc] = self.patch_cfg["patch_index"]

            return img, gt_semantic_seg, ood_seg

    def render_patch_new(self, img, gt_semantic_seg):

        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            patch = random.choice(self.patch_queue).clone()
            patch2 = random.choice(self.patch_queue).clone()
            if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                patch[...] = torch.randn(patch.size()).to(img.device)
                patch2[...] = torch.randn(patch2.size()).to(img.device)
            hc, wc = patch.size(1), patch.size(2)
            hc, wc = min(hc, h), min(wc, w)
            patch = patch[:, :hc, :wc]
            
            torch_resize = Resize([hc, wc])
            patch2 = torch_resize(patch2)

            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)

            lam = np.random.beta(1.0, 1.0)
             
            img[i, :, y:y+hc, x:x+wc] = lam * patch + (1-lam) * patch2#img[i, :, y:y+hc, x:x+wc]
            gt_semantic_seg[i, :, y:y+hc, x:x+wc] = self.decode_head.ignore_index
            ood_seg[i, :, y:y+hc, x:x+wc] = self.patch_cfg["patch_index"]

            return img, gt_semantic_seg, ood_seg
    def render_patch_newv2(self, img, gt_semantic_seg):

        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            pathnum = 3#random.randint(0,2)
            for j in range(pathnum):
                patch = random.choice(self.patch_queue).clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]

                x = random.randint(0, w - wc)
                y = random.randint(0, h - hc)

                lam = np.random.beta(1.0, 1.0)
                img[i, :, y:y+hc, x:x+wc] = patch#lam * patch + (1-lam) * patch2#img[i, :, y:y+hc, x:x+wc]
                gt_semantic_seg[i, :, y:y+hc, x:x+wc] = self.decode_head.ignore_index
                ood_seg[i, :, y:y+hc, x:x+wc] = self.patch_cfg["patch_index"]

        return img, gt_semantic_seg, ood_seg



    def _ood_head_forward_train(self, x, img_metas, ood_seg):

        losses = dict()
        loss_aux = self.patch_head.forward_train(
            x, img_metas, ood_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'ood'))

        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.add_patch_new(img)
        img, gt_semantic_seg, ood_seg = self.render_patch_newv2(img, gt_semantic_seg)
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        ood_inp = [v.detach() for v in x]
        loss_ood = self._ood_head_forward_train(ood_inp, img_metas, ood_seg)
        losses.update(loss_ood)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def _get_ood_segmentation_score(
            self,
            img,
            feats,
            seg_logit,
            img_metas
    ):
        logit_ood = super(PatchEncoderDecoder, self)._get_ood_segmentation_score(img, feats, seg_logit, img_metas)

        out_ood = self.patch_head.forward_test(feats, img_metas, self.test_cfg)
        out_ood = out_ood[:, 1:2]

        out_ood = out_ood * self.merge_weight + logit_ood

        return out_ood
