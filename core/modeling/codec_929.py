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
from mmseg.models.builder import HEADS, MODELS, LOSSES
from mmseg.models.segmentors import EncoderDecoder

from mmcv.utils.parrots_wrapper import _BatchNorm

from .uncertainty import get_estimator
from ..utils import otsu_thresholding
from PIL import Image
from copy import deepcopy
from core.data.transform import build_tensor_transform_from_cfg
import cv2
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

from . import augment

@MODELS.register_module()
class PatchEncoderDecoder(BaseOODEncoderDecoder):
    """Variational segmentors
    """

    DEFAULT_PATCH_CFG = {
        "aspect" : (0.8, 1.2),
        "area" : (0.03, 0.06),
        "patch_index" : 1,
        "noise_ratio" : 0.5,
        "max_patch_num" : 3
    }

    def __init__(self,
                 patch_head,
                 patch_cfg = None,
                 logit_loss_cfg = None,
                 freeze_segment = True,
                 patch_queue_size = 30,
                 transforms = [],
                 **kwargs):
        super(PatchEncoderDecoder, self).__init__(**kwargs)
        self.patch_head = HEADS.build(patch_head)
        self.freeze_semgnet = freeze_segment
        self.patch_cfg = patch_cfg if patch_cfg is not None else dict()

        for required_key, default_val in self.DEFAULT_PATCH_CFG.items():
            if required_key not in self.patch_cfg:
                self.patch_cfg[required_key] = default_val

        self.transforms = build_tensor_transform_from_cfg(transforms)

        self.patch_queue = deque(maxlen=patch_queue_size)
        self.random_list = [i for i in range(patch_queue_size)]
        self.merge_weight = self.test_cfg.get("merge_weight", 2.0)

        if logit_loss_cfg is not None:
            self.loss = LOSSES.build(logit_loss_cfg)
        else:
            self.loss = None
        self.count = 0
        self.segment_part = ['backbone', 'decode_head', 'auxiliary_head']
        self.randout = augment.RandomRoadObject(rcp=0.5)
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

    def add_patch(self, img, gt_semantic_seg):
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
            patch_gt = gt_semantic_seg[i, :, y:y+hc, x:x+wc].clone()
            self.patch_queue.append((patch, patch_gt))


            aspect = np.random.uniform(*self.patch_cfg["aspect"])
            area = np.random.uniform(*self.patch_cfg["area"]) * (h * w)

            wc = int(math.sqrt(aspect * area))
            hc = int(math.sqrt(area / aspect))
            wc = min(w, wc)
            hc = min(h, hc)
            x = random.randint(0, w - wc)
            y = random.randint(0, h - hc)
            patch = img[i, :, y:y+hc, x:x+wc].clone().detach()
            patch_gt = gt_semantic_seg[i, :, y:y+hc, x:x+wc].clone()
            self.patch_queue.append((patch, patch_gt))



    def render_patch(self, img, gt_semantic_seg):

        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        #ood_seg[gt_semantic_seg==self.decode_head.ignore_index] = self.patch_cfg["patch_index"]
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            #cv2.imwrite("./results/saveimg/"+str(i)+'ori.jpg', img[i].cpu().numpy()*255)
            #savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB')
            #savet.save("./results/saveimg/"+str(self.count)+'ori.jpg')
            num_patch = 10#random.randint(1, self.patch_cfg["max_patch_num"])
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch = self.patch_queue[patch[0]][0].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
       
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
       
                x = random.randint(0, w - wc)
                y = random.randint(0, h - hc)
       
                img[i, :, y:y+hc, x:x+wc] = self.transforms(patch)
                gt_semantic_seg[i, :, y:y+hc, x:x+wc] = self.decode_head.ignore_index
                ood_seg[i, :, y:y+hc, x:x+wc] = self.patch_cfg["patch_index"]#patch_area = ood_seg[i, :, y:y+hc, x:x+wc]
                #patch_area[patch_gt==self.decode_head.ignore_index] = self.patch_cfg["patch_index"]
            #savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB') 
            #savet.save("./results/saveimg/"+str(self.count)+'patch.jpg')
            #self.count+=1
            #cv2.imwrite("./results/saveimg/"+str(i)+'patch.jpg', img[i].cpu().numpy()*255)
        return img, gt_semantic_seg, ood_seg

    def render_patchv2(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))] 
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10#random.randint(1, self.patch_cfg["max_patch_num"])
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)

                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i] = self.randout(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index) 
            #savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB') 
            #savet.save("./results/saveimg_hull/"+str(self.count)+'patch.jpg')
            #self.count+=1
        return img, gt_semantic_seg, ood_seg





    def render_patchv3(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i] = self.randout.forwardv2(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index)
            #savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB')
            #savet.save("./results/saveimg_rand/"+str(self.count)+'patch.jpg')
            #self.count+=1
        return img, gt_semantic_seg, ood_seg

    def render_patchv4(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i] = self.randout.forwardv3(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index)
            savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB') 
            savet.save("./results/saveimg_rand/"+str(self.count)+'patch.jpg')
            self.count+=1
        return img, gt_semantic_seg, ood_seg



    def render_patchv5(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i] = self.randout.forwardv4(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index)

            savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB')
            savet.save("./results/saveimg_rand/"+str(self.count)+'patch.jpg')
            self.count+=1
        return img, gt_semantic_seg, ood_seg

    def render_patchv6(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                if np.random.uniform() < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i] = self.randout.forwardv4(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index)

        return img, gt_semantic_seg, ood_seg





    def _ood_head_forward_train(self, x, img_metas, ood_seg, ood_score=None):

        losses = dict()
        loss_aux = self.patch_head.forward_train(
            x, img_metas, ood_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'ood'))

        if ood_score is not None and self.loss is not None:
            ood_out_logit = self.patch_head.forward_test(x, img_metas, self.test_cfg)
            id_logit, ood_logit = torch.split(ood_out_logit, 1, dim=1)
            ood_logit = resize(
                input=ood_logit,
                size=ood_seg.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners)

            ood_logit = ood_logit + ood_score
            loss_logit = self.loss(ood_logit, ood_seg)
            losses.update({"ood_logit_loss": loss_logit})

        return losses

    def _refine_ood_gt(self, ood_score, ood_gt):

        batchsize = ood_score.size(0)
        h, w = ood_score.size(2), ood_score.size(3)

        ood_score = ood_score.view(batchsize, -1)
        ood_gt = ood_gt.view(batchsize, -1)
        refined_gt = ood_gt.clone()

        mask = (ood_gt == self.patch_cfg["patch_index"])

        thresh = otsu_thresholding(ood_score, nbins=25, weight=mask.float())

        refined_gt[mask & (ood_score <= thresh)] = self.decode_head.ignore_index
        refined_gt = refined_gt.view(batchsize, -1, h, w)

        return refined_gt


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
        self.add_patch(img, gt_semantic_seg)
        img, gt_semantic_seg, ood_seg = self.render_patchv5(img, gt_semantic_seg)
        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)


        losses.update(loss_decode)
        ood_inp = [v.detach() for v in x]

        # refine ood seg
        with torch.no_grad():
            seg_out = self._decode_head_forward_test(x, img_metas)
            seg_out = resize(
                input=seg_out,
                size=ood_seg.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners
            )
            ood_score = super(PatchEncoderDecoder, self)._get_ood_segmentation_score(x, ood_inp, seg_out, img_metas)
            ood_seg = self._refine_ood_gt(ood_score, ood_seg)

        loss_ood = self._ood_head_forward_train(ood_inp, img_metas, ood_seg, ood_score)
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
