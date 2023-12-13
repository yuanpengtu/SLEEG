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
from mmseg.models.segmentors import EncoderDecoder, CascadeEncoderDecoder

from mmcv.utils.parrots_wrapper import _BatchNorm

from .uncertainty import get_estimator
from ..utils import otsu_thresholding
from PIL import Image
from copy import deepcopy
from core.data.transform import build_tensor_transform_from_cfg
import cv2
#from .postprocess import BoundarySuppressionWithSmoothing
import torch.autograd as ag

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
        self.patch_queue_whole = deque(maxlen=patch_queue_size * 2)
        self.mask_queue_whole = deque(maxlen=patch_queue_size * 2)


        self.random_list = [i for i in range(patch_queue_size)]
        self.merge_weight = self.test_cfg.get("merge_weight", 2.0)

        if logit_loss_cfg is not None:
            self.loss = LOSSES.build(logit_loss_cfg)
        else:
            self.loss = None
        self.count, self.times = 0, 0
        self.segment_part = ['backbone', 'decode_head', 'auxiliary_head']
        self.randout = augment.RandomRoadObject(rcp=0.5)
        #self.multi_scale = BoundarySuppressionWithSmoothing()
        self.ignore_index = 255

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
                gt_semantic_seg[i, :, y:y+hc, x:x+wc] = self.ignore_index
                ood_seg[i, :, y:y+hc, x:x+wc] = self.patch_cfg["patch_index"]#patch_area = ood_seg[i, :, y:y+hc, x:x+wc]
                #patch_area[patch_gt==self.decode_head.ignore_index] = self.patch_cfg["patch_index"]
            #savet = Image.fromarray((255 * img[i].transpose(0,2).transpose(0,1).cpu().numpy()).astype('uint8')).convert('RGB') 
            #savet.save("./results/saveimg/"+str(self.count)+'patch.jpg')
            #self.count+=1
            #cv2.imwrite("./results/saveimg/"+str(i)+'patch.jpg', img[i].cpu().numpy()*255)
        return img, gt_semantic_seg, ood_seg

    def render_patchv4(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        corrdinate = []
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            corrdinate_per = []
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                prob_v = np.random.uniform()
                if prob_v < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i], corrdinate_i = self.randout(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.ignore_index)
                corrdinate_per.append(corrdinate_i)
            corrdinate.append(corrdinate_per)
        return img, gt_semantic_seg, ood_seg, num_patch, corrdinate

    def render_patchv5(self, img, gt_semantic_seg):
        ood_seg = gt_semantic_seg.new_zeros(gt_semantic_seg.size())
        batchsize = img.size(0)
        self.random_list = [i for i in range(len(self.patch_queue))]
        corrdinate = []
        
        for i in range(batchsize):
            h, w = img.size(2), img.size(3)
            num_patch = 10
            row, col = [], []
 
            corrdinate_per = []
            for _ in range(num_patch):
                patch = random.sample(self.random_list,1)
                patch, patch_gt = self.patch_queue[patch[0]][0].clone(), self.patch_queue[patch[0]][1].clone()
                prob_v = np.random.uniform()
                if prob_v < self.patch_cfg["noise_ratio"]:
                    patch[...] = torch.randn(patch.size()).to(img.device)
                hc, wc = patch.size(1), patch.size(2)
                hc, wc = min(hc, h), min(wc, w)
                patch = patch[:, :hc, :wc]
                hulldict = dict()
                hulldict['image'] = img[i].cpu()
                hulldict['label'] = gt_semantic_seg[i].cpu()
                img[i], gt_semantic_seg[i], ood_seg[i], corrdinate_i = self.randout.forwardv1(hulldict, gt_semantic_seg[i], ood_seg[i], patch, self.patch_cfg["patch_index"],self.decode_head.ignore_index)
                corrdinate_per.append(corrdinate_i)
            corrdinate.append(corrdinate_per)
        return img, gt_semantic_seg, ood_seg, num_patch, corrdinate




    def _ood_head_forward_train(self, x, img_metas, ood_seg, ood_score=None):

        losses = dict()
        loss_aux = self.patch_head.forward_train(
            x, img_metas, ood_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'ood'))

        if ood_score is not None and self.loss is not None:
            ood_out_logit = self.patch_head.forward_test(x, img_metas, self.test_cfg)
            print("ood:",ood_out_logit.shape, ood_score.shape)
            print(img_metas)
            id_logit, ood_logit = torch.split(ood_out_logit, 1, dim=1)
            ood_logit = resize(
                input=ood_logit,
                size=ood_seg.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners)
            print("ood2:", ood_logit.shape)
            ood_logit = ood_logit + ood_score
            loss_logit = self.loss(ood_logit, ood_seg)
            losses.update({"ood_logit_loss": loss_logit})
            return losses, loss_logit
        return losses

    def _refine_ood_gt(self, ood_score, ood_gt):

        batchsize = ood_score.size(0)
        h, w = ood_score.size(2), ood_score.size(3)

        ood_score = ood_score.view(batchsize, -1)
        ood_gt = ood_gt.view(batchsize, -1)
        refined_gt = ood_gt.clone()

        mask = (ood_gt == self.patch_cfg["patch_index"])

        thresh = otsu_thresholding(ood_score, nbins=25, weight=mask.float())

        refined_gt[mask & (ood_score <= thresh)] = self.ignore_index#decode_head.ignore_index
        refined_gt = refined_gt.view(batchsize, -1, h, w)

        return refined_gt



    def forward_trainv1(self, img, img_metas, gt_semantic_seg):
        mask_adv = nn.Parameter(torch.zeros(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).float(), requires_grad=True).cuda()
        img = img + mask_adv
        
        self.add_patch(img, gt_semantic_seg)
        x = self.extract_feat(img)
        ood_inp = x
        ood_seg =  gt_semantic_seg.new_zeros(gt_semantic_seg.size())
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
        loss_ood, loss_logit = self._ood_head_forward_train(ood_inp, img_metas, ood_seg, ood_score)        
        perturbe = ag.grad(loss_logit, inputs=[mask_adv], retain_graph=True)[0]
        perturbe = perturbe.detach()        
        num_images = 5 

        score = pow(10, 9)
        img_final, gt_semantic_seg_final, img_metas_final = img.clone(), gt_semantic_seg.clone(), img_metas
        ood_seg_final = ood_seg
        for _ in range(num_images):
           img_new, gt_semantic_seg_new, img_metas_new  = img.clone(), gt_semantic_seg.clone(), img_metas
           img_new, gt_semantic_seg_new, ood_seg_new, num_patch, corrdinate = self.render_patchv4(img_new, gt_semantic_seg_new)
           if score > torch.sum(perturbe * (img - img_new)):
              score = torch.sum(perturbe * (img - img_new))
              img_final, gt_semantic_seg_final, ood_seg_final = img_new, gt_semantic_seg_new, ood_seg_new

        img, gt_semantic_seg, ood_seg = img_final, gt_semantic_seg_final, ood_seg_final
        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        ood_inp = [v.detach() for v in x]

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
        loss_ood,_ = self._ood_head_forward_train(ood_inp, img_metas, ood_seg, ood_score)
        losses.update(loss_ood)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)        
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
        
        self.add_patch(img, gt_semantic_seg)
        img, gt_semantic_seg, ood_seg, num_patch, corrdinate = self.render_patchv4(img, gt_semantic_seg)
        mask_adv = nn.Parameter(torch.zeros(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).float(), requires_grad=True).cuda()
        img = img + mask_adv

        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        ood_inp = x#[v.detach() for v in x]

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
        loss_ood, loss_logit = self._ood_head_forward_train(ood_inp, img_metas, ood_seg, ood_score)
        perturbe = ag.grad(loss_logit, inputs=[mask_adv], retain_graph=True)[0]
        perturbe = perturbe.detach()        
        for i in range(img.size(0)):
            for j in range(num_patch):
                img[i,:,:,:] = img[i,:,:,:] + perturbe[i,:,:,:] * 0.5
                img[i, :, corrdinate[i][j][0]+corrdinate[i][j][2], corrdinate[i][j][1]+corrdinate[i][j][3]] = img[i, :, corrdinate[i][j][0]+corrdinate[i][j][2], corrdinate[i][j][1]+corrdinate[i][j][3]] + 1.5 * perturbe[i, :, corrdinate[i][j][0]+corrdinate[i][j][2], corrdinate[i][j][1]+corrdinate[i][j][3]]
        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        ood_inp = [v.detach() for v in x]

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
        loss_ood,_ = self._ood_head_forward_train(ood_inp, img_metas, ood_seg, ood_score)
        losses.update(loss_ood)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)        

        """
        self.add_patch(img, gt_semantic_seg)
        loss_decode_f, loss_ood_f, loss_aux_f = dict(), dict(), dict()
        loss_ood_f['ood.loss_ce'] = 0.0
        losses = dict()
        img_meta = []
        gt_semantic_seg_meta = []
        ood_seg_meta = []
        pic_num = 2
        for i in range(pic_num):
            img_new, gt_semantic_seg_new, img_metas_new  = img.clone(), gt_semantic_seg.clone(), img_metas
            img_new, gt_semantic_seg_new, ood_seg, num_patch, corrdinate = self.render_patchv4(img_new, gt_semantic_seg_new)
            if img_meta == []:
               img_meta, gt_semantic_seg_meta, ood_seg_meta = img_new, gt_semantic_seg_new, ood_seg
            else:
               img_meta = torch.cat([img_meta, img_new], dim=0)
               gt_semantic_seg_meta = torch.cat([gt_semantic_seg_meta, gt_semantic_seg_new], dim=0)
               ood_seg_meta = torch.cat([ood_seg_meta, ood_seg], dim=0)
        x = self.extract_feat(img_meta)
        ood_inp = [v.detach() for v in x]
        with torch.no_grad():
             seg_out = self._decode_head_forward_test(x, img_metas_new)
             seg_out = resize(input=seg_out,size=ood_seg_meta.size()[2:], mode="bilinear",align_corners=self.align_corners)
             ood_score = super(PatchEncoderDecoder, self)._get_ood_segmentation_score(x, ood_inp, seg_out, img_metas_new)
             ood_seg_meta = self._refine_ood_gt(ood_score, ood_seg_meta)
        loss_ood_all = []
        for i in range(pic_num):
            ood_per = [ood_inp[0][i].unsqueeze(0), ood_inp[1][i].unsqueeze(0), ood_inp[2][i].unsqueeze(0), ood_inp[3][i].unsqueeze(0)]
            loss_ood,_ = self._ood_head_forward_train(ood_per, img_metas_new, ood_seg_meta[i].unsqueeze(0), ood_score[i].unsqueeze(0))
            if loss_ood_f['ood.loss_ce']<loss_ood['ood.loss_ce']:
               loss_ood_f = loss_ood
               loss_decode_f = self._decode_head_forward_train([x[0][i].unsqueeze(0), x[1][i].unsqueeze(0), x[2][i].unsqueeze(0), x[3][i].unsqueeze(0)], img_metas, gt_semantic_seg_meta[i].unsqueeze(0))
               if self.with_auxiliary_head:
                  loss_aux_f = self._auxiliary_head_forward_train([x[0][i].unsqueeze(0), x[1][i].unsqueeze(0), x[2][i].unsqueeze(0), x[3][i].unsqueeze(0)], img_metas_new, gt_semantic_seg_meta[i].unsqueeze(0))               
               
        losses.update(loss_decode_f)
        losses.update(loss_ood_f)
        losses.update(loss_aux_f)
        """ 
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
        if out_ood.shape[2]!=logit_ood.shape[2]:
           out_ood = resize(
               input=out_ood,
               size=logit_ood.shape[2:],
               mode="bilinear",
               align_corners=self.align_corners)
        out_ood = out_ood * self.merge_weight + logit_ood
        return out_ood


@MODELS.register_module()
class PatchRenderCascadedEncoderDecoder(
        CascadeEncoderDecoder, PatchEncoderDecoder
    ):

    def __init__(
            self,
            num_stages,
            **kwargs
        ):

        self.num_stages = num_stages
        PatchEncoderDecoder.__init__(self, **kwargs)

    def encode_decode(self, img, img_metas):

        x = self.extract_feat(img)
        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
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






















