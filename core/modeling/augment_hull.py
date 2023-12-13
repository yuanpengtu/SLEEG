import torch
import math
import numbers
import random
import numpy as np


import torchvision.transforms as torch_tr
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.spatial import ConvexHull
from skimage.draw import polygon as Polygon

class RandomRoadObject(object):
    def __init__(self, rcp, ignore_index=2, max_random_poly=10, min_sz_px=32, max_sz_px=256):
        self.rcp = rcp
        self.max_random_poly = max_random_poly
        self.min_sz_px = min_sz_px
        self.max_sz_px = max_sz_px
        self.ignore_index = ignore_index
    def __call__(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        randseed = 1024#sample.get("randseed", -1)
        if randseed >= 0:
            state = np.random.get_state()
            np.random.seed(randseed)

        img, mask = sample["image"], sample["label"].squeeze(0)
        img_orig = np.array(img)
        img_copy = np.array(img)
        mask_orig= np.array(mask)
        mask_copy = np.array(mask)
        num_objs = 1#np.random.randint(0, self.max_random_poly)

        #img1 = Image.fromarray(img_copy.squeeze(0).transpose(1, 2, 0).astype(np.uint8))#.convert('RGB')
        #img1.save('./results/saveimg/oricut.png')
        img = img.cuda() 
        for i in range(0, num_objs):
            w = patch.shape[1]#np.random.randint(self.min_sz_px, self.max_sz_px)
            pts = np.random.rand(10,2) * w
            hull = ConvexHull(pts)
            rr, cc = Polygon(pts[hull.vertices, 0], pts[hull.vertices, 1], [w, w])
            #ids = np.where(mask_copy[w:mask_copy.shape[0]-w-1, w:mask_copy.shape[1]-w-1] == 1)
            #if ids[0].size > 0:
            #idd = np.random.randint(0, img.shape[0])
            row, col = np.random.randint(0, img.shape[1]-w), np.random.randint(0, img.shape[2]-w)#ids[0][idd], ids[1][idd]

            # clone image part
            #x, y = np.random.randint(0, img_copy.shape[1] - w - 1), np.random.randint(0, img_copy.shape[0] - w - 1)
            #cut_img = img_orig[y:y+w, x:x+w, :]  
            #cut_mask = np.zeros(shape=cut_img.shape[:2], dtype=np.uint8)
            #mc = mask_orig[y:y+w, x:x+w]  
            #flag = (mc > 0) & (mc < 255)
            #cut_mask[flag] = mc[flag] 
            img_copy[:, row+rr, col+cc] = np.array(patch[:, rr, cc].cpu())
            mask_copy[row+rr, col+cc]   = index_ood#cut_mask[rr, cc]
            img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
            gt_semantic_seg[:, row+rr, col+cc] = index_seg
            ood_seg[:, row+rr, col+cc] = index_ood
        #img_copy[mask_copy>0, :] = img_orig[mask_copy>0, :]
        #mask_copy = mask_copy & mask_orig
        #mask = Image.fromarray(mask_copy.astype(np.uint8))
        #print(img_copy.shape, mask_copy.shape)
        #img = Image.fromarray(img_copy.squeeze(0).transpose(1, 2, 0).astype(np.uint8))#.convert('RGB')
        #img.save('./results/saveimg/cut.png')
        #mask.save('./results/saveimg/mask.png')
        if randseed >= 0:
            np.random.set_state(state)

        return img, gt_semantic_seg, ood_seg
