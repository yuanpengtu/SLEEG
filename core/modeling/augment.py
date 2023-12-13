import torch
import math
import numbers
import random
import numpy as np


import torchvision.transforms as torch_tr
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.spatial import ConvexHull
from skimage.draw import polygon as Polygon
import cv2

from skimage.draw import random_shapes







def add_shape(patch_size):
    """ return a Tensor with a random shape
        1 for the shape
        0 for the background
        The size is controle by args.patch_size
    """
    image, _ = random_shapes(patch_size[0:2], min_shapes=6, max_shapes=10,
                             intensity_range=(0, 50),  min_size=patch_size[2],
                             max_size=patch_size[3], allow_overlap=True, num_channels=1)
    image = torch.round(1 - torch.FloatTensor(image)/255.)
    return image.squeeze().cuda()

def generate_mask_random_patch(images, patch_size):
    """ Generate a mask to attack a random shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).cuda()
    for i in range(len(mask)):
        x = random.randint(0, width - patch_size[0])
        y = random.randint(0, height - patch_size[1])
        h = patch_size[1]
        w = patch_size[0]
        shape = add_shape(patch_size)
        mask[i, x:x + w, y:y + h] = shape == 1.
    return mask.view(bsize, 1, width, height).expand_as(images)
def generate_mask_fractal(images, args):
    """ Generate a mask to attack a fractal shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, 3, width, height).cuda()

    try:
        fractal = next(args.fractal)
    except StopIteration:
        args.fractal = iter(data_loader("Fractal", args))
        fractal = next(args.fractal, None)
    fractal = fractal.cuda()

    for i in range(len(mask)):
        x = random.randint(50, width - patch_size[0])
        y = random.randint(0, height - patch_size[1])
        h = patch_size[1]
        w = patch_size[0]
        patch = torch.where(fractal[i].unsqueeze(0) > 0, torch.FloatTensor([1.]), torch.FloatTensor([0.])).cuda()
        mask[i, :, x:x + w, y:y + h] = patch
        # texture[i] += images[i, :, x:x + w, y:y + h].mean() - texture.mean() - 0.2
        # images[i, :] = torch.where(mask[i, :] > 0, texture[i, :], images[i, :])

    return mask, images








def uniform_random(left, right, size=None):
    rand_nums = (right - left) * np.random.random(size) + left
    return rand_nums
def random_polygon(edge_num, center, radius_range):
    angles = uniform_random(0, 2 * np.pi, edge_num)
    angles = np.sort(angles)
    random_radius = uniform_random(radius_range[0], radius_range[1], edge_num)
    x = np.cos(angles) * random_radius
    y = np.sin(angles) * random_radius
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    points = np.concatenate([x, y], axis=1)
    points += np.array(center)
    points = np.round(points).astype(np.int32)
    return points

class RandomRoadObject(object):
    def __init__(self, rcp, ignore_index=2, max_random_poly=10, min_sz_px=32, max_sz_px=256):
        self.rcp = rcp
        self.max_random_poly = max_random_poly
        self.min_sz_px = min_sz_px
        self.max_sz_px = max_sz_px
        self.ignore_index = ignore_index
        np.random.seed(1024)
    def __call__(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        num_objs = 1#np.random.randint(0, self.max_random_poly)
        img = img.cuda() 
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]#np.random.randint(self.min_sz_px, self.max_sz_px)
            pts = np.random.rand(15, 2) * w
            pts[:,0], pts[:, 1] = np.random.rand(15)*w, np.random.rand(15)*h
            hull = ConvexHull(pts)
            rr, cc = Polygon(pts[hull.vertices, 0], pts[hull.vertices, 1], [w, h])
            row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-h)) #np.random.randint(0, img.shape[1]-w), np.random.randint(0, img.shape[2]-w)#ids[0][idd], ids[1][idd]
            # clone image part
            img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
            gt_semantic_seg[:, row+rr, col+cc] = index_seg
            ood_seg[:, row+rr, col+cc] = index_ood
        return img, gt_semantic_seg, ood_seg, [row, col, rr, cc]



    def forwardv1(self, sample, row, col, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        num_objs = 1#np.random.randint(0, self.max_random_poly)
        img = img.cuda()
        img_multi = []
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]
            pts = np.random.rand(15, 2) * w
            pts[:,0], pts[:, 1] = np.random.rand(15)*w, np.random.rand(15)*h
            hull = ConvexHull(pts)
            rr, cc = Polygon(pts[hull.vertices, 0], pts[hull.vertices, 1], [w, h])
            #row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-h)) 
            img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
            gt_semantic_seg[:, row+rr, col+cc] = index_seg
            ood_seg[:, row+rr, col+cc] = index_ood
        return img, gt_semantic_seg, ood_seg, [row, col, rr, cc]





    def forwardv2(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        img_orig = np.array(img)
        img_copy = np.array(img)
        mask_orig= np.array(mask)
        mask_copy = np.array(mask)
        num_objs = 1
        img = img.cuda()
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]#np.random.randint(self.min_sz_px, self.max_sz_px)
            #pts = np.random.rand(20, 2) * w
            #pts[:,0], pts[:, 1] = np.random.rand(20)*w, np.random.rand(20)*h
            points = random_polygon(20, [min(w//2, h//2), max(w//2, h//2)], [min(w//2, h//2)-10, max(w//2, h//2)])     
            image_mask = np.zeros((w,h), dtype=np.uint8)            
            image_mask = cv2.fillPoly(image_mask, [points], (1, 0, 0))
            rr,cc = [], []
            for i in range(w):
                for j in range(h):
                    if image_mask[i][j]==1:
                       rr.append(i)
                       cc.append(j)
            rr,cc = np.array(rr), np.array(cc)
            row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-h))
            img_copy[:, row+rr, col+cc] = np.array(patch[:, rr, cc].cpu())
            mask_copy[row+rr, col+cc]   = index_ood#cut_mask[rr, cc]
            img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
            gt_semantic_seg[:, row+rr, col+cc] = index_seg
            ood_seg[:, row+rr, col+cc] = index_ood
        return img, gt_semantic_seg, ood_seg


    def forwardv3(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        img_orig = np.array(img)
        img_copy = np.array(img)
        mask_orig= np.array(mask)
        mask_copy = np.array(mask)
        num_objs = 1
        img = img.cuda()
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]#np.random.randint(self.min_sz_px, self.max_sz_px)
            #pts = np.random.rand(20, 2) * w
            pts[:,0], pts[:, 1] = np.random.rand(20)*w, np.random.rand(20)*h


            dst = cv2.cornerHarris(patch,blockSize= 2,ksize= 3,k= 0.04)
            print(dst)


            hull = ConvexHull(pts)
            rr, cc = Polygon(pts[hull.vertices, 0], pts[hull.vertices, 1], [w, w])
            row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-w)) #np.random.randint(0, img.shape[1]-w), np.random.randint(0, img.shape[2]-w)#ids[0][idd], ids[1][idd]
            # clone image part
            #print(pts, row, col)
            img_copy[:, row+rr, col+cc] = np.array(patch[:, rr, cc].cpu())
            mask_copy[row+rr, col+cc]   = index_ood#cut_mask[rr, cc]
            img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
            gt_semantic_seg[:, row+rr, col+cc] = index_seg
            ood_seg[:, row+rr, col+cc] = index_ood
        return img, gt_semantic_seg, ood_seg


    def forwardv4(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        img_orig = np.array(img)
        img_copy = np.array(img)
        mask_orig= np.array(mask)
        mask_copy = np.array(mask)
        num_objs = 1#np.random.randint(0, self.max_random_poly)
        img = img.cuda()
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]#np.random.randint(self.min_sz_px, self.max_sz_px)
            pts = np.random.rand(24, 2) * w
            pts[:,0], pts[:, 1] = np.random.rand(24)*w, np.random.rand(24)*h
            for j in range(3):
                hull = ConvexHull(pts[j*8:(j+1)*8, :])
                rr, cc = Polygon(pts[hull.vertices, 0], pts[hull.vertices, 1], [w, h])
                row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-h))
                # clone image part
                img_copy[:, row+rr, col+cc] = np.array(patch[:, rr, cc].cpu())
                mask_copy[row+rr, col+cc]   = index_ood#cut_mask[rr, cc]
                img[:, row+rr, col+cc] = patch[:, rr, cc].cuda()
                gt_semantic_seg[:, row+rr, col+cc] = index_seg
                ood_seg[:, row+rr, col+cc] = index_ood
        return img, gt_semantic_seg, ood_seg


    def forwardv5(self, sample, gt_semantic_seg, ood_seg, patch, index_ood, index_seg):
        img, mask = sample["image"], sample["label"].squeeze(0)
        num_objs = 1
        img = img.cuda()
        st = torch.Tensor([patch.unsqueeze(0).shape[2], patch.unsqueeze(0).shape[3], 60, 90]).int()
        for i in range(0, num_objs):
            w,h = patch.shape[1], patch.shape[2]
            mask = generate_mask_random_patch(patch.unsqueeze(0), st)
            mask = mask.squeeze(0)
            #print(mask.shape, patch.shape, img[:, row+w, col+h].shape)
            mask_sub = torch.sum(mask, dim=0).unsqueeze(0)//3
            
            row, col = random.choice(range(img.shape[1]-w)), random.choice(range(img.shape[2]-h))
            #print(mask.shape, patch.shape, img[:, row:row+w, col:col+h].shape)# clone image part
            img[:, row:row+w, col:col+h] = patch.cuda() * mask + (1-mask) * img[:, row:row+w, col:col+h]
            gt_semantic_seg[:, row:row+w, col:col+h] = index_seg * torch.ones(gt_semantic_seg[:, row:row+w, col:col+h].shape).cuda() * mask_sub + (1-mask_sub) * gt_semantic_seg[:, row:row+w, col:col+h]
            ood_seg[:, row:row+w, col:col+h] = index_ood * torch.ones(ood_seg[:, row:row+w, col:col+h].shape).cuda() * mask_sub + (1-mask_sub) * ood_seg[:, row:row+w, col:col+h]
        return img, gt_semantic_seg, ood_seg


