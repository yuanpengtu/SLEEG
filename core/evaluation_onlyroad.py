"""
Adapted from https://github.com/matejgrcic/DenseHybrid/tree/853ac07cb5315bbee550649bd8e1c7076c66a0e2/evaluations
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import cv2
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from core.inference import AnomalyDetect
import matplotlib.pyplot as plt
from torchvision.utils import save_image
class OODEvaluation:

    def __init__(
            self,
            detect : AnomalyDetect,
            img_dir : str,
            gt_dir : str,
            img_suffix : str = ".jpg",
            gt_suffix : str = ".png",
            ignore_id = 2
    ):
        self.detect = detect
        self.ignore_id = ignore_id
        #self.ignore_id = 255
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        
        self.eval_paris = dict()
        countall = 0
        for root, dirs, files in os.walk(gt_dir):
            for file in files:
                #print(file, gt_suffix)
                if not file.endswith(gt_suffix):
                    continue
                countall+=1
                #if countall>10:
                #   break
                gt_file = osp.join(root, file)
                img_file = osp.join(
                    root.replace(gt_dir, img_dir),
                    file.replace(gt_suffix, img_suffix)
                )
                road_file = img_file.replace('/leftImg8bit/', '/gtCoarse/')                
                road_file = road_file.replace('_leftImg8bit.png', '_gtCoarse_labelTrainIds.png')
                if not osp.exists(img_file):
                    continue

                file_id = file.replace(img_suffix, "")
                self.eval_paris[file_id] = (img_file, gt_file, road_file)

        print(f"Load {len(self.eval_paris)} image-gt pairs for evaluation")
        self.eval_results = dict()

    def calculate_auroc(self, conf, gt):

        fpr, tpr, threshold = roc_curve(gt, conf)
        precision, recall, threshold_t = precision_recall_curve(gt, conf)
        

        #recall = np.arange(len(tpr)) * tpr /np.sum(gt)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0

        print('Started FPR search.')
        k = 0.0
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        print(k)
        return roc_auc, fpr_best, k, precision, recall

    def calculate_ood_scores(self, averaged_by_img = False):

        total_conf = []
        total_gt = []
        count = 0
        tmp = []#[0, 3, 11, 15, 19, 20, 22, 23, 25, 27]
        dictscore = dict()
        count = 1
        with tqdm(total=len(self.eval_paris)) as progress_bar, torch.no_grad():
            counts = 0
            for step, (img, gt, road_file) in self.eval_paris.items():
                img = np.array(Image.open(img))
                road_file = np.array(Image.open(road_file))#.sum(axis=1)
                lbl = np.array(Image.open(gt))
                #print(lbl.shape, road_file.shape)
                res = self.detect.inference(img)
                res = self.detect.postprocess(res)
                newlbl = lbl.flatten()
                  
                #road_file = 
                
                road_file = road_file.flatten()
                road_file[road_file==0] = 1 
                road_file[road_file==2] = 1
                road_file[road_file==255] = 0
                newlbl[newlbl==0] = 5
                newlbl[newlbl==1] = 6
                
                newlbl = road_file * newlbl
                newlbl[newlbl==0] = 2
                newlbl[newlbl==5] = 0
                newlbl[newlbl==6] = 1 
                #print(torch.unique(torch.Tensor(road_file),return_counts=True))
                #print(torch.unique(torch.Tensor(lbl),return_counts=True))
                #lbl[lbl==2]=0
                #lbl[lbl==0] = 5
                #lbl[lbl==1] = 0
                #lbl[lbl==5] = 1 #lost and found dataset
                
  
                #count1 = [0, 0, 0, 0]
                #print(count1)
                #lbl[lbl==0] = 30
                #lbl[lbl==1] = 60
                #lbl[lbl==2] = 90
                #lblnew = cv2.applyColorMap(lbl, cv2.COLORMAP_JET)#cv2.cvtColor(lbl, cv2.COLOR_GRAY2BGR)
                
                #cv2.imwrite('results/savelaf/color_'+str(count)+'.png', lblnew)
                #save_image(torch.Tensor(res["pred_vis"]), 'results/savestatic/predict_'+str(count)+'.png')
                #print(res["anomaly_vis"].shape)
                #save_image(torch.cat([torch.Tensor(lbl), torch.Tensor(lbl), torch.Tensor(lbl)], dim=0), 'results/savelaf/color_'+str(count)+'.png')
                #cv2.imwrite('results/ori'+str(count)+'.png', img)                
                conf_probs = res["anomaly"]
                count+=1
                #dictscore[img_dir] = conf_probs 

                #count += 1
                abnormal_vis = res["anomaly_vis"]
                #lab = lbl.flatten().copy()
                #lab[lab==0] = 1
                #lab[lab==2] = 0 
                #abnormal_vis *= lab.reshape(1024,2048,1)
                #lblimg = lbl
                #lblimg[lblimg!=1] = 0
                #lblimg[lblimg==1] = 255
                #lblnewi = lblimg.copy()
                #lblnewi[lblnewi!=20] = 0
                #lblnewi = torch.Tensor(lblnewi).unsqueeze(2)
                #lblimg = torch.Tensor(lblimg).unsqueeze(2)
                #lblimg = torch.cat([lblnewi, lblnewi, lblimg], dim=2)
                #imgnew = img * 0.5 +  lblimg.numpy() * 0.5
                #cv2.imwrite("./results/savestatic/vis_"+str(count)+'.png', abnormal_vis)
                #cv2.imwrite("./results/savenewlaf/predict_"+str(count)+'.png', res["pred_vis"].reshape(1024, 2048, 3))
                #cv2.imwrite("./results/savestatic/oriimg_"+str(count)+'.png',  imgnew)
                #print(img_dir, "hello")
                #lbl[lbl==255] = 2 
                #print(max(road_file), min(road_file))
                label = newlbl#.flatten()
                #label[label==255] = 2
                #print(list(set(label)))
                conf_probs = conf_probs.flatten()
                #label[label==] = self.ignore_id
                gt = label[label != self.ignore_id]
                total_gt.append(gt)
                conf = conf_probs[label != self.ignore_id]
                total_conf.append(conf)

                progress_bar.update(1)
        #np.save('anomaly_validation.npy', dictscore)                 
        if averaged_by_img:

            total_AP = []
            total_fpr = []
            total_roc = []
            cou = 1
            precisions_list, recalls_list = [], []
            for conf, gt in zip(total_conf, total_gt):

                item_ap = average_precision_score(gt, conf)
                total_AP.append(item_ap)
                roc_auc, fpr, t, precision, recall = self.calculate_auroc(conf, gt)
                #print(precision, recall)
                total_roc.append(roc_auc)
                total_fpr.append(fpr)
                #precisions_list.extend(precision)
                #recalls_list.extend(recall)
                
                #plt.plot(recall, precision)
                #fontsize = 14
                #plt.xlabel('Recall', fontsize = fontsize)
                #plt.ylabel('Precision', fontsize = fontsize)
                #plt.title('Precision Recall Curve')
                #plt.legend()
                #f = plt.gcf() 
                #f.savefig("./results/PR/savePR"+str(cou)+".png")
                #plt.show()
                #f.clear()
                #cou+=1
            ap = float(np.nanmean(total_AP)) * 100.
            fpr = float(np.nanmean(total_fpr)) * 100.
            roc_auc = float(np.nanmean(total_roc)) * 100.

        else:
            total_gt = np.concatenate(total_gt, axis=0)
            total_conf = np.concatenate(total_conf, axis=0)
            ap = average_precision_score(total_gt, total_conf)
            roc_auc, fpr, threshold, precision, recall = self.calculate_auroc(total_conf, total_gt)
            """
            plt.plot(recall, precision)
            fontsize = 14
            plt.xlabel('Recall', fontsize = fontsize)
            plt.ylabel('Precision', fontsize = fontsize)
            plt.title('Precision Recall Curve')
            plt.legend()
            f = plt.gcf() 
            f.savefig("./results/PR/savePRall"+".png")
            plt.show()
            f.clear()            
            """


        self.eval_results["AP"] = round(ap*100., 2)
        self.eval_results["FPR"] = round(fpr*100., 2)
        self.eval_results["AUROC"] = round(roc_auc*100., 2)

    def __iter__(self):

        yield from self.eval_results.items()
