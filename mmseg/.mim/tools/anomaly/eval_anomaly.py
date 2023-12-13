import os
import sys
sys.path.append(os.path.dirname(os.path.dirname("../")))
import torch
import torch.nn.functional as F
import numpy as np


from argparse import ArgumentParser
import core
from terminaltables import AsciiTable
from core.inference import AnomalyDetect
from core.evaluation import OODEvaluation

def main():
    parser = ArgumentParser()
    parser.add_argument('--img-root', help='path to ood data',
                        default="./data/segmentme/datasets/dataset_FishyLAF/leftImg8bit")
    parser.add_argument('--img-suffix', help="suffix of img files", default="_leftImg8bit.png")
    parser.add_argument('--gt-root', help='path to gt data',
                        default="./data/segmentme/datasets/dataset_FishyLAF/ood")
    parser.add_argument('--gt-suffix', help="suffix of gt files", default="_ood_segmentation.png")

    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    anomal_det = AnomalyDetect(args.config, args.checkpoint, args.device)
    evaluator = OODEvaluation(
        anomal_det, args.img_root, args.gt_root, img_suffix=args.img_suffix, gt_suffix=args.gt_suffix
    )
    evaluator.calculate_ood_scores()

    headers = [("Metric", "Value")]
    headers.extend(evaluator)
    tab = AsciiTable(headers)
    print(tab.table)


if __name__ == '__main__':
    main()
