import os
import sys
sys.path.append(os.path.dirname(os.path.dirname("../")))
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import os
import os.path as osp

from argparse import ArgumentParser

from core.inference import AnomalyDetect
from road_utils.road_anomaly_benchmark.evaluation import Evaluation
from road_utils.road_anomaly_benchmark.paths import DIR_SRC


def road_benchmark(
        detect : AnomalyDetect,
        method_name = "baseline",
        dataset_name = 'AnomalyTrack-All',
        vis_anomaly = True
):

    vis_path = osp.join(DIR_SRC, "vis", method_name, dataset_name)
    if not osp.exists(vis_path):
        os.makedirs(vis_path, exist_ok=True)

    ev = Evaluation(
        method_name = method_name,
        dataset_name = dataset_name
    )

    for frame in tqdm(ev.get_frames()):
        # run method here
        result = detect.inference(frame.image)
        assert "anomaly" in result
        # provide the output for saving
        pred = result["anomaly"]
        ev.save_output(frame, pred)

        if vis_anomaly:
            filename = osp.join(vis_path, f"{frame.fid}.png")
            anomaly_vis = result.get("anomaly_vis", None)
            if anomaly_vis is not None:
                cv2.imwrite(filename, anomaly_vis)

    # wait for the background threads which are saving
    ev.wait_to_finish_saving()


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='anomaly dataset name in road', default='AnomalyTrack-validation')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    anomal_det = AnomalyDetect(args.config, args.checkpoint, args.device)
    road_benchmark(anomal_det, dataset_name=args.dataset, method_name="baseline")


if __name__ == '__main__':
    main()
