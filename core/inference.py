import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

from typing import Union, Dict, Tuple, List
from PIL import Image
from itertools import chain

from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.apis import init_segmentor
from mmseg.core.evaluation import get_palette
from mmcv.parallel import collate, scatter
from .cityscapes_labels import create_id_to_train_id_mapper, colorize_labels as colorize_cityscapes_labels, \
    create_id_to_name, create_name_to_id, denormalize_city_image, colorize_osr_labels as colorize_osr_cityscapes_labels, \
    create_osr_id_to_train_id_mapper
class AnomalyDetect:

    def __init__(
            self,
            config : str,
            checkpoint : str,
            device : Union[str, int],
            palette : str = "cityscapes"
    ):

        self.model = init_segmentor(config, checkpoint, device)
        self.model.test_cfg.update({"with_ood" : True})
        self.palette = chain(*get_palette(palette))

        self.cfg = self.model.cfg
        self.device = device
        # build the data pipeline
        test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(test_pipeline)

    def parse_results(self, results : Union[List[torch.Tensor], Tuple[List[torch.Tensor]]]):

        if self.cfg.model.test_cfg.get("with_ood", False):
            assert isinstance(results, tuple)
            assert len(results) == 2
            pred, anomaly = (v[0] for v in results)

            pred = pred.astype(np.uint8)
            anomaly = anomaly.astype(np.float32)
        else:
            assert isinstance(results, torch.Tensor)
            pred = results[0].astype(np.uint8)
            anomaly = None

        output = {
            "pred": pred,
            "anomaly": anomaly
        }

        return output

    def inference(self, img : np.ndarray):

        # prepare data
        data = dict(img=img)
        data = self.test_pipeline(data)

        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)

        output = self.parse_results(result)
        output.update({"img" : img})

        output = self.postprocess(output)

        return output

    def vis_output(self, results : Dict[str, np.ndarray]):

        pred = results.get("pred", None)
        if pred is None:
            return results
        color = colorize_cityscapes_labels(pred)
        
        mask = Image.fromarray(pred, mode="L")
        mask.putpalette(self.palette)

        #results["pred_vis"] = np.array(mask)
        results["pred_vis"] = np.array(color)
        return results

    def vis_anomaly(self, results : Dict[str, np.ndarray]):

        anomaly = results.get("anomaly", None)
        if anomaly is None:
            return results

        anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min())

        aimg = (anomaly * 255.0).astype(np.uint8)
        aimg = cv2.applyColorMap(aimg, colormap=cv2.COLORMAP_JET)
        output = cv2.addWeighted(results["img"], 0.5, aimg, 0.5, 0.0)

        results["anomaly_vis"] = output

        return results

    def postprocess(self, results : Dict[str, np.ndarray]):

        results = self.vis_output(results)
        results = self.vis_anomaly(results)

        return results

