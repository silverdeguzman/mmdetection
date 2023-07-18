# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet.registry import METRICS
from ..functional import eval_tile_map


@METRICS.register_module()
class TilingMetric(BaseMetric):
    """Tiling evaluation metric.

    Evaluate detection mAP for synthetic data

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        ioa_thrs (float or List[float]): IoA threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'target'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.3,
                 ioa_thrs: Union[float, List[float]] = 0.3,
                #  scale_ranges: Optional[List[tuple]] = None,
                 scale_ranges: Optional[List[tuple]] = [(0, 300)],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) else iou_thrs
        self.ioa_thrs = [ioa_thrs] if (isinstance(ioa_thrs, float)
                                       or ioa_thrs is None) else ioa_thrs
        assert isinstance(self.iou_thrs, list) and isinstance(
            self.ioa_thrs, list)
        assert len(self.iou_thrs) == len(self.ioa_thrs)
        self.scale_ranges = scale_ranges

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for ix, data_sample in enumerate(data_samples):
            result = dict()

            instances = data_sample['instances']
            gt_labels = []
            gt_bboxes = []
            
            for ins in instances:
                gt_labels.append(ins['bbox_label'])
                gt_bboxes.append(ins['bbox'])
            ann = dict(
                img_id=data_sample['img_id'],
                labels=np.array(gt_labels, dtype=np.int64),
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4)))
            
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            # self.filter_result(result)

            self.results.append((ann, result))

    def filter_result(self, result: dict):
        # filter predictions by certain classes
        cls_ids = [0, 2, 4, 6, 7, 14, 33]
        keep_inds = []
        for ind, label in enumerate(result['labels']):
            if label in cls_ids:
                keep_inds.append(ind)
        result['labels'] = result['labels'][keep_inds]
        result['bboxes'] = result['bboxes'][keep_inds]
        result['scores'] = result['scores'][keep_inds]
        assert(len(result['labels']) \
               == len(result['bboxes']) \
               == len(result['scores']))
    
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        
        logger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        eval_results = OrderedDict()
        
        mean_aps = []
        for i, (iou_thr,
                ioa_thr) in enumerate(zip(self.iou_thrs, self.ioa_thrs)):
            print_log(f'\n{"-" * 15}iou_thr, ioa_thr: {iou_thr}, {ioa_thr}'
                      f'{"-" * 15}')

            mean_ap, _ = eval_tile_map(
                preds,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                ioa_thr=ioa_thr,
                logger=logger)

            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results

    def results2json(self, results: Sequence[dict], json_outfile: str):
        """Dump the detection results to a COCO style json file.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            json_outfile (str): Filepath to save resulting json to

        Returns:
            dict: Possible keys are "bbox" and values are corresponding 
            filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                # data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)
        with open(json_outfile, 'w') as f:
            json.dump(f, indent=4)
        return bbox_json_results
    
    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]