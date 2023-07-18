# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool

import numpy as np
from mmengine.logging import print_log
from mmengine.utils import is_str
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps
from .class_names import get_classes
from .mean_ap import tpfp_default, average_precision, print_map_summary

def get_cls_results(det_results, annotations, class_id, ):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """

    cls_dets = []
    cls_gts = []
    cls_gts_ignore = []

    # detection predictions start with index 0
    # hack: remap all det labels to 0 for pretrained models
    for det in det_results:            
        det_labels = det['labels'] * 0
        det_inds = det_labels == class_id
        cls_bboxes = det['bboxes'][det_inds, :]
        cls_scores = det['scores'][det_inds]
        # combine bboxes and scores together, shape is [N, 5]
        cls_bboxes_scores = np.hstack(
                    [cls_bboxes, cls_scores.reshape((-1, 1))])
        cls_dets.append(cls_bboxes_scores)

    # annotations have categories starting with index 1
    for ann in annotations:
        gt_inds = ann['labels']-1 == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore']-1 == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
    
    return cls_dets, cls_gts, cls_gts_ignore


def eval_tile_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             ioa_thr=None,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4,
             use_legacy_coordinate=False,
             eval_mode='area'):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Defaults to None.
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Defaults to None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc", "imagenet_det", etc. Defaults to None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmengine.logging.print_log()` for details.
            Defaults to None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1],
            PASCAL VOC2007 uses `11points` as default evaluate mode, while
            others are 'area'. Defaults to 'area'.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    assert eval_mode in ['area', '11points'], \
        f'Unrecognized {eval_mode} mode, only "area" and "11points" ' \
        'are supported'
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = 1
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # There is no need to use multi processes to process
    # when num_imgs = 1 .
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)

    eval_results = []
    
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
            tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
        
        if num_imgs > 1:
            # compute tp and fp for each image with multiple processes
            args = []
            if ioa_thr is not None:
                args.append([ioa_thr for _ in range(num_imgs)])

            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)],
                    [use_legacy_coordinate for _ in range(num_imgs)]))
        else:
            tpfp = tpfp_fn(
                cls_dets[0],
                cls_gts[0],
                cls_gts_ignore[0],
                iou_thr,
                area_ranges,
                use_legacy_coordinate,
                ioa_thr=ioa_thr)
            tpfp = [tpfp]

        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
                    bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        ap = average_precision(recalls, precisions, eval_mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    if num_imgs > 1:
        pool.close()
    
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)
    import ipdb; ipdb.set_trace()
    return mean_ap, eval_results
