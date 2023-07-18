# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import torch

from mmengine.config import Config
from mmdet.datasets import TilingDataset
from mmengine.evaluator import Evaluator

from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from pathlib import Path 
import cv2 as cv 
import pickle 
import json 
import time 


class LoadImage:
    def __call__(self, results):
        x, y, w, h = results['tile_coords']
        img = cv.imread(results['img_path'])
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tile = rgb[y:y+h,x:x+w]
        results['img'] = tile
        results['ori_shape'] = img.shape 
        results['img_shape'] = results['img'].shape
        return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')

    
    call_args = vars(parser.parse_args())
    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''


    init_kws = ['config', 'checkpoint', 'device', 'palette', 'work_dir']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)
    
    if not torch.cuda.is_available():
        init_args['device'] = 'cpu'

    return init_args, call_args



def main(image_inputs):
    init_args, call_args = parse_args()
    cfg = Config.fromfile(init_args['config'])
    
    inferencer = DetInferencer(model=init_args['config'],
                               weights=init_args['checkpoint'], 
                               device=init_args['device'],
                               palette=init_args['palette'])
    if not isinstance(image_inputs, TilingDataset):
        return
    dataset = image_inputs
    
    num_inferenced = 0
    results_list = []
    preds_list = []

    predictions_pickle = call_args['out_dir'] + '/predictions.pickle'
    results_pickle = call_args['out_dir'] + '/results.pickle'
    results_json = call_args['out_dir'] + '/results.json'

    tic = time.time()

    for idx, data_batch in enumerate(dataset):
        img = data_batch['img']
        img_path = data_batch['img_path']

        inputs = img
        file_name = f"{Path(img_path).parent.name}/{Path(img_path).name}"
        save_args = {"tile_id": data_batch['tile_id'], "tile_coords": data_batch['tile_coords'], 
                     "file_name": file_name}
        
        # results is a dict, preds is a list
        results, preds = inferencer(inputs=inputs, save_args=save_args, **call_args)
        preds[0].instances = data_batch["instances"]

        # add gt and metadata to results
        gt_metadata = {'file_name': file_name, 'img_id': data_batch['img_id'],
                     'img_shape': data_batch['img_shape'][:2], 'ori_shape': data_batch['ori_shape'][:2],
                     'tile_id': data_batch['tile_id'], 'tile_coords': data_batch['tile_coords'], 
                     'instances': data_batch['instances']}
        # remove visualization in order to save to json
        results.update(gt_metadata)
        results['pred_instances'] = preds[0].pred_instances
        results.pop('visualization', None)
        results_list.append(results)
        preds_list.extend(preds)
        
        num_inferenced += 1

    toc = time.time()
    print(f'Time elapsed = {toc-tic}')

    assert len(preds_list) == len(results_list)
    with open(predictions_pickle, 'wb') as f:
        pickle.dump(preds_list, f)
    with open(results_pickle, 'wb') as f:
        pickle.dump(results_list, f)
    # with open(results_json, 'w') as f:
    #     json.dump(results_list, f)

    print(f"Number of tiles inferences = {num_inferenced}")
    print('done')


if __name__ == "__main__":
    '''
    python tools/custom_eval.py configs/centernet/centernet_r18_8xb16-crop512-140e_tiling.py pretrained-weights/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth 
    '''
    register_all_modules()
    data_root = ""
    annotations_path = "test_single_annotations.txt"

    pipeline = [
        LoadImage(),
    ]

    toy_dataset = TilingDataset(tile_size=(640, 640), 
                                data_root=data_root, 
                                ann_file=annotations_path,
                                pipeline=pipeline)

    main(toy_dataset)