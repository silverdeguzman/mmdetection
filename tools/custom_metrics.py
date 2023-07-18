import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine.config import Config
from mmdet.utils import register_all_modules

from mmengine.evaluator import Evaluator
from mmengine.fileio import load

from pathlib import Path 
import pickle 
import json 


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet get offline eval metrics given a prediction pickle file or results json')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='dump predictions to a pickle file for offline evaluation')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    results_json = args.out_dir + '/results.json'
    predictions_pickle = args.out_dir + '/results.pickle'
    data = load(predictions_pickle)

    evaluator = Evaluator(metrics=dict(type='TilingMetric'))
    evaluator.offline_evaluate(data, chunk_size=128)

if __name__ == "__main__":
    register_all_modules()

    main()