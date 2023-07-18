# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch 

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner 
from mmengine.runner import load_checkpoint, weights_to_cpu, save_checkpoint
from mmengine.model.utils import revert_sync_batchnorm
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmdet.registry import MODELS

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--pretrained-weights', 
        default=None,
        help='weights pth, if available this will export weights to onnx')
    parser.add_argument(
        '--onnx-path',
        default='model.onnx',
        help='paths to save exported onnx weights')
    parser.add_argument(
        '--dynamic',
        action='store_true',
        default=False,
        help='export onnx model with dynamic batch size')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # build model
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)

    if args.pretrained_weights:
        pretrained_weights = args.pretrained_weights
        checkpoint = load_checkpoint(model, pretrained_weights)
        num_channels = 6
        dummy_input = torch.randn(1, num_channels, 512, 512, requires_grad=True)
        if args.dynamic:
            dynamic_axes = {'input': {0 : 'batch_size'},
                            'heatmap' : {0 : 'batch_size'}, 'wh': {0 : 'batch_size'}, 'offset': {0 : 'batch_size'}}
            onnx_path = args.onnx_path.replace('.onnx', '-dynamic.onnx')
        else:
            dynamic_axes = None
            onnx_path = args.onnx_path.replace('.onnx', '-b1.onnx')
        model.eval()
        print('Exporting model to ONNX')
        torch.onnx.export(model, 
                      dummy_input,
                      onnx_path,
                      verbose=True,
                      export_params=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['heatmap', 'wh', 'offset'],
                      do_constant_folding=True,
                      dynamic_axes=dynamic_axes)

    else:
        model.train()
        data_loader = Runner.build_dataloader(cfg.train_dataloader)
        optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)

        for idx, data_batch in enumerate(data_loader):
            if idx == 10:
                break
            data = model.data_preprocessor(data_batch)
            imgs_shape = data['inputs'].shape
            dummy_imgs = torch.randn(imgs_shape[0], 6, imgs_shape[2], imgs_shape[3])
            data['inputs'] = dummy_imgs
            out = model.train_step(data, optim_wrapper)
            # print(idx, out['loss'])
            
        # start training    
        # runner.train()

        # Save model weights
        print('Saving weights')
        checkpoint = {
            'state_dict': weights_to_cpu(model.state_dict()),
        }
        savepath = f'{cfg.work_dir}/dummy.pth'
        save_checkpoint(
            checkpoint,
            savepath, 
        )



    print('done')


if __name__ == '__main__':
    main()
