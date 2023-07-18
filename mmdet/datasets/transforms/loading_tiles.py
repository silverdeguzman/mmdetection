# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import cv2 as cv 
from mmcv.transforms import BaseTransform
from mmengine.fileio import get

from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadTileImage(BaseTransform):
    """Load an image from ``results['img_path']`` and crops to tile coords rect.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'color',
        imdecode_backend: str = 'cv2',
        backend_args: dict = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        x, y, w, h = results['tile_coords']
        img_bytes = get(results['img_path'], backend_args=self.backend_args)
        full_img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        tile = full_img[y:y+h,x:x+w]

        if self.to_float32:
            tile = tile.astype(np.float32)

        results['img'] = tile
        results['img_shape'] = tile.shape[:2]
        results['full_img_shape'] = full_img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'backend_args={self.backend_args})')
        return repr_str