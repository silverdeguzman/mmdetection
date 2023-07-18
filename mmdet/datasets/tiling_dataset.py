from mmengine.dataset import BaseDataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path 
import json 

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class TilingDataset(BaseDetDataset):
    
    METAINFO = {
       'classes': ('drone'),
        'palette': [(220, 20, 60)]
    }

    def __init__(self, 
                 tile_size: Tuple[int, int], 
                 backend_args: dict = None,
                 **kwargs) -> None:
        assert len(tile_size) == 2, 'tile size expects a (h, w) value pair'
        self.tile_size = tile_size
        self.offset = 0
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        print("*****")
    
        annotations_path = Path(self.ann_file)
        
        assert isinstance(annotations_path, Path) and annotations_path.suffix == '.txt'
        with open(f'{annotations_path}', 'r') as f:
            lines = f.readlines()
        annotations_path_list = [Path(f'{self.data_root}/{line.strip()}') for line in lines if not line.isspace()]
        
        data_list = []
        for ap in annotations_path_list:
            with open(ap, 'r') as f:
                annotations = json.load(f)
            parent_dir = ap.parent.name
            data_list_i = self._tile_images_and_annotations(annotations, parent_dir)
            data_list.extend(data_list_i)
        return data_list

    def _tile_images_and_annotations(self, annotations, parent_dir) -> List[dict]:
        item_list = []
        count = 0

        for idx, image in enumerate(annotations['images']):
            fname = image['file_name']
            frame_index = image['frame_index']
            height, width = image['height'], image['width']
            # rects are [x, y, w, h] and x, y are top left
            tile_rects = get_tile_rects(height, width, self.tile_size[0], self.tile_size[1])
            image_full_path = (f'{self.data_root}/{parent_dir}/{fname}')
       
            frame_annotations = annotations['frame_annotations'][idx]
            assert frame_index == frame_annotations['frame_index']
            tile_annotations = self._tile_annotation(frame_annotations, height, width)

            for t_idx, tile_coord in enumerate(tile_rects):
                tile_id = tile_annotations[t_idx]['tile_id']
                instances = tile_annotations[t_idx]['instances']

                data_info = {
                    "img_path": image_full_path,
                    "img_id": count + self.offset,
                    "img_id_str": f'{idx}_{tile_id}',
                    "height": height,
                    "width": width,
                    "tile_coords": tile_coord,
                    "tile_id": tile_id,
                    "instances": instances
                }
                count += 1
                item_list.append(data_info)

        self.offset += count    
        return item_list
    
    def _tile_annotation(self, annotations, height, width) -> List[dict]:
        annotations = annotations['annotations']
        tile_annotations = tile_video_annotations(
                                annotations, height, width, self.tile_size[0], self.tile_size[1])
        return tile_annotations
    
def tile_video_annotations(annotations: List[Dict], image_h: int, image_w: int, tile_h: int, tile_w: int) -> List[Dict]:
    # rects are from original image resolution
    rects = get_tile_rects(image_h, image_w, tile_h, tile_w) 
    full_tile_annotations = []

    for tile_id, rect in enumerate(rects):
        x, y, w, h = rect 
        left = x 
        top = y 
        right = x+w 
        bottom = y+h
        tile = [left, top, right, bottom]
        tile_annotations = []

        for annotation in annotations:
            if annotation_inside_tile(annotation, tile):
                # remaps bbox coords to tile resolution
                tile_annotation = get_tile_annotation(annotation, tile)
                tile_annotations.append(tile_annotation)

        full_tile_annotations.append({'tile_id': tile_id, 'instances': tile_annotations})
    return full_tile_annotations

def get_tile_rects(image_h: int, image_w: int, tile_h: int, tile_w: int) -> List[List[int]]:
        h_rem = image_h % tile_h
        w_rem = image_w % tile_w
        
        nw = image_w // tile_w
        nh = image_h // tile_h
        
        if h_rem == 0:
            last_row_h = tile_h
        else:
            last_row_h = h_rem 
            nh+=1
        if w_rem == 0:
            last_col_w = tile_w
        else:
            last_col_w = w_rem 
            nw+=1

        rects = []

        for i in range(nw):
            for j in range(nh):
                # these are the cropped coords x, y, w, h in original image
                cx = i * tile_w
                cy = j * tile_h
                if (i == nw-1):
                    cw = last_col_w
                else:
                    cw = tile_w
                if (j == nh-1):
                    ch = last_row_h
                else:
                    ch = tile_h
                rect = [cx, cy, cw, ch]
                rects.append(rect)      
        return rects

def annotation_inside_tile(annotation: Dict, tile: List[int]) -> bool:
    left, top, w, h = annotation["bbox"]
    right = left + w
    bottom = top + h

    if left >= tile[2]:
        return False
    if top  >= tile[3]:
        return False
    if right <= tile[0]:
        return False
    if bottom <= tile[1]:
        return False 
    return True

def get_tile_annotation(annotation: Dict, tile: List[int], center_coords: bool = False) -> Dict:
    bbox_left, bbox_top, w, h = annotation["bbox"]
    bbox_right = bbox_left + w
    bbox_bottom = bbox_top + h

    tile_left, tile_top, tile_right, tile_bottom = tile 

    # shifts box coordinates in the case only part of the box is in
    # the tile. otherwise, it keeps its original coordinates
    new_bbox_left = max(bbox_left, tile_left)
    new_bbox_top = max(bbox_top, tile_top)
    new_bbox_right = min(bbox_right, tile_right)
    new_bbox_bottom = min(bbox_bottom, tile_bottom)

    tile_annotation = annotation.copy()

    # convert back to x,y,w,h
    if center_coords:
        # [cx, cy, w, h]
        new_bbox = [(new_bbox_left + new_bbox_right) * 0.5, 
                    (new_bbox_top + new_bbox_bottom) * 0.5, 
                    new_bbox_right - new_bbox_left,
                    new_bbox_bottom - new_bbox_top]  
    else:
        # [tlx, tly, w, h]
        new_bbox = [new_bbox_left, 
                    new_bbox_top, 
                    new_bbox_right - new_bbox_left,
                    new_bbox_bottom - new_bbox_top] 
        
    # shift box x,y coordinates to (0,0) origin
    new_bbox[0] -= tile_left 
    new_bbox[1] -= tile_top 

    # bbox instances is xyxy coordinates
    new_bbox[2] += new_bbox[0]
    new_bbox[3] += new_bbox[1]

    # update the tile annotation to use the tiled bbox 
    tile_annotation['bbox'] = new_bbox
    tile_annotation['bbox_label'] = tile_annotation['category_id']

    del tile_annotation['centroid']
    del tile_annotation['position']
    del tile_annotation['velocity']
    del tile_annotation['acceleration']
    del tile_annotation['category_id']
    return tile_annotation