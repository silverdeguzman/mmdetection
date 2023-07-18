_base_ = './centernet_r18-6ch_8xb16-crop512-140e_coco.py'

model = dict(backbone=dict(in_channels=12))
