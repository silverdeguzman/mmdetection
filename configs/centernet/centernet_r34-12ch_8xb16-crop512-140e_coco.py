_base_ = './centernet_r34-12ch-dcnv2_8xb16-crop512-140e_coco.py'

model = dict(neck=dict(use_dcn=False))
