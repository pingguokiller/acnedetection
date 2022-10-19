_base_ = [
    '../_base_/models/skin_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/skin_instance_bbox.py',
    '../_base_/schedules/skin_schedule_2x.py', '../_base_/default_runtime.py'
]
