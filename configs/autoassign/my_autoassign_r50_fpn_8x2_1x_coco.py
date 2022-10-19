# We follow the original implementation which
# adopts the Caffe pre-trained backbone.
_base_ = [
    './coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='AutoAssign',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
#img_norm_cfg = dict(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(mean=[144.578, 107.304, 90.519], std=[78.271, 63.2, 56.992], to_rgb=True)

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(norm_decay_mult=0.))

#total_epochs = 12
# learning policy
lr_config = dict(
    policy='step',  # 优化策略
    warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=50,  # 在初始的500次迭代中学习率逐渐增加 # 50
    warmup_ratio=0.001,  # 起始的学习率 # 0.001
    step=[8*16, 11*16], # [8, 11]*[1, 16]  8*16, 11*16
)
# total_epochs = 100  # 最大epoch数
runner = dict(type='EpochBasedRunner', max_epochs=12*16) # 12 12*16

checkpoint_config = dict(interval=1) # 每1个epoch存储一次模型

evaluation = dict(start=8*16, interval=1, metric=['bbox', 'segm']) # [8, 8*16]