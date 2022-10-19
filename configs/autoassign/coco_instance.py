# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/zhangjw/research/skin_data/coco/'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[144.578, 107.304, 90.519], std=[78.271, 63.2, 56.992], to_rgb=True)
# 用于caffe
img_norm_cfg = dict(
    mean=[144.578, 107.304, 90.519], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        scale_factor = 1.0, # 没有img_scale就需要有scale_factor
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/train_whole.json', # train_crop_1024 # train_whole #
    #     img_prefix=data_root + 'images/train_whole/',  # train_crop_1024 # train_whole #
    #     pipeline=train_pipeline),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_crop_1024.json',
        img_prefix=data_root + 'images/train_crop_1024/',
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=16,
        type=dataset_type,
        ann_file=data_root + 'annotations/val_crop_1024.json', # val_whole # val_crop_1024 #
        img_prefix=data_root + 'images/val_crop_1024/', # val_whole # val_crop_1024 #
        pipeline=test_pipeline),
    # test=dict(
    #     samples_per_gpu=1,
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_test2017_whole.json', # 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'images/test2017_whole/', #'images/val2017/',
    #     pipeline=test_pipeline)
    test=dict(
        samples_per_gpu=1,
        type=dataset_type,
        ann_file=data_root + 'annotations/all_whole.json',
        img_prefix=data_root + 'images/all_whole/',
        pipeline=test_pipeline),
)
evaluation = dict(interval=1, metric='bbox', classwise=True)
