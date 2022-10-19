dataset_type = 'CocoDataset'
data_root = '/home/zhangjw/research/skin_data/coco/'

# TODO: mean 和 std
img_norm_cfg = dict(
    mean=[144.578, 107.304, 90.519], std=[78.271, 63.2, 56.992], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[150.88884, 111.05136, 95.698715], std=[75.45097, 61.343353, 56.674812], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[150.768, 110.990, 95.532], std=[75.509, 61.466, 56.842], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='MyTransform', crop_size=1024), #  attention-based sample # default: 1024
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(2300, 1724), ratio_range=(1.0, 1.0), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute', allow_negative_crop=False),

    # TODO: 没有flip会报错 KeyError: 'flip'
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32), # size_divisor：保证图像大小为32的倍数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'position']), #, 'position'
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#         img_scale=[(2300, 1724)],
        scale_factor = 1.0, # 没有img_scale就需要有scale_factor
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute'),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_crop_1024.json', # train_crop_1024 # train_whole #
        img_prefix=data_root + 'images/train_crop_1024/',  # train_crop_1024 # train_whole #
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=2,
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


evaluation = dict(start=1, interval=1, metric=['bbox', 'segm']) # [8, 8*16]
