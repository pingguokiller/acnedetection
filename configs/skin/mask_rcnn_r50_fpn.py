# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50, # 网络层数 # 50 101
        num_stages=4,  # resnet的stage数量
        out_indices=(0, 1, 2, 3),  #+ 输出的stage的序号
        frozen_stages=-1,  # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；
        # 如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # resnet50，输入的各个stage的通道数 # [256, 512, 1024, 2048]  38， 34：[64, 128, 256, 512]
        out_channels=256,  # 输出的特征层的通道数
        num_outs=4, # 输出的特征层的数量
        asff=False,
    ),
    rpn_head=dict(
        type='RPNHeadNwd', # RPNHeadNwd RPNHead
        is_rpn_nwd=True, # 是否采用 is_rpn_nwd loss
        patchwise_attention=False, # 是否采用 patchwise_attention
        in_channels=256,  # RPN网络的输入通道数
        feat_channels=256,  # 特征层的通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[8, 16], # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
            scales=[8], # default: 8
            ratios=[1/2, 1.0, 2.0],  # anchor的宽高比
            strides=[4, 8, 16, 32]),  # 在每个特征层上的anchor的步长（对应于原图） # , default: [4, 8, 16, 32] 64
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],  # 均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # RoIExtractor类型
            # XXX: ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
            # sampling_ratio(int): number of inputs samples to take for each
            # output sample. 0 to take samples densely for current models.
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            # roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=128), 
            out_channels=256,  # 输出通道数
            # 特征图的步长
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',  # 全连接层类型
            in_channels=256,  # 输入通道数
            fc_out_channels=1024,  # 输出通道数
            roi_feat_size=7,  # ROI特征层尺寸
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，
            # 续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
            reg_class_agnostic=False,  
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0) #是否采用全部level的feature map
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=10,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',  # RCNN网络正负样本划分
                pos_iou_thr=0.5, # default: 0.7 # 大于该阈值的 dt就分配上
                neg_iou_thr=0.3, # 与该gt的IOU在该阈值以下的 dt设置为0， 标志该dt与该gt并不重合
                # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，
                # 则忽略所有的anchors，否则保留最大IOU的anchor
                min_pos_iou=0.3,
                match_low_quality=True,
                # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',  # 正负样本提取器类型
                num=256,  # 需提取的正负样本数量之后
                pos_fraction=0.5,  # 正样本比例
                neg_pos_ub=-1,  # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
                add_gt_as_proposals=False),  # 把ground truth加入proposal作为正样本
            allowed_border=-1,
            pos_weight=-1,  # 正样本权重，-1表示不改变原始的权重
            debug=False,
            iou_label=False),  # RPN iou_label
        rpn_proposal=dict(
#             nms_across_levels=False,
#             nms_pre=2000,
#             nms_post=1000,
#             max_num=1000,
#             nms_thr=0.7,
#             min_bbox_size=0
            nms_pre=2000, # default:2000 else：800
            max_per_img=1000, # 一个图像最多有多少个proposal default:1000 else：800
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512, # default: 512
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
#             nms_across_levels=False,  # 在所有的fpn层内做nms
#             nms_pre=1000,  # 在nms之前保留的的得分最高的proposal数量
#             nms_post=1000,  # 在nms之后保留的的得分最高的proposal数量
#             max_num=1000,  # 在后处理完成之后保留的proposal数量
#             nms_thr=0.7,  # nms阈值
#             min_bbox_size=0),  # 最小bbox尺寸
            nms_pre=2000, # small:2000 # 2000>1000(0.004)
            max_per_img=1000, # small:1000 # default:1000 曾经好过：800
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=10),
        rcnn=dict(
            score_thr=0.05, # ROI bbox [bbox_num, 11]
            # score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.5), # 0.5 会尽可能地滤掉无用
            max_per_img=200,  # # small:100 big:600 max_per_img表示最终输出的det bbox数量 曾经好过：100
            mask_thr_binary=0.5)))
