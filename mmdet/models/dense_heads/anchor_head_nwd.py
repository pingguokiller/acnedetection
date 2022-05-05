import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
import numpy as np
from mmdet.utils import get_root_logger

# 计算nwd
def cal_nwd(bbox_decoded_pred, bbox_decoded_targets):

    na = bbox_decoded_pred[0].new_zeros(*bbox_decoded_pred.shape)
    nb = bbox_decoded_pred[0].new_zeros(*bbox_decoded_pred.shape)
    na[:, 0] = (bbox_decoded_pred[:, 0] + bbox_decoded_pred[:, 2]) / 2
    na[:, 1] = (bbox_decoded_pred[:, 1] + bbox_decoded_pred[:, 3]) / 2
    na[:, 2] = (bbox_decoded_pred[:, 2] - bbox_decoded_pred[:, 0]) / 2
    na[:, 3] = (bbox_decoded_pred[:, 3] - bbox_decoded_pred[:, 1]) / 2

    nb[:, 0] = (bbox_decoded_targets[:, 0] + bbox_decoded_targets[:, 2]) / 2
    nb[:, 1] = (bbox_decoded_targets[:, 1] + bbox_decoded_targets[:, 3]) / 2
    nb[:, 2] = (bbox_decoded_targets[:, 2] - bbox_decoded_targets[:, 0]) / 2
    nb[:, 3] = (bbox_decoded_targets[:, 3] - bbox_decoded_targets[:, 1]) / 2

    C = 32
    NWD = torch.exp(-1 * ((na - nb).pow(2).sum(-1).sqrt()) / C)
    return NWD

# iou
def cal_iou(bboxes1, bboxes2, mode='iou', eps=1e-6):
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
    rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

    wh = torch.clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - overlap


    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode =='iou':
        return ious
    else:
        enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
        enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])

        # calculate gious
        enclose_wh = torch.clamp(enclosed_rb - enclosed_lt, min=0)

        if mode == 'giou':
            enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
            enclose_area = torch.max(enclose_area, eps)
            gious = ious - (enclose_area - union) / enclose_area
            return gious
        else: # diou
            cw = enclose_wh[..., 0]
            ch = enclose_wh[..., 1]
            c2 = cw ** 2 + ch ** 2 + eps

            b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
            b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
            b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
            b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]

            left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
            right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            rho2 = left + right

            # DIoU
            dious = ious - rho2 / c2
            return dious


@HEADS.register_module()
class AnchorHeadNwd(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 patchwise_attention=False,
                 is_rpn_nwd=True):
        super(AnchorHeadNwd, self).__init__(init_cfg)
        self.patchwise_attention = patchwise_attention
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')

            if not hasattr(self.train_cfg, 'iou_label'):
                self.train_cfg.iou_label = False

            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        # 表示每一层每一个 有多少个anchors
        self.num_anchors = self.anchor_generator.num_base_anchors[0] # [3, 3, 3, 3, 3]

        # 20211208
        self.is_rpn_nwd = is_rpn_nwd

        self._init_layers()

    # RPNHead forward 的第二步
    def forward(self, feats, **kwargs):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, **kwargs)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    # 对每张图片进行get_target
    def _get_targets_single(self, flat_anchors, valid_flags,
        gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        # MaxIoUAssigner
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore,None if self.sampling else gt_labels)
        # RandomSampler
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        # sampling_result 里面的 pos_inds 存储了 拥有 正样本的 anchor_idxs
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        # 全部设置为1 在 cross_entropy_loss.py 的 _expand_onehot_labels会将label反转一次
        if self.train_cfg.iou_label:
            labels = anchors.new_full((num_valid_anchors, ), 0)
        else:
            labels = anchors.new_full((num_valid_anchors, ), self.num_classes, dtype=torch.long)

        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float) # 默认是0
        # 从anchor里面随机抽出一些正样本 、 负样本
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox: # True
                # pos_bboxes:实际上是 anchors
                # pos_gt_bboxes：是对应的gt_bboxes

                # pos_bbox_targets：实际上是根据 anchors gt_bboxes 算出来的对应的 delta_x,y,w,h 目标；  而我们希望用网络预测出来跟这个 delta_x,y,w,h 目标，进行损失计算
                # DeltaXYWHBBoxCoder encode
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
                #neg_bbox_targets = self.bbox_coder.encode(sampling_result.neg_bboxes, sampling_result.neg_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                #neg_bbox_targets = sampling_result.neg_gt_bboxes

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0  # 只有正样本才计算loss # 灵动20211110 这个地方可以对小目标进行加成
            # bbox_targets[neg_inds, :] = neg_bbox_targets
            # bbox_weights[neg_inds, :] = 1.0  # 只有正样本才计算loss # 灵动202129 这个地方可以对小目标进行加成

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                if self.train_cfg.iou_label:
                    labels[pos_inds] = assign_result.max_overlaps[pos_inds] # 灵动20211110 将IOU设置为label
                else:
                    labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

            if self.train_cfg.pos_weight <= 0: # True
                label_weights[pos_inds] = 1.0 # assign_result.max_overlaps[pos_inds] # 灵动20211110 这个地方可以设置一个IOU
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0: # True
            if self.train_cfg.iou_label:
                labels[neg_inds] = assign_result.max_overlaps[neg_inds] # 灵动20211110 将IOU设置为label

            label_weights[neg_inds] = 1.0 # this time1 - assign_result.max_overlaps[neg_inds] # 1.0 this time

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)

    # 这个函数决定了哪些anchor参与loss的计算
    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None,
        gt_labels_list=None, label_channels=1, unmap_outputs=True, return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image. # 这里将gt_bbox原始坐标 转为
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list, concat_valid_flag_list,
            gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list,
            img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results: # False
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    ###########################################################################################
    #
    #   valid_rpn 验证RPN的有效性：(参考函数_get_targets_single中的用法)
    #
    ###########################################################################################
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def valid_rpn(self, proposal_list, gt_bboxes, img_metas):
    #     """
    #     【20210825-INCREASE】 验证RPN的有效性：(参考函数_get_targets_single中的用法)
    #     GT有没有取到: RPN输出的数据中, 每个GT上有没有roi, 有的话有多少个, iou是多少
    #     ROIs类别是否判断正确: 由于RPN默认只区分fg/bg, 暂不考虑
    #     Args:
    #         proposal_list: batch_size * [num_rois, [4; cls_scores]]
    #         gt_bboxes: batch_size * [num_gts, 4]
    #         img_metas: {}
    #     Returns:
    #
    #     """
    #     logger = get_root_logger('INFO')
    #     log_str = ""
    #     num_imgs = len(img_metas)
    #
    #     # 与GT进行匹配，考虑最简单的情况，取消了assign的后2个参数; 由于assign返回的是字典类型，所以不能使用multi_apply
    #     # assign_results = multi_apply(self.assigner.assign, proposal_anchor_list, gt_bboxes)
    #     for i in range(num_imgs):
    #         assign_results = self.assigner.assign(proposal_list[i][:, :4], gt_bboxes[i]);
    #         gt_inds = assign_results.gt_inds
    #         max_overlaps = assign_results.max_overlaps
    #         # 是不是每一个GT都被取到了
    #         num_gts = gt_bboxes[i].shape[0]
    #         log_str += f'\ti:g{num_gts}'
    #         cnt_miss_gt = 0
    #         for j in range(num_gts):
    #             j_inds = (gt_inds == j)
    #             if j_inds.sum().item() == 0:
    #                 cnt_miss_gt += 1
    #                 continue
    #             # j_cnt07 = (max_overlaps[j_inds] > 0.7).sum().item()
    #             # j_cnt05 = (max_overlaps[j_inds] > 0.5).sum().item()
    #             # log_str += f'{j_cnt05 - j_cnt07}-{j_cnt07}_'
    #
    #         cnt07 = (max_overlaps > 0.7).sum().item()
    #         cnt05 = (max_overlaps > 0.5).sum().item()
    #         if cnt_miss_gt > 0:
    #             log_str += f'\tm{cnt_miss_gt}'
    #         else:
    #             log_str += '\t'
    #         log_str += f'\t{cnt05}\t{cnt07}'
    #     logger.info(log_str)



    def loss_single(self, cls_score, nwd_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """


        labels = labels.reshape(-1) # # 256*256*3*3
        label_weights = label_weights.reshape(-1) # [256*256*3*3, 128*128*3*3]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if self.is_rpn_nwd:
            nwd_score = nwd_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) # # 256*256*3*3
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples, iou_label=self.train_cfg.iou_label)  # 768


        # regression loss # 只有正样本
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        # L1距离
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)

        if not self.is_rpn_nwd:
            return loss_cls, torch.tensor(0.0).to(loss_cls.device), loss_bbox

        # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
        # is applied directly on the decoded bounding boxes, it
        # decodes the already encoded coordinates to absolute format.
        anchors = anchors.reshape(-1, 4)

        # 先将labels进行normallize; 然后用L1loss通过IOU回归损失去计算
        label_weights = torch.unsqueeze(label_weights, 1)
        label_weights[:, 0] = bbox_weights[:, 0]
        idxs = label_weights > 0
        idxs = idxs[:, 0]

        # 预测
        nwd_score = nwd_score[idxs]
        nwd_score = torch.sigmoid(nwd_score)  # 预测 nwd

        if anchors[idxs].shape[0] == 0:
            loss_nwd = bbox_pred.new_tensor(0)
        else:
            # 获取 labels_nwd
            loss_type = 'nwd' # nwd iou giou diou
            if loss_type == 'nwd':
                bbox_decoded_pred = self.bbox_coder.decode(anchors[idxs], bbox_pred[idxs])
                bbox_decoded_targets = self.bbox_coder.decode(anchors[idxs], bbox_targets[idxs])

                if anchors[idxs].shape[0] == 0:
                    pass

                # cal_nwd
                NWD = cal_nwd(bbox_decoded_pred, bbox_decoded_targets)
                labels_nwd = torch.unsqueeze(NWD, 1)
            else: # GIOU
                # bbox_targets 其中非0的就是 分配给对应 IOU > 0.5的anchor的gt_bbox
                # anchors中对应位置就是bbox, 两者在一起可以直接计算 iou giou, 这个时候一一计算就行
                bbox_decoded_pred = self.bbox_coder.decode(anchors[idxs], bbox_pred[idxs]) #anchors[idxs]
                bbox_decoded_targets = self.bbox_coder.decode(anchors[idxs], bbox_targets[idxs])

                if anchors[idxs].shape[0] == 0:
                    pass

                labels_nwd = cal_iou(bbox_decoded_pred, bbox_decoded_targets, mode=loss_type)
                labels_nwd = torch.unsqueeze(labels_nwd, 1)


            #pos_and_neg_num = nwd_score.shape[0]
            # loss=-ylog(p)-(1-y)log(1-p)
            loss_nwd = -(1-(labels_nwd-nwd_score).abs()).log().sum()/num_total_samples
            #loss_nwd = self.loss_bbox(nwd_score, NWD, label_weights, avg_factor=num_total_samples)

        # 约束 预测的 cls_core 和 算出来的 nwd是一致的
        return loss_cls, loss_nwd, loss_bbox  #+ 0.1*(loss_width + loss_x + loss_height + loss_y)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             nwd_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device #

        # 把所有的 anchrolist集合到一起了,
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)

        if cls_reg_targets is None:
            return None

        #
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        # 虽然这里虽然有很多，但是实际上通过weights等，只计算了256个采样框的损失
        losses_cls, losses_nwd, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            nwd_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_nwd=losses_nwd, loss_bbox=losses_bbox)

    # 这个地方不会走进来的， get_bboxes 在子类 RPNHead中有实现
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.

        Example:
            >>> import mmcv
            >>> self = AnchorHeadNwd(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list

    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.

        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
