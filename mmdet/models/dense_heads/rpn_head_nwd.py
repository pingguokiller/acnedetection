import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from ..builder import HEADS
from .anchor_head import AnchorHead
from .anchor_head_nwd import AnchorHeadNwd


@HEADS.register_module()
class RPNHeadNwd(AnchorHeadNwd):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 **kwargs):
        super(RPNHeadNwd, self).__init__(1, in_channels, init_cfg=init_cfg, **kwargs)



    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_head = nn.Sequential(
           nn.Conv2d(self.in_channels, self.feat_channels, 1, padding=0, stride=1, bias=True),
           nn.GroupNorm(32, self.feat_channels),
           nn.ReLU(inplace=True)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1, stride=1, bias=True),
            nn.GroupNorm(32, self.feat_channels),
            nn.ReLU(inplace=True)
        )

        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1) # , 2
        self.rpn_nwd = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1) # , 2
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1) # , 2

    def forward_single(self, x, **kwargs): #
        """Forward feature map of a single scale level."""
        cls_head_feat = self.cls_head(x)  # [B, 256, 256, 256]
        rpn_cls_score = self.rpn_cls(cls_head_feat) # [B,  3, [256, 128, 64, 32], 256]

        bbox_head_feat = self.bbox_head(x)  # [B, 256, 256, 256]
        if self.is_rpn_nwd:
            rpn_nwd_score = self.rpn_nwd(bbox_head_feat) # [B,  3, [256, 128, 64, 32], 256]
        else:
            rpn_nwd_score = None

        rpn_bbox_pred = self.rpn_reg(bbox_head_feat) # [B, 12, [256, 128, 64, 32], 256]
        return rpn_cls_score, rpn_nwd_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             nwd_scores,
             bbox_preds,
             gt_bboxes,
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
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNHeadNwd, self).loss(
            cls_scores,
            nwd_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_nwd=losses['loss_nwd'], loss_rpn_bbox=losses['loss_bbox'])
        #return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    @force_fp32(apply_to=('cls_scores', 'bbox_preds')) # this way # test 的时候也是先RPN获取feature map, 然后proposal
    def get_bboxes(self,
                   cls_scores,
                   cls_nwds,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   return_rpn_bfnms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
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
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores) # 5
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)] # [256, 128, 64, 32, 16]

        # 各个featuremap 提出的所有anchor
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)

        result_list = []
        rpn_bfnms_list = []
        for img_id in range(len(img_metas)):  #遍历每张图
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels) # *cls_nwds[i][img_id].detach()
            ]

            if self.is_rpn_nwd:
                cls_nwds_list = [
                    cls_nwds[i][img_id].detach() for i in range(num_levels) # *cls_nwds[i][img_id].detach()
                ]
            else:
                cls_nwds_list = None

            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if return_rpn_bfnms:
                proposals, rpn_bfnms = self._get_bboxes_single(cls_score_list, cls_nwds_list, bbox_pred_list, mlvl_anchors, img_shape,
                    scale_factor, cfg, rescale, return_bfnms=return_rpn_bfnms) # cfg: train_cfg.rpn_proposal
                rpn_bfnms_list.append(rpn_bfnms)
            else:
                proposals = self._get_bboxes_single(cls_score_list, cls_nwds_list, bbox_pred_list, mlvl_anchors, img_shape,
                    scale_factor, cfg, rescale, return_bfnms=return_rpn_bfnms)  # cfg: train_cfg.rpn_proposal

            result_list.append(proposals) # 每个最多只有1000个

        # 是否返回rpn bfnms
        if return_rpn_bfnms:
            return result_list, rpn_bfnms_list
        else:
            return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           cls_nwds,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           return_bfnms=False):
        """Transform outputs for a single batch item into bbox predictions.

          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)): # 针对每一层，逐级
            # 筛出某一层的proposal
            # if idx!=3:
            #    continue
            rpn_cls_score = cls_scores[idx] # 3, 256, 256
            if self.is_rpn_nwd:
                rpn_nwd_score = cls_nwds[idx] # 3, 256, 256

            rpn_bbox_pred = bbox_preds[idx] # 12, 256, 256
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)

            if self.is_rpn_nwd:
                rpn_nwd_score = rpn_nwd_score.permute(1, 2, 0)

            if self.use_sigmoid_cls: # True
                rpn_cls_score = rpn_cls_score.reshape(-1) # 256*256*3

                if self.is_rpn_nwd:
                    cls_alpha = 1

                    # feature_size = rpn_nwd_score.shape[0]
                    # cls_alpha_list =[1, 1, 0, 0]
                    # if feature_size==256:
                    #     cls_alpha = cls_alpha_list[0]
                    # elif feature_size==128:
                    #     cls_alpha = cls_alpha_list[1]
                    # elif feature_size==64:
                    #     cls_alpha = cls_alpha_list[2]
                    # else: # 32
                    #     cls_alpha = cls_alpha_list[3]

                    rpn_nwd_score = rpn_nwd_score.reshape(-1) # 256*256*3
                    #scores = rpn_cls_score.sigmoid() * rpn_nwd_score.sigmoid().pow(cls_alpha) # featuremap 每个点每个anchor 的分类 score
                    scores = (rpn_cls_score.sigmoid().pow(cls_alpha) * rpn_nwd_score.sigmoid().pow(2-cls_alpha)).sqrt() # featuremap 每个点每个anchor 的分类 score
                else:
                    scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4) # 256*256*3, 4

            # 各层的anchors
            anchors = mlvl_anchors[idx]

            # 只取前2000个anchors这导致了很多小的的anchor分数低的anchor过去了
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]   # 选择分数最高的2000个anchor, 这里实际上是预测前景和背景
                i = 0

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            # 2000个
            level_ids.append(scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores) # 2000+2000+2000+2000+768 = 8768
        anchors = torch.cat(mlvl_valid_anchors) # 8768*4  anchors
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds) # 8768*4  # RPN预测出来的bbox
        # 根据预测偏差值 进行回归
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape) # proposals: [499.8211, 979.9355, 522.2711, 1023.9187]
        ids = torch.cat(level_ids)
        # anchor: [500.6863, 981.3726, 523.3137, 1026.6274] rpn_bbox_pred: [-0.0422, -0.0458, -0.0079, -0.0285]
        if cfg.min_bbox_size >= 0: # 对最小bbox_size有限制
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0: # 8768*4  =>  [3593, 5] # NMS 之后再进行选择
            # test: {'type': 'nms', 'iou_threshold': 0.7} # ids: level_ids
            dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        # 是否返回 NMS之前的
        if return_bfnms:
            return dets[:cfg.max_per_img], [proposals, scores, ids, mlvl_anchors, [cls_scores, cls_nwds], bbox_preds]
        else:
            return dets[:cfg.max_per_img] # 最大预测1000个

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        cfg = copy.deepcopy(self.test_cfg)

        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        batch_size = cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(
            cfg.nms_pre, device=cls_scores[0].device, dtype=torch.long)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(-1)[..., 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4)
            anchors = mlvl_anchors[idx]
            anchors = anchors.expand_as(rpn_bbox_pred)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, rpn_bbox_pred.shape[1])
            if nms_pre > 0:
                _, topk_inds = scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                # Mind k<=3480 in TensorRT for TopK
                transformed_inds = scores.shape[1] * batch_inds + topk_inds
                scores = scores.reshape(-1, 1)[transformed_inds].reshape(
                    batch_size, -1)
                rpn_bbox_pred = rpn_bbox_pred.reshape(
                    -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                anchors = anchors.reshape(-1, 4)[transformed_inds, :].reshape(
                    batch_size, -1, 4)
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_rpn_bbox_pred = torch.cat(mlvl_bbox_preds, dim=1)
        batch_mlvl_proposals = self.bbox_coder.decode(
            batch_mlvl_anchors, batch_mlvl_rpn_bbox_pred, max_shape=img_shapes)

        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        batch_mlvl_scores = batch_mlvl_scores.unsqueeze(2)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        dets, _ = add_dummy_nms_for_onnx(batch_mlvl_proposals,
                                         batch_mlvl_scores, cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
