from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    # @abstractmethod
    # def valid_rpn(self, **kwargs):
    #     """
    #     【20210824-INCREASE】 验证RPN的有效性：
    #     RPN输出的数据中，每个GT（只分fg/bg）上有没有roi，有的话有多少个，iou是多少
    #     """
    #     pass

    # RPNHead 的训练，会进入这里
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        # [ 分类 [B*ratio_num(3)*[256, 128, 64, 32, 16]] , 回归 [B*ratio_num(3*4)*[256, 128, 64, 32, 16]]  ]
        # 2： 进入 AnchorHead 的 forward
        kwargs['img_metas'] = img_metas
        outs = self(x, **kwargs) # 1
        if gt_labels is None: # RPN 只算前景后景预测的 损失
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) # gt_bboxes 这儿出现过一次 1*4 感觉不太对

        # 4: AnchorHead: loss
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore) # 计算loss 是不需要proposal参与的。。。，这块还需要看一下。
        if proposal_cfg is None: # train_cfg.rpn_proposal
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)

            # 【INCREASE】调试RPN的有效性
            # proposal_list: batch_size * [num_rois, [4, cls_scores]]
            #self.valid_rpn(proposal_list, gt_bboxes, img_metas)

            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)
