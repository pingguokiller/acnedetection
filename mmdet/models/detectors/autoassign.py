from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class AutoAssign(SingleStageDetector):
    """Implementation of `AutoAssign: Differentiable Label Assignment for Dense
    Object Detection <https://arxiv.org/abs/2007.03496>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AutoAssign, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

        # 1: ./models/dense_heads/fcos_head.py FCOSHead forward()
        # 2: ./models/dense_heads/autoassign_head.py AutoAssignHead forward_single()
        # 3: ./models/dense_heads/anchor_free_head.py AnchorFreeHead forward_single()
        # 4: ./models/dense_heads/autoassign_head.py AutoAssignHead loss()