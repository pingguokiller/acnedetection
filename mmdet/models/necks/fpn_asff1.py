import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch

from ..builder import NECKS


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 asff=False): # 1
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.asff = asff

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # ASFF
        if asff:
            self.asff_1_0_conv = ConvModule(self.out_channels, self.out_channels, 1, stride=2, padding=0)
            self.asff_2_0_conv = ConvModule(self.out_channels, self.out_channels, 1, stride=2, padding=0)
            self.asff_2_1_conv = ConvModule(self.out_channels, self.out_channels, 1, stride=2, padding=0)

            compress_c = 32
            self.weight_level_0_0 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_0_1 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_0_2 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_levels_0 = ConvModule(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

            self.weight_level_1_0 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_1_1 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_1_2 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_levels_1 = ConvModule(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

            self.weight_level_2_0 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_2_1 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_level_2_2 = ConvModule(self.out_channels, compress_c, 1, 1)
            self.weight_levels_2 = ConvModule(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # [256, 512, 1024, 2048] channel越来越多，层级越来越高，feature map空间尺寸越来越小

        # build laterals  # 这4个测向卷积层 会将这个4个feature map 全部 卷积到  4 个 256 维度  [256, 256, 256, 256]
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)

        # 只有origin才哦组这条路
        if not self.asff:
            # build top-down path
            for i in range(used_backbone_levels - 1, 0, -1):
                # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
                #  it cannot co-exist with `size` in `F.interpolate`.
                if 'scale_factor' in self.upsample_cfg:
                    laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:] # 低下一层的空间尺寸，都是要大一号
                    laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)   # 上采样； 底层的 Feature map  是 经过 高层的feature map上采样插值 + 过来的。

        # ASFF 先只试用一下对 前3层的改造
        if self.asff:
            for asff_index in range(3): # 0, 1, 2
                # 高层
                if asff_index == 0: #  [B, 256, [256, 128, 64, 32]]
                    prev_shape = laterals[0].shape[2:]
                    level_0_resized = laterals[0]
                    level_1_resized = F.interpolate(laterals[1], size=prev_shape, **self.upsample_cfg)
                    level_2_resized = F.interpolate(laterals[2], size=prev_shape, **self.upsample_cfg)

                    level_0_weight_v = self.weight_level_0_0(level_0_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_1_weight_v = self.weight_level_0_1(level_1_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_2_weight_v = self.weight_level_0_2(level_2_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]

                    #
                    levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)  # [B, 64*3, 256, 256]
                    levels_weight = self.weight_levels_0(levels_weight_v)  # [B, 3, 256, 256]
                    levels_weight = F.softmax(levels_weight, dim=1)  # [B, 3, 256, 256]

                elif asff_index == 1: #
                    prev_shape = laterals[1].shape[2:]
                    level_0_resized = self.asff_1_0_conv(laterals[0]) # size: 256 => 128
                    level_1_resized = laterals[1]
                    level_2_resized = F.interpolate(laterals[2], size=prev_shape, **self.upsample_cfg)

                    level_0_weight_v = self.weight_level_1_0(level_0_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_1_weight_v = self.weight_level_1_1(level_1_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_2_weight_v = self.weight_level_1_2(level_2_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]

                    #print(level_0_weight_v.shape, level_1_weight_v.shape, level_2_weight_v.shape)
                    levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)  # [B, 64*3, 256, 256]
                    levels_weight = self.weight_levels_1(levels_weight_v)  # [B, 3, 256, 256]
                    levels_weight = F.softmax(levels_weight, dim=1)  # [B, 3, 256, 256]

                elif asff_index == 2: #
                    level_0_resized = F.max_pool2d(laterals[0], 3, stride=2, padding=1)
                    level_0_resized = self.asff_2_0_conv(level_0_resized)  # size: 256 => 64
                    level_1_resized = self.asff_2_1_conv(laterals[1])  # size: 128 => 64
                    level_2_resized = laterals[2]

                    level_0_weight_v = self.weight_level_2_0(level_0_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_1_weight_v = self.weight_level_2_1(level_1_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]
                    level_2_weight_v = self.weight_level_2_2(level_2_resized)  # [B, 256, 256, 256] => [B, 64, 256, 256]

                    #
                    levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)  # [B, 64*3, 256, 256]
                    levels_weight = self.weight_levels_2(levels_weight_v)  # [B, 3, 256, 256]
                    levels_weight = F.softmax(levels_weight, dim=1)  # [B, 3, 256, 256]


                fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                                    level_1_resized * levels_weight[:, 1:2, :, :] + \
                                    level_2_resized * levels_weight[:, 2:, :, :]  # [B, 256, [256, 128, 64]]

                laterals[asff_index] = fused_out_reduced

        # build outputs [B, 256, [256, 128, 64, 32]]
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) # FPN standard conv
        ] # [B, 256, [256, 128, 64, 32]]
        # part 2: add extra levels
        if self.num_outs > len(outs): # B*256*[256, 128, 64, 32] #featuremap 的空间尺寸都是没有变化的，主要是+了高层的上采样
            # use max pool to get more levels on top of outputs！！！
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
