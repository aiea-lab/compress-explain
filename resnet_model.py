import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
# from utils.conv_type import PretrainConv
# PretrainConv = nn.Conv2d
from torch.distributions.normal import Normal
from layers import SubnetConv, SubnetLinear, testConv

from load_model import PretrainConv

norm_mean, norm_var = 0.0, 1.0

class Mask(nn.Module):
    def __init__(self, init_value=[1], planes=None):
        super().__init__()
        self.planes = planes
        self.weight = torch.nn.Parameter(torch.Tensor(init_value))

    def forward(self, input):
        weight = self.weight

        if self.planes is not None:
            weight = self.weight[None, :, None, None]
        
        return input * weight

class Conv2d_KSE(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=True, G=4, T=0):
        super(Conv2d_KSE, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.isbias = bias
        self.T = T

        if G == 0:
            if output_channels >= input_channels:
                self.G = input_channels
            else:
                self.G = math.ceil(input_channels/output_channels)
        else:
            self.G = G
        self.group_num = self.G
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.Tensor(input_channels), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def __repr__(self):
        return self.__class__.__name__ \
               + "({" + str(self.input_channels) \
               + "}, {" + str(self.output_channels) \
               + "}, kernel_size={" + str(self.kernel_size) + "}, stride={" + \
               str(self.stride) + "}, padding={" + str(self.padding) + "})"

    def forward(self, input):
        # transform cluster and index into weight for training
        cluster_weights = []
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            clusters = cluster.permute(1, 0, 2, 3).contiguous().view(
                self.cluster_num[g] * cluster.shape[1], self.kernel_size, self.kernel_size)
            cluster_weight = clusters[
                self.__getattr__("cluster_indexs_" + str(g)).data].contiguous().view(
                self.output_channels, cluster.shape[1], self.kernel_size, self.kernel_size)

            cluster_weights.append(cluster_weight)

        if len(cluster_weights) == 0:
            weight = self.full_weight
        else:
            weight = torch.cat((self.full_weight, torch.cat(cluster_weights, dim=1)), dim=1)

        return F.conv2d(torch.index_select(input, 1, self.channels_indexs), weight, self.bias,
                        stride=self.stride,
                        padding=self.padding)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv7x7(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


def preconv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return PretrainConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def preconv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return PretrainConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups, dilation=dilation)

def preconv7x7(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return PretrainConv(
        in_planes,
        out_planes,
        kernel_size=7,
        stride=stride,
        padding=3,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def kseconv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return Conv2d_KSE(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)

def kseconv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return Conv2d_KSE(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def hydraconv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return SubnetConv(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, groups=groups, dilation=dilation, padding=dilation)

def hydraconv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return SubnetConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def hydraconv7x7(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return SubnetConv(
        in_planes,
        out_planes,
        kernel_size=7,
        stride=stride,
        padding=3,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lottery: bool = False,
        kse: bool = False,
        mask: bool = False,
        hydra: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if lottery:
            self.conv3x3 = preconv3x3
            self.conv1x1 = preconv1x1
        elif kse:
            self.conv3x3 = kseconv3x3
            self.conv1x1 = kseconv1x1
        elif hydra:
            self.conv3x3 = hydraconv3x3
            self.conv1x1 = hydraconv1x1
        else: 
            self.conv3x3 = conv3x3
            self.conv1x1 = conv1x1
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        m = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask = Mask(m) if mask else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.mask is not None:
            out = self.mask(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lottery: bool = False,
        kse: bool = False,
        hydra: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if lottery:
            self.conv3x3 = preconv3x3
            self.conv1x1 = preconv1x1
        elif kse:
            self.conv3x3 = kseconv3x3
            self.conv1x1 = kseconv1x1
        elif hydra:
            self.conv3x3 = hydraconv3x3
            self.conv1x1 = hydraconv1x1
        else: 
            self.conv3x3 = conv3x3
            self.conv1x1 = conv1x1
        self.conv1 = self.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = self.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = self.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lottery: bool = False,
        kse: bool = False,
        mask: bool = False,
        attribute_preserve: bool = False,
        hydra: bool = False,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.lottery = lottery
        self.kse = kse
        self.mask = mask
        self.attribute_preserve = attribute_preserve
        self.hydra = hydra
        self.groups = groups
        self.base_width = width_per_group
        if lottery:
            self.conv3x3 = preconv3x3
            self.conv1x1 = preconv1x1
            self.conv7x7 = preconv7x7
        elif kse:
            self.conv3x3 = kseconv3x3
            self.conv1x1 = kseconv1x1
            self.conv7x7 = conv7x7
        elif hydra:
            self.conv3x3 = hydraconv3x3
            self.conv1x1 = hydraconv1x1
            self.conv7x7 = hydraconv7x7
        else: 
            self.conv3x3 = conv3x3
            self.conv1x1 = conv1x1
            self.conv7x7 = conv7x7
        self.conv1 = self.conv7x7(3, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, PretrainConv) or isinstance(m, Conv2d_KSE):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, lottery=self.lottery, mask=self.mask, kse=self.kse, hydra=self.hydra
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    lottery=self.lottery,
                    mask=self.mask,
                    kse=self.kse,
                    hydra=self.hydra,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, mode=None, TS=None, grad_out=None, erase_channel=None) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if mode == "eval" or self.attribute_preserve == False or x.requires_grad == False:
            pass
        else:
            x.retain_grad()
        out = self.layer4(x)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.attribute_preserve and x.requires_grad != False:
            if mode == 'swa':
                if not isinstance(grad_out, torch.autograd.Variable):
                    ind = out.data.max(1)[1]
                    grad_out = out.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)

                swa = self.cal_grad(out, grad_out, TS, [x], erase_channel)
                return out, swa, grad_out
            else:
                return out, [x], None

        return out

    def forward(self, x: Tensor, mode=None, TS=None, grad_out=None, erase_channel=None) -> Tensor:
        return self._forward_impl(x, mode, TS, grad_out, erase_channel)
    
    def cal_grad(self, out, grad_out, TS, feature, erase_channel):
        attributions = []
        if TS == 'Teacher':
            out.backward(grad_out, retain_graph=True)
            feat = feature[0].clone().detach()
            grad = feature[0].grad.clone().detach()
            if erase_channel is not None:
                for erase_c in erase_channel:
                        feat[:,erase_c,:,:] = 0
            linear = torch.sum(torch.sum(grad, 3, keepdim=True), 2, keepdim=True)        # batch, 512, 1, 1
            channel = linear * feat                                                      # batch, 512, 7, 7
            swa = torch.sum(channel, 1, keepdim=True)                                  # batch, 1, 7, 7
            attributions.append(F.relu(swa))
            return attributions

        elif TS == 'Student':
            out.backward(grad_out, create_graph=True)
            linear = torch.sum(torch.sum(feature[0].grad, 3, keepdim=True), 2, keepdim=True)        # batch, 512, 1, 1
            channel = linear * feature[0]                                                           # batch, 512, 7, 7
            swa = torch.sum(channel, 1, keepdim=True)                                             # batch, 1, 7, 7
            attributions.append(F.relu(swa))
        return attributions

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet34(*, weights: Optional[Tensor] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    # weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)

