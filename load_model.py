# import torch

class ModelInfo:
    def __init__(self, model, criterion, optimizer, name):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.name = name
        self.loss = None
        self.accuracy = None
        self.batch_size = None
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def __repr__(self):
        return self.name

# import models
import torch
import torch.nn as nn
import math
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions.normal import Normal
device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
norm_mean, norm_var = 0.0, 1.0

cfg = {'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
                
class_cfg = []

class VGG(nn.Module):
    def __init__(self, builder, vgg_name, layer_cfg=None, num_classes=100):
        super(VGG, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(builder,cfg[vgg_name])
        self.classifier = builder.conv1x1_fc(512, num_classes)
        #self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out.flatten(1)

    def _make_layers(self, builder, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [builder.conv3x3(in_channels,
                                     x),
                           builder.batchnorm(x),
                           builder.activation()]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class Builder(object):
    def __init__(self, conv_layer, bn_layer):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, bias=False):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=bias,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=bias,
            )
        else:
            return None

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )


    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv1x1_fc(self, in_planes, out_planes, stride=1):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride=stride, bias=True)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        return self.bn_layer(planes)

    def activation(self):
        return (lambda: nn.ReLU(inplace=True))()



LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask, b_mask):
        return b_mask

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class PretrainConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape, device=device))
        self.b_mask = nn.Parameter(torch.ones(self.weight.shape, requires_grad=False, device=device))

    def forward(self, x):
        mask = GetMask.apply(self.mask, self.b_mask)
        sparseWeight = mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
        
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        w = self.weight.detach().cpu()
        w = w.view(-1) #c_out * (c_in * k * k) -> 4 * (c_out * c_in * k * k / 4)
        m = self.mask.detach().cpu()
        m = m.view(-1)
        b_m = self.b_mask.detach().cpu()
        b_m = b_m.view(-1)
        #import pdb; pdb.set_trace()
        _, indice = torch.topk(torch.abs(w), int(w.size(0)*prune_rate), largest=False)
        b_m[indice] = 0
        m[indice] = 0.99
        self.b_mask = nn.Parameter(b_m.view(self.weight.shape), requires_grad=False)
        self.mask = nn.Parameter(m.view(self.weight.shape))


def get_builder(conv_layer=PretrainConv, bn_layer=LearnedBatchNorm):
    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer)

    return builder

def get_prune_rate(model, pr_cfg):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * pr_cfg[i])
            i += 1
    print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % ((all_params-prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))

def generate_pr_cfg(model):
    pr_cfg = []
    weights = []
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate") and name != 'fc' and name != 'classifier':
            conv_weight = module.weight.data.detach().cpu()   
            weights.append(conv_weight.view(-1)) 
    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1 - 0.9))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    #Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    pr_cfg.append(0)
    get_prune_rate(model, pr_cfg)

    return pr_cfg

def get_model(fn, conv_layer=PretrainConv, bn_layer=LearnedBatchNorm):
    arch = 'vgg19_cifar10'
    print("=> Creating model '{}'".format(arch))
    model = VGG(get_builder(conv_layer, bn_layer), num_classes=10, vgg_name='vgg19').to(device)
    ckpt = torch.load(fn, map_location=device)
    #import pdb;pdb.set_trace()
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device)
    return model

def get_model_pruned(fn):
    pr_cfg = []
    arch = 'vgg19_cifar10'
    print("=> Creating model '{}'".format(arch))
    model = VGG(get_builder(), num_classes=10, vgg_name='vgg19').to(device)
    ckpt = torch.load('./saved_models/vgg19_cifar10.pt', map_location=device)
    #import pdb;pdb.set_trace()
    model.load_state_dict(ckpt['state_dict'], strict=False)
    pr_cfg = generate_pr_cfg(model)
    model = VGG(get_builder(), num_classes=10, vgg_name='vgg19').to(device)
    i = 0
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(pr_cfg[i])
            # m.set_prune_rate(0.9)
            i += 1
    model = model.to(device)
    model.load_state_dict(torch.load(fn)["state_dict"])

    return model



import math
import torch
import torch.nn as nn

# from .init_utils import weights_init



defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class Mask(nn.Module):
    def __init__(self, init_value=[1], fc=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(init_value))
        self.fc = fc

    def forward(self, input):
        if self.fc:
            weight = self.weight
        else:
            weight = self.weight[None, :, None, None]
        return input * weight

class GraSP_VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True, is_sparse=False, is_mask=False):
        super(GraSP_VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.dataset = dataset
        num_classes = 10
        if is_sparse:
            self.feature = self.make_sparse_layers(cfg, batchnorm)
            self.classifier = nn.Linear(cfg[-1], num_classes)
        elif is_mask:
            self.feature = self.make_mask_layers(cfg, batchnorm)
            self.classifier = nn.Linear(cfg[-1], num_classes)
        else:
            self.feature = self.make_layers(cfg, batchnorm)
            self.classifier = nn.Linear(cfg[-1], num_classes)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def make_mask_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = PretrainConv(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def make_sparse_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                init_value = m
                layers += [Mask(init_value)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
    
def get_GraSP_VGG(fn, cfg=None):
    model = GraSP_VGG(cfg=cfg).to(device)
    ckpt = torch.load(fn)
    model.load_state_dict(ckpt, strict=False)
    return model.to(device)

def get_pretrain_grasp(fn):
    model = GraSP_VGG().to(device)
    ckpt = torch.load(fn).state_dict()
    model.load_state_dict(ckpt, strict=False)
    return model.to(device)

def get_GAL_VGG(fn):
    model = GraSP_VGG(is_sparse=True).to(device)
    ckpt = torch.load(fn)
    model.load_state_dict(ckpt['state_dict_s'], strict=True)
    return model.to(device)

def get_lottery_VGG(fn):
    model = GraSP_VGG(is_mask=True).to(device)
    ckpt = torch.load(fn)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.to(device)

def get_prune_scratch_VGG(fn, cfg=None):
    cpkt = torch.load(fn)
    cfg = []
    for i in range(20):
        if 'feature.{}.conv.weight'.format(i) in cpkt.keys():
            cfg.append(cpkt['feature.{}.conv.weight'.format(i)].shape[0])
        if i in [2, 4, 9, 14]:
            cfg.append('M')
    new_dict = {}
    prev = 'conv'
    prev_num = 0
    cnt = 0
    for k, v in cpkt.items():
        if 'feature' in k and k.split('.')[2] != prev:
            prev = k.split('.')[2]
            cnt += 1
            if 'conv' in k:
                cnt += 1
        if 'feature' in k:        
            if int(k.split('.')[1]) - prev_num == 2:
                cnt += 1
            prev_num = int(k.split('.')[1])
            new_dict['feature.{}.{}'.format(cnt, k.split('.')[3])] = v
        else:
            new_dict[k] = v
    model = GraSP_VGG(cfg=cfg).to(device)
    model.load_state_dict(new_dict, strict=False)
    return model.to(device)

def get_attribute_preserve_VGG(fn, cfg=None):
    model = GraSP_VGG(cfg=cfg).to(device)
    ckpt = torch.load(fn)
    new_dict = {}
    for k, v in ckpt['state_dict'].items():
        new_dict[k[7:]] = v
    model.load_state_dict(new_dict, strict=True)
    return model.to(device)