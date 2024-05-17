import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from torchvision.models import alexnet, AlexNet_Weights
from torchvision.io import read_image
from torchvision import transforms, datasets

import time

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
from PIL import Image

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from xml.etree.ElementTree import Element as ET_Element
from typing import Any, Dict, Tuple
import collections
from xml.etree.ElementTree import parse as ET_parse
device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'

class VOCnew(datasets.VOCDetection):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, range(len(classes))))   

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        

        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(datasets.VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
                objs = [def_dic["object"]]
                lbl = np.zeros(len(VOCnew.classes))
                for ix, obj in enumerate(objs[0][0]):        
                    obj_class = VOCnew.class_to_ind[obj['name']]
                    lbl[obj_class] = 1
                return lbl
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def validate(model, testLoader, loss_func):
    model.eval()

    losses = AverageMeter(':.4e')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            if batch_idx == 0:
                print(outputs[0])
                print(targets[0])

        current_time = time.time()
        print(
            'Test Loss {:.4f}\t\tTime {:.2f}s\n'
            .format(float(losses.avg), (current_time - start_time))
        )
    return losses.avg


def train(model, optimizer, trainLoader, loss_func, epoch):
    train_batch_size = 128
    model.train()
    losses = AverageMeter(':.4e')
    print_freq = len(trainLoader.dataset) // train_batch_size // 10
    start_time = time.time()
    i = 0 
    for batch, (inputs, targets) in enumerate(trainLoader):
        i+=1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            print(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * targets.shape[0], len(trainLoader.dataset),
                    float(losses.avg), cost_time
                )
            )
            start_time = current_time

train_dataset = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(330),
                    transforms.Pad(30),
                    transforms.RandomCrop(300),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))

val_dataset = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='val', download=False,
                transform=transforms.Compose([
                    transforms.Resize(330), 
                    transforms.CenterCrop(300),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))


class VocModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34()
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

model = VocModel(num_classes=20, pretrained=False).to(device)
model.load_state_dict(torch.load('../saved_models/resnet34_pretrain_best_25.pt'))
epochs = 10
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
# opt_func = torch.optim.Adam

# num_epochs = 160
# learning_rate = 0.1
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
# loss_func = nn.MultiLabelSoftMarginLoss()
loss_func = nn.BCEWithLogitsLoss()
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def training(model, prune_level):
    bst_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay = weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.00001)

    for epoch in range(epochs):
        train(model, optimizer, train_loader, loss_func, epoch)
        loss = validate(model, val_loader, loss_func)
        if loss < bst_loss:
            torch.save(model.state_dict(), '/persistentvol/compress-explain/saved_models/resnet34_unstructure_prune_voc_{}.pt'.format(prune_level))
            bst_loss = loss
            print('Saved best model to /persistentvol/compress-explain/saved_models/resnet34_unstructure_prune_voc_{}.pt'.format(prune_level))
        sched.step()

def vanilla_prune(model, conv_prune=0.3, linear_prune=0.6):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=conv_prune)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=linear_prune)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

# prune pass 1
vanilla_prune(model, conv_prune=0.2, linear_prune=0.2)
training(model, "iter_0.2")
# save_model(model, 'prune_pass_1.pt')

# prune pass 2
vanilla_prune(model, conv_prune=0.4, linear_prune=0.4)
training(model, "iter_0.4")
# save_model(model, 'prune_pass_2.pt')

# prune pass 3
vanilla_prune(model, conv_prune=0.6, linear_prune=0.6)
training(model, "iter_0.6")
# save_model(model, 'prune_pass_3.pt')

# prune pass 4
vanilla_prune(model, conv_prune=0.8, linear_prune=0.8)
training(model, "iter_0.8")
# save_model(model, 'prune_pass_4.pt')

# prune pass 4
vanilla_prune(model, conv_prune=0.85, linear_prune=0.85)
training(model, "iter_0.85")
# save_model(model, 'prune_pass_4.pt')

vanilla_prune(model, conv_prune=0.9, linear_prune=0.9)
training(model, "iter_0.9")
# save_model(model, 'prune_pass_4.pt')

# prune pass 4
model.load_state_dict(torch.load('../saved_models/resnet34_pretrain_best_25.pt'))
vanilla_prune(model, conv_prune=0.8, linear_prune=0.8)
training(model, "oneshot_0.8")
# save_model(model, 'prune_pass_4.pt')

# prune pass 4
model.load_state_dict(torch.load('../saved_models/resnet34_pretrain_best_25.pt'))
vanilla_prune(model, conv_prune=0.85, linear_prune=0.85)
training(model, "oneshot_0.85")
# save_model(model, 'prune_pass_4.pt')

model.load_state_dict(torch.load('../saved_models/resnet34_pretrain_best_25.pt'))
vanilla_prune(model, conv_prune=0.9, linear_prune=0.9)
training(model, "oneshot_0.9")
# save_model(model, 'prune_pass_4.pt')
