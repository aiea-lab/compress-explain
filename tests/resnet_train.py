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

fp = open('resnet.log', 'w')

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

        current_time = time.time()
        print(
            'Test Loss {:.4f}\t\tTime {:.2f}s\n'
            .format(float(losses.avg), (current_time - start_time)), file=fp
        )
        print(
            'Test Loss {:.4f}\t\tTime {:.2f}s\n'
            .format(float(losses.avg), (current_time - start_time))
        )
    return losses.avg


def train(model, optimizer, sched, trainLoader, loss_func, epoch):
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
                ), file=fp
            )
            print(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * targets.shape[0], len(trainLoader.dataset),
                    float(losses.avg), cost_time
                )
            )
            start_time = current_time
    sched.step()

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

# model = resnet().to(device)

class VocModel(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(weights=weights)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

model = VocModel(num_classes=20, weights=models.ResNet34_Weights.DEFAULT).to(device)
# model.load_state_dict(torch.load('../saved_models/resnet34_pretrain_best_9.pt'))
epochs = 30
max_lr = 0.0001
grad_clip = None
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

bst_loss = 1e10
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay = weight_decay)
# sched = torch.optim.lr_scheduler.LinearLR(optimizer, max_lr, epochs=epochs,
#                                             steps_per_epoch=len(train_loader))
sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.00001)

for epoch in range(epochs):
    train(model, optimizer, sched, train_loader, loss_func, epoch)
    loss = validate(model, val_loader, loss_func)
    if loss < bst_loss:
        torch.save(model.state_dict(), '/persistentvol/compress-explain/saved_models/resnet34_pretrain_{}.pt'.format(epoch))
        bst_loss = loss
        print('Saved best model to /persistentvol/compress-explain/saved_models/resnet34_pretrain_{}.pt'.format(epoch), file=fp)
        print('Saved best model to /persistentvol/compress-explain/saved_models/resnet34_pretrain_{}.pt'.format(epoch))