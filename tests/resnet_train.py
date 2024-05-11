import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
print(device)

def getVOC(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'pascalVOC-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building VOCDetection data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.VOCDetection(
                root=data_root, image_set='train', download=True,
                transform=transforms.Compose([
                    transforms.RandomChoice([
                        transforms.CenterCrop(300),
                        transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
                        ]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.VOCDetection(
                root=data_root, image_set='val', download=True,
                transform=transforms.Compose([
                    transforms.Resize(330), 
                    transforms.CenterCrop(300),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

train_loader, test_loader = getVOC(batch_size=256, num_workers=4)

model = models.resnet18(num_classes=20)
model.to(device)

num_epochs = 160
learning_rate = 0.1
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.0005, momentum = 0.9)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

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
    
"""Computes the precision@k for the specified values of k"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def train(model, optimizer, trainLoader, epoch):
    train_batch_size = 256

    model.train()
    losses = AverageMeter(':.4e')
    accurary = AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // train_batch_size // 10
    #print_freq = 1
    #import pdb;pdb.set_trace()
    start_time = time.time()
    i = 0 
    pop_config = np.array([0] * 32)
    for batch, (inputs, targets) in enumerate(trainLoader):
        i+=1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()
        prec1 = accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            print(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * 256, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time
    print("epoch{} pop_configuration {}".format(epoch, pop_config))

def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = AverageMeter(':.4e')
    accurary = AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        print(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg


for epoch in range(num_epochs):
    train(model, optimizer, train_loader, epoch)
    test_acc = validate(model, test_loader)
    scheduler.step()

    is_best = best_acc < test_acc
    best_acc = max(best_acc, test_acc)
    if is_best:
        torch.save(model.state_dict(), '../saved_models/resnet18_pretrain_best_{:.3f}.pt'.format(best_acc))
