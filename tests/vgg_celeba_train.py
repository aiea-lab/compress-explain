import torch
from torchvision import datasets, transforms
import torch.nn as nn
import time
import torch.optim as optim

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class GraSP_VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True, is_sparse=False, is_mask=False, num_classes=1, input_size=32):
        super(GraSP_VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.dataset = dataset
        self.feature = self.make_layers(cfg, batchnorm)
        self.classifier = nn.Linear(cfg[-1]*(input_size//32)**2, num_classes)

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

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

def getCeleba(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, download=False, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'celeba'))
    num_workers = kwargs.setdefault('num_workers', 0)
    kwargs.pop('input_size', None)
    print("Building celeba data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CelebA(
                root=data_root, split='train', download=download,
                transform=transforms.Compose([
                    transforms.Resize(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CelebA(
                root=data_root, split='valid', download= download if train else False,
                transform=transforms.Compose([
                    transforms.Resize(128),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
    
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
        inputs, targets = inputs.to(device), torch.unsqueeze(targets[:,31], -1).float().to(device).to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
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


train_loader, val_loader = getCeleba(batch_size=64, num_workers=0, download=True)

# model = resnet().to(device)
model = GraSP_VGG(dataset='celeba', input_size=128).to(device)
num_epochs = 160
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.0005, momentum = 0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
loss_func = nn.BCEWithLogitsLoss()
batch_size = 128

bst_loss = 1e10

for epoch in range(num_epochs):
    train(model, optimizer, train_loader, loss_func, epoch)
    loss = validate(model, val_loader, loss_func)
    if loss < bst_loss:
        torch.save(model.state_dict(), '/persistentvol/compress-explain/saved_models/vgg_pretrain_celeba_{}.pt'.format(epoch))
        bst_loss = loss
        print('Saved best model to /persistentvol/compress-explain/saved_models/vgg_pretrain_celeba_{}.pt'.format(epoch))
torch.save(model.state_dict(), '/persistentvol/compress-explain/saved_models/vgg_pretrain_celeba_latest.pt')
bst_loss = loss
print('Saved Latest model to /persistentvol/compress-explain/saved_models/vgg_pretrain_celeba_latest.pt')

