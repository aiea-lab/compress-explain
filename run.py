import torch
from torchvision import datasets, transforms
import os
import load_model
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


log_interval = 100
test_interval = 5
def train(model, epochs=150, lr=0.001, decreasing_lr='80,120', wd=0, save='vanilla_pruning'):
    best_acc = 0
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    decreasing_lr = list(map(int, decreasing_lr.split(',')))
    t_begin = time.time()
    for epoch in range(epochs):
        model.train()
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = correct * 1.0 / len(data)
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data.item(), acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * epochs - elapse_time
        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        # misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

        if epoch % test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                indx_target = target.clone()
                # if use_cuda:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.cross_entropy(output, target).data.item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader) # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))
            if acc > best_acc:
                # new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                # misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                torch.save(model.state_dict(), '/saved_models/{}_best.pt'.format(save))
                # old_file = new_file
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


def evaluate(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        indx_target = target.clone()
        # if use_cuda:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

    test_loss = test_loss / len(test_loader) # average over number of mini-batch
    acc = 100. * correct / len(test_loader.dataset)
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), acc))


train_loader, test_loader = get10(batch_size=200, num_workers=1)
pretrain_model = load_model.get_GraSP_VGG('./saved_models/pretrain_best_lottery.pt')
pretrain_model.pruneing.vanilla_prune(0.8, 0.9)
train(pretrain_model, epochs=10, lr=0.00001,  save='vanilla_pruning_one_shot')

pretrain_model = load_model.get_GraSP_VGG('./saved_models/pretrain_best_lottery.pt')
pretrain_model.pruneing.vanilla_prune(0.16, 0.2)
train(pretrain_model, epochs=5, lr=0.00001,  save='vanilla_pruning_iterative_0')
pretrain_model.pruneing.vanilla_prune(0.16, 0.2)
train(pretrain_model, epochs=5, lr=0.00001,  save='vanilla_pruning_iterative_1')
pretrain_model.pruneing.vanilla_prune(0.16, 0.2)
train(pretrain_model, epochs=5, lr=0.00001,  save='vanilla_pruning_iterative_2')
pretrain_model.pruneing.vanilla_prune(0.16, 0.2)
train(pretrain_model, epochs=5, lr=0.00001,  save='vanilla_pruning_iterative_3')
pretrain_model.pruneing.vanilla_prune(0.16, 0.1)
train(pretrain_model, epochs=10, lr=0.00001,  save='vanilla_pruning_iterative_4')

