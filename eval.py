import torch
import time
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'

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
    
def validate(model, testLoader, device='cuda:0'):
    global best_acc
    model.eval()

    losses = AverageMeter(':.4e')
    accurary = AverageMeter(':6.3f')

    start_time = time.time()
    loss_func = nn.CrossEntropyLoss().cuda()

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
    return accurary.avg.item()


# GradCAM

def grad_cam_init(model_x, model_y, target_layers_x=None, target_layers_y=None):
    if target_layers_y == None:
        target_layers_y = [model_y.features[-1]]
    
    if target_layers_x == None:
        target_layers_x = [model_x.features[-1]]

    cam_x = GradCAM(model=model_x, target_layers=target_layers_x)
    cam_y = GradCAM(model=model_y, target_layers=target_layers_y)
    return cam_x, cam_y

def IoU(x, y, threshold=0.3):
    binary1 = (x > threshold).astype(int)
    binary2 = (y > threshold).astype(int)
    combined = binary1 + binary2
    intersection = np.count_nonzero(combined == 2)
    union = np.count_nonzero(combined > 0)
    if union == 0:
        return (int)(intersection>0)
    iou = intersection / union
    return iou

def l2_difference(x, y):
    return np.sum((x-y)**2)


def cosine_similarity(x, y):
    x_flat = x.flatten()
    y_flat = y.flatten()
    norm_denom = np.linalg.norm(x_flat)*np.linalg.norm(y_flat)
    if norm_denom == 0:
        return 1
    return np.dot(x_flat, y_flat) / (norm_denom)

def grad_cam_batch(model_x, model_y, data_loader, comp_methods=None, target_layers_x=None, target_layers_y=None, threshold=0.3):
    comparison_methods = []
    batch_comp = []
    batch_size = 0
    method_map = {'cosine': cosine_similarity, 'l2': l2_difference, 'iou': IoU}
    inv_method_map = {v: k for k, v in method_map.items()}
    if comp_methods == None:
        comparison_methods = method_map.values()
        batch_comp = [ 0 for _ in range(len(comparison_methods)) ]
    else:
        for i in comp_methods:
            comparison_methods.append(method_map[i])
            batch_comp.append(0)

    cam_x, cam_y = grad_cam_init(model_x, model_y, target_layers_x=target_layers_x, target_layers_y=target_layers_y)
    for b, (batch,labels) in enumerate(data_loader):
        batch_size += batch.shape[0]
        for j, img in enumerate(batch):
            map_x = cam_x(input_tensor=img[None,:,:,:], targets=[ClassifierOutputTarget(labels[j])])
            map_y = cam_y(input_tensor=img[None,:,:,:], targets=[ClassifierOutputTarget(labels[j])])
            for i, comp_method in enumerate(comparison_methods):
                if comp_method == IoU:
                    difference = comp_method(map_x, map_y, threshold=threshold)
                else:
                    difference = comp_method(map_x, map_y)
                batch_comp[i] += difference
        # if b%10==0:
        #     print(f"batch: {b}")
        #     for i, comp_method in enumerate(comparison_methods):
        #         print(f"{inv_method_map[comp_method]}: {batch_comp[i]/batch_size}")
    return np.array(batch_comp) / batch_size
