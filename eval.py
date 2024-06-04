import torch
import time
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import average_precision_score, f1_score
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

def compute_mAP(labels,outputs):
    AP = []
    for i in range(labels.shape[0]):
        AP.append(average_precision_score(labels[i],outputs[i]))
    return np.mean(AP)

def compute_f1(labels, outputs):
    outputs = outputs > 0.5
    return f1_score(labels, outputs, average="samples")

def validate_multilabel(model, testLoader, device='cuda:0', loss_func=nn.CrossEntropyLoss()):
    global best_acc
    model.eval()

    losses = AverageMeter(':.4e')
    mAP = AverageMeter(':6.3f')
    f1 = AverageMeter(':.4e')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            labels_cpu = targets.cpu().detach().numpy()
            outputs_cpu = outputs.cpu().detach().numpy()
            mAP.update(compute_mAP(labels_cpu, outputs_cpu), inputs.size(0))
            f1.update(compute_f1(labels_cpu, outputs_cpu), inputs.size(0))

        current_time = time.time()
        print(
            'Test Loss {:.4f}\tmAP {:.2f}%\tf1 score {:.2f}\tTime {:.2f}s\n'
            .format(float(losses.avg), float(mAP.avg*100), float(f1.avg), (current_time - start_time))
        )
    return mAP.avg.item()

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
        pred_x = torch.argmax(model_x(batch.to(device)), dim=1)
        pred_y = torch.argmax(model_y(batch.to(device)), dim=1)
        batch_size += torch.sum(pred_x == pred_y)
        for j, img in enumerate(batch):
            if pred_x[j] == pred_y[j]:
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
    return torch.tensor(batch_comp).to(device) / batch_size

def grad_cam_batch_labels(model_x, model_y, data_loader, comp_methods=None, target_layers_x=None, target_layers_y=None, threshold=0.3):
    comparison_methods = []
    batch_comp = []
    tot_size = 0
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
        labels = labels.to(device)
        batch = batch.to(device)
        pred_x = model_x(batch) > 0.5
        pred_y = model_y(batch) > 0.5
        intersection = torch.logical_and(torch.logical_and(labels, pred_x), pred_y)
        # intersection = (labels and pred_x) and pred_y
        # tot_size += torch.sum(pred_x == pred_y)
        for j, img in enumerate(batch):
            # print(intersection[j])
            # print([labels[j,int(i.item())] for i in intersection[j].nonzero()])
            targets = [ClassifierOutputTarget(int(i.item())) for i in intersection[j].nonzero()]
            # for i in intersection[j].nonzero():
            # if targets:
            for target in targets:
                tot_size += 1
                map_x = cam_x(input_tensor=img[None,:,:,:], targets=[target])
                map_y = cam_y(input_tensor=img[None,:,:,:], targets=[target])
                for i, comp_method in enumerate(comparison_methods):
                    if comp_method == IoU:
                        difference = comp_method(map_x, map_y, threshold=threshold)
                    else:
                        difference = comp_method(map_x, map_y)
                    batch_comp[i] += difference
        # if b%10==0:
        #     print(f"batch: {b}")
        #     for i, comp_method in enumerate(comparison_methods):
        #         print(f"{inv_method_map[comp_method]}: {batch_comp[i]/tot_size}")
    # print(tot_size, len(data_loader.dataset))
    return torch.tensor(batch_comp).to(device) / tot_size

def confidence_difference(model_x, model_y, data_loader, MSE=False):
    batch_size = 0
    confidence_dif = 0
    for b, (batch,labels) in enumerate(data_loader):
        batch = batch.to(device)
        batch_size += batch.shape[0]
        logits_x = model_x(batch)
        logits_y = model_y(batch)
        index = list(np.vstack((np.array(range(len(labels))),labels)))
        confidence_x = torch.nn.Softmax()(logits_x)[index]
        confidence_y = torch.nn.Softmax()(logits_y)[index]
        if MSE:
            confidence_dif += torch.sum((confidence_x-confidence_y)**2).item()
        else:
            confidence_dif += torch.sum(torch.abs(confidence_x-confidence_y)).item()
    return confidence_dif / batch_size
    
