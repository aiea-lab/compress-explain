import torch
from torch import nn
from torchvision import datasets, transforms, models
# import os
# import load_model
import eval
# import numpy as np
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from PIL import Image
# import torchvision.transforms as T
# import numpy as np
# import torch.nn.utils.prune as prune
# import matplotlib.pyplot as plt
from dataset import VOCnew
# from sklearn.metrics import average_precision_score, f1_score
import resnet_model
import layers

device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
print(device)

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# def compute_mAP(labels,outputs):
#     AP = []
#     for i in range(labels.shape[0]):
#         AP.append(average_precision_score(labels[i],outputs[i]))
#     return np.mean(AP)

class VocModel(nn.Module):
    def __init__(self, num_classes, weights=None, mask=False, lottery=False, attribute_preserve=False, hydra=False):
        super().__init__()
        # Use a pretrained model
        self.network = resnet_model.resnet34(weights=weights, mask=mask, lottery=lottery, attribute_preserve=attribute_preserve, hydra=hydra)
        # Replace last layer
        self.network.fc = layers.SubnetLinear(self.network.fc.in_features, num_classes) if hydra else nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

model = VocModel(num_classes=20).to(device)
model.load_state_dict(torch.load('./saved_models/resnet34_pretrain_25.pt'))
model.eval()

vanilla_model = VocModel(num_classes=20).to(device)
vanilla_model.load_state_dict(torch.load('./saved_models/resnet34_unstructure_prune_voc_iter_0.8.pt'))
vanilla_model.eval()

lottery_model = VocModel(num_classes=20, lottery=True).to(device)
lottery_model.load_state_dict(torch.load('./saved_models/lottery_resnet34_80_sparse.pt')['state_dict'])
lottery_model.eval()

# gal_model = VocModel(num_classes=20, mask=True).to(device)
# gal_model.load_state_dict(torch.load('./saved_models/gal_resnet34_60_sparse.pt')['state_dict_s'])
# gal_model.eval()

hydra_model = VocModel(num_classes=20, hydra=True).to(device)
hydra_model.load_state_dict(torch.load('./saved_models/hydra_resnet34_80.pt')['state_dict'], strict=False)
for _, v in hydra_model.named_modules():
    if hasattr(v, "set_prune_rate"):
        v.set_prune_rate(0.2)
hydra_model.eval()

attribute_preserve_dict = {}
for key, val in torch.load('./saved_models/resnet34_attribute_preserve_02_0.790.pt')['state_dict'].items():
    attribute_preserve_dict[key[7:]] = val

attribute_preserve_model = VocModel(num_classes=20, attribute_preserve=True).to(device)
attribute_preserve_model.load_state_dict(attribute_preserve_dict)

# TESTS
vanilla_results = eval.grad_cam_batch_labels(model, vanilla_model, test_loader, comp_methods=['cosine', 'l2', 'iou'], target_layers_x=[model.network.layer4[2].conv2], target_layers_y=[vanilla_model.network.layer4[2].conv2], threshold=0.3)
f = open("results.txt", "a")
f.write(f"vanilla_results: {str(vanilla_results)}")
f.close()
lottery_results = eval.grad_cam_batch_labels(model, lottery_model, test_loader, comp_methods=['cosine', 'l2', 'iou'], target_layers_x=[model.network.layer4[2].conv2], target_layers_y=[lottery_model.network.layer4[2].conv2], threshold=0.3)
f = open("results.txt", "a")
f.write(f"lottery_results: {str(lottery_results)}")
f.close()
# gal_results = eval.grad_cam_batch_labels(model, gal_model, test_loader, comp_methods=['cosine', 'l2', 'iou'], target_layers_x=[model.network.layer4[2].conv2], target_layers_y=[gal_model.network.layer4[2].conv2], threshold=0.3)
hydra_results = eval.grad_cam_batch_labels(model, hydra_model, test_loader, comp_methods=['cosine', 'l2', 'iou'], target_layers_x=[model.network.layer4[2].conv2], target_layers_y=[hydra_model.network.layer4[2].conv2], threshold=0.3)
f = open("results.txt", "a")
f.write(f"hydra_results: {str(hydra_results)}")
f.close()
attribute_preserve_results = eval.grad_cam_batch_labels(model, attribute_preserve_model, test_loader, comp_methods=['cosine', 'l2', 'iou'], target_layers_x=[model.network.layer4[2].conv2], target_layers_y=[attribute_preserve_model.network.layer4[2].conv2], threshold=0.3)
f = open("results.txt", "a")
f.write(f"attribute_preserve_results: {str(attribute_preserve_results)}")
f.close()
