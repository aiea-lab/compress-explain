import torch
import torch.nn.utils.prune as prune

def vanilla_prune(model, conv_prune=0.3, linear_prune=0.6):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=conv_prune)
            prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=linear_prune)
            prune.remove(module, 'weight')