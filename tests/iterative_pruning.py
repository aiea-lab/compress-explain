import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from torchvision.models import alexnet, AlexNet_Weights
from torchvision.io import read_image
from torchvision import transforms, datasets

import time

print(f"torch version: {torch.__version__}")


# Check if there are multiple devices (i.e., GPU cards)-
print(f"Number of GPU(s) available = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}')

# Step 1: Initialize model with the best available weights
weights = AlexNet_Weights.IMAGENET1K_V1
model = alexnet(weights=weights)
model.to(device)
model.eval()
print(model)

def load(path, preprocess, num_workers=4, batch_size=2048, train_split=0.8):

    train_set = datasets.ImageFolder(root=path+'/train', transform=preprocess)
    # test_set = datasets.ImageFolder(root=path+'/val', transform=preprocess)
    train_size = int(train_split * len(train_set))
    test_size = len(train_set) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_set, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, test_loader

preprocess = weights.transforms()
train_loader, test_loader = load('ILSVRC/Data/CLS-LOC', preprocess, num_workers=8)
print(f"train loaders: {train_loader}")


model.train()
def pruning(model, conv_prune=0.1, linear_prune=0.2):
    for name, module in model.named_modules():
        # prune 10% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.1)
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    print(dict(model.named_buffers()).keys())  # to verify that all masks exist
    return model




criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.00001)
# optimizer=torch.optim.Adam(model.classifier.parameters(),lr=0.001)
epochs = 10


def eval_model(model):
    tst_corr = 0
    tst_tot = 0
    loss_tot = 0
    batches = 0
    with torch.no_grad():
        for b, (X_test,y_test) in enumerate(test_loader):
            y_test = y_test.to(device) ###added
            X_test = X_test.to(device)
            y_val=model(X_test)
            predicted=torch.max(y_val.data,1)[1]
            btach_corr=(predicted==y_test).sum()
            btach_tot=len(y_test)
            tst_corr+=btach_corr
            tst_tot+=btach_tot
            loss=criterion(y_val,y_test)
            loss=loss.cpu().detach().numpy()
            loss_tot+=loss
            batches += 1
            if b%100==0:
                print(f"loss: {loss_tot/batches}, accuracy: {tst_corr/tst_tot}")

    print(f"loss: {loss_tot/batches}, accuracy: {tst_corr/tst_tot}")

# eval_model(model)

def training(model):
    start_time=time.time()
    train_losses=[]
    test_losses=[]
    train_correct=[]
    test_correct=[]
    model.train()
    for i in range(epochs):
        trn_corr=0
        tst_corr=0
        tot = 0
        for b, (X_train,y_train) in enumerate(train_loader):
            b+=1
            y_train = y_train.to(device) ###added
            X_train = X_train.to(device)
            y_pred=model(X_train)
            loss=criterion(y_pred,y_train)
            #Update parameters
            predicted=torch.max(y_pred.data,1)[1]
            batch_corr=(predicted==y_train).sum()
            trn_corr+= batch_corr
            tot += len(y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(b, loss.item())
            if b%100==0:
                print(f'epoch: {i:2}  batch: {b:4} [{tot:6}]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item()*100/(tot):7.3f}%')

        loss=loss.cpu().detach().numpy()
        train_losses.append(loss)
        train_correct.append(trn_corr)

        with torch.no_grad():
            for b, (X_test,y_test) in enumerate(test_loader):
                y_test = y_test.to(device) ###added
                X_test = X_test.to(device)
                b+=1

                y_val=model(X_test)
                predicted=torch.max(y_val.data,1)[1]
                btach_corr=(predicted==y_test).sum()
                tst_corr+=btach_corr

        loss=criterion(y_val,y_test)
        loss=loss.cpu().detach().numpy()
        test_losses.append(loss)
        test_correct.append(tst_corr)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

# prune pass 1
model = pruning(model)
model = training(model)
save_model(model, 'prune_pass_1.pt')

# prune pass 2
model = pruning(model)
model = training(model)
save_model(model, 'prune_pass_2.pt')

# prune pass 3
model = pruning(model)
model = training(model)
save_model(model, 'prune_pass_3.pt')

# prune pass 4
model = pruning(model)
model = training(model)
save_model(model, 'prune_pass_4.pt')