from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths=[]
labels=[]
for dirname, _, filenames in os.walk('imagenet_mini/imagenet-mini/train'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            paths+=[(os.path.join(dirname, filename))]
            label=re.split(r'/|\\',dirname)[-1]
            labels+=[label]
            
tpaths=[]
tlabels=[]
for dirname, _, filenames in os.walk('imagenet_mini/imagenet-mini/val'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            tpaths+=[(os.path.join(dirname, filename))]
            label=re.split(r'/|\\',dirname)[-1]
            tlabels+=[label]           


class_names=sorted(set(labels))
#print(class_names)
# print(len(class_names))
N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N)) 
# reverse_mapping=dict(zip(N,class_names))    


df=pd.DataFrame(columns=['path','label'])
df['path']=paths
df['label']=labels
df['label']=df['label'].map(normal_mapping)

tdf=pd.DataFrame(columns=['path','label'])
tdf['path']=tpaths
tdf['label']=tlabels
tdf['label']=tdf['label'].map(normal_mapping)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        path = self.dataframe.loc[index, 'path']
        label = self.dataframe.loc[index, 'label']
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label
    


def get_train_loader(transform, batch_size=32):
    train_ds = CustomDataset(df, transform=transform)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True)

def get_test_loader(transform, batch_size=32):
    test_ds = CustomDataset(tdf, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=True) 