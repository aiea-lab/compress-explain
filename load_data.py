from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def load_data(path, preprocess, loader=datasets.default_loader, num_workers=4, batch_size=2048, train_split=0.8):

    train_set = datasets.ImageFolder(root=path+'/train', transform=preprocess, loader=loader)
    train_size = int(train_split * len(train_set))
    test_size = len(train_set) - train_size
    train_dataset, test_dataset = random_split(train_set, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, test_loader