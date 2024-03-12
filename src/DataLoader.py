import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def Dataset(batch_size):

    train_dataset = MNIST(root = './data/', transform = transforms.ToTensor(),download = True, train=True)
    test_dataset = MNIST(root = './data/', transform = transforms.ToTensor(),download = True , train = False)

    train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle =True)
    test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = False)
    
    return train_loader , test_loader