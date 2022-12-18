import argparse
from time import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_dataset
from model import SDN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--layer1', default=15, type=int) # The number of groups in layer 1
    parser.add_argument('--layer2', default=11, type=int) # The number of groups in layer 2
    parser.add_argument('--group', default=2, type=int) # The number of neurons in each group
    parser.add_argument('--alpha', default=4.0, type=float)
    
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    
    return parser.parse_args()

def train_epoch(model, loader, opt, criterion):
    model.train()
    _loss, acc, total = 0,0,0
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        x = x.reshape(-1, 784)
        output = model(x)
        
        opt.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        opt.step()
        
        _loss += loss * len(y)
        pred = output.max(1)[1]
        acc += (pred == y).float().sum()
        total += len(y)
        
    return (_loss/total).item(), (acc/total).item()
        
def eval_epoch(model, loader, criterion):
    model.eval()
    _loss, acc, total = 0,0,0
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        x = x.reshape(-1, 784)

        output = model(x)
        loss = criterion(output, y)        
        
        _loss += loss * len(y)
        pred = output.max(1)[1]
        acc += (pred == y).float().sum()
        total += len(y)
        
    return (_loss/total).item(), (acc/total).item()   

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    train_loader, test_loader = load_dataset(dataset, args.batch_size)
    epochs = args.epochs
    model = SDN(784, args.layer1, args.layer2, 10, args.group, args.alpha).cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    loss, acc = np.zeros((20, 2)), np.zeros((20,2))
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, opt, criterion)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion)
        logs = [train_loss, train_acc, test_loss, test_acc]
        print(logs)
        loss[epoch] = [train_loss, test_loss]
        acc[epoch] = [train_acc, test_acc]
        
    torch.save(model.cpu(), f'SDN_models/{dataset}.pth')
    
    plt.plot(loss)
    plt.grid()
    plt.legend(['train', 'test'], fontsize=16)
    plt.title('Cross-Entropy Loss', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'SDN_figs/loss_{dataset}.png', dpi=200, )
    plt.clf()
    
    plt.plot(acc)
    plt.grid()
    plt.legend(['train', 'test'], fontsize=16)
    plt.title('Accuracy', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'SDN_figs/acc_{dataset}.png', dpi=200)
    plt.clf()