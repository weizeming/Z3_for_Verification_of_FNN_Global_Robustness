import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os


from utils import load_dataset,load_dataset_kmeans


class AutoEncoder(nn.Module):
    def __init__(self, h1=256, h2=128) -> None:
        super().__init__()
        
        self.en1 = nn.Linear(784, h1)
        self.en2 = nn.Linear(h1, h2)
        
        self.de1 = nn.Linear(h2, h1)
        self.de2 = nn.Linear(h1, 784)
        
    def encoder(self, x):
        y = F.relu(self.en1(x))
        return self.en2(y)
    
    def decoder(self, c):
        y = F.relu(self.de1(c))
        return self.de2(y)
    
    def forward(self, x):
        c = self.encoder(x)
        return self.decoder(c)
    
if __name__ == '__main__':
    """
    parameters
    """
    dataset = 'MNIST' # choices: ['MNIST', 'FashionMNIST']
    num_epoch = 100
    bs = 128
    lr = 0.001
    h1 = 256
    h2 = 64
    save_num = 100
    
    model = AutoEncoder(h1, h2).cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_loader_k, train_loader, test_loader = load_dataset_kmeans(dataset)   
    all_loss = np.zeros((num_epoch, 2))
    
    for epoch in range(num_epoch):
        train_loss, test_loss = 0,0
        for x, _ in train_loader:
            x=x.cuda()
            x=x.reshape(-1,784)
            hat_x = model(x)
            
            loss = criterion(x, hat_x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * len(x)
        train_loss/=60000
        
        for idx, (x, _) in enumerate(test_loader):
            x=x.cuda()
            x=x.reshape(-1,784)
            hat_x = model(x)
        
            loss = criterion(x, hat_x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            test_loss += loss.item() * len(x)
            
                    
        test_loss /= 10000
        all_loss[epoch] = [train_loss, test_loss]
        print(f'Epoch {epoch}\t train loss={train_loss:.4f}\t test loss={test_loss:.4f}')
    
    plt.plot(all_loss)
    plt.grid()
    plt.legend(['train', 'test'], fontsize=16)
    plt.title('MSE Loss', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'AE_figs/loss_{dataset}.png', dpi=200, )
    plt.clf()
    
    # Find prototypes
    c = torch.zeros(100, h2).cuda()
    for k,loader_list in enumerate(train_loader_k):
        for m,loader in enumerate(loader_list):
            counter = 0
            for idx, (x,y) in enumerate(loader):
                x = x.reshape(-1, 784)
                x = x.cuda()
                cx = model.encoder(x)
                for i, X in enumerate(cx):
                    #label = y[i]
                    c[k*10+m] += X
                    counter += 1
            c[k*10+m] /= counter
    '''
    for idx, (x,y) in enumerate(train_loader):
        x = x.reshape(-1, 784)
        x = x.cuda()
        cx = model.encoder(x)
        for i, X in enumerate(cx):
            label = y[i]
            c[label] += X
    '''
    #c /= 6000
    
    p = model.decoder(c)
    p = torch.clamp(p, 0, 1)
    
    torch.save(p, f'AE_figs/{dataset}_prototype.pth')
    
    p = p.detach().cpu().numpy()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(p[i].reshape(28,28))
    #plt.title(f'Prototype of {dataset}', fontsize=18)
    #plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.2)
    plt.savefig(f'AE_figs/{dataset}_prototype.png', dpi=400)