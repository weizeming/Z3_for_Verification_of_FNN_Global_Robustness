import torch
from model import SDN

def load_model(dataset):
    model = torch.load(f'SDN_models/{dataset}.pth')
    model.cpu()
    k = model.group
    W = [model.fc1.weight.data,
         model.fc2.weight.data,
         model.fc3.weight.data,
    ]
    B = [model.fc1.bias.data,
         model.fc2.bias.data,
         model.fc3.bias.data,
    ]
    N = [w.shape[0] for w in W]
    W = [w.numpy().tolist() for w in W]
    B = [b.numpy().tolist() for b in B]
    g = model.group
    a = model.alpha
    #print(N)
    return N, W, B, g, a


def generate_APs(N, k=2):
    APs = []
    AP1 = []
    AP2 = []
    for i in range(N[0] // 2 + 1):
        for j in range(N[0] // 2 + 1):
            if i == j and i < N[0] // 2:
                continue
            AP1.append([i,j])
    for i in range(N[1] // 2 + 1):
        for j in range(N[1] // 2 + 1):
            if i == j and i < N[1] // 2:
                continue
            AP2.append([i,j])
    for ap1 in AP1:
        for ap2 in AP2:
            APs.append(ap1+ap2)
    return APs
        
def load_prototype(dataset, target):
    all_pro = torch.load(f'AE_figs/{dataset}_prototype.pth')
    return all_pro[target].cpu().detach().numpy().tolist()
    
    
if __name__ == '__main__':
    #N, W, B = load_model('MNIST')
    #print(len(generate_APs(N)))
    pass