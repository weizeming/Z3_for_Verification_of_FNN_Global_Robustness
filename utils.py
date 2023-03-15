from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
import numpy as np

def load_dataset(dataset, batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'MNIST':
        train_set = datasets.MNIST(root='data',
            train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='data',
            train=False, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root='data',
            train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='data',
            train=False, download=True, transform=transform)
    else:
        raise ValueError
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def load_dataset_kmeans(dataset, batch_size=256, cluster=10):
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    if dataset == 'MNIST':
        train_set = datasets.MNIST(root='data',
            train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='data',
            train=False, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root='data',
            train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='data',
            train=False, download=True, transform=transform)
    else:
        raise ValueError
    
    train_set_labeled = []
    for k in range(10):
        subset_indices = [i for i in range(len(train_set)) if train_set[i][1] == k]
        train_set_labeled.append(Subset(train_set, subset_indices))
    for k in range(10):
        labeled_set = train_set_labeled[k]
        train_set_labeled[k] = []
        labeled_npset = np.array([x[0].numpy() for x in labeled_set])
        labeled_npset = labeled_npset.reshape(labeled_npset.shape[0], -1)
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(labeled_npset)
        # set up dataset for each cluster
        for i in range(cluster):
            cluster_indices = [j for j in range(len(kmeans.labels_)) if kmeans.labels_[j] == i]
            train_set_labeled[k].append(Subset(labeled_set, cluster_indices))
        


    train_loader_k = []
    for k in range(10):
        train_loader_k.append([])
        for i in range(cluster):
            train_loader_k[k].append(DataLoader(train_set_labeled[k][i], batch_size=batch_size, shuffle=True))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader_k,train_loader, test_loader


if __name__ == '__main__':
    train_loader_k,train_loader, test_loader = load_dataset_kmeans('MNIST')
    #train_set,train_loader, test_loader = load_dataset('FashionMNIST')
