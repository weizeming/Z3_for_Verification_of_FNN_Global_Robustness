from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_dataset(dataset, batch_size=256):
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
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_dataset('MNIST')
    train_loader, test_loader = load_dataset('FashionMNIST')