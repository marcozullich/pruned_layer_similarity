from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def CIFAR10(data_path, batch_train, batch_test=None, download=True, train=True, **kwargs):
    '''Returns trainloader and testloader (if requested) for CIFAR10.

    Parameters:
    data_path -- the path where the downloaded datasets will be stored or are already stored
    batch_train -- the batch size for the training set
    batch_test -- the batch size for the test set. If None, no training set is
        returned (default None)
    download -- whether to download the dataset (default True)
    train -- prepares training set for training purposes; data augmentation
        is applied and the train dataloader supports shuffling
    **kwargs -- additional arguments to pass on to DataLoader
    
    Returns:
    trainloader -- DataLoader for the training set
    testloader -- DataLoader for the test set
    '''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if train:
        transform_train = transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15,translate=(.1,.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        transform_train = transform_test
    
    trainset = datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=download)
    trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=train, **kwargs)
    
    testloader = None 

    if batch_test is not None:
        testset = datasets.CIFAR10(root=data_path, train=False, transform=transform_train, download=download)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=False, **kwargs)
    
    return trainloader, testloader
    

def SVHN(data_path, batch_train, batch_test=None, download=True, train=True, **kwargs):
    '''Returns trainloader and testloader (if requested) for SVHN.

    Parameters:
    data_path -- the path where the downloaded datasets will be stored or are already stored
    batch_train -- the batch size for the training set
    batch_test -- the batch size for the test set. If None, no training set is
        returned (default None)
    download -- whether to download the dataset (default True)
    train -- prepares training set for training purposes;  train dataloader supports shuffling
    **kwargs -- additional arguments to pass on to DataLoader
    
    Returns:
    trainloader -- DataLoader for the training set
    testloader -- DataLoader for the test set
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.SVHN(data_path, split="train", transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=train, **kwargs)

    testloader = None
    if batch_test is not None:
        testset = datasets.SVHN(data_path, split="test", transform=transform, download=True)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=False, **kwargs)
    
    return trainloader, testloader

def MNIST(data_path, batch_train, batch_test=None, download=True, train=True, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.1307],[.3081]),
    ])

    trainset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=train, **kwargs)

    testloader = None
    if batch_test is not None:
        testset = datasets.MNIST(data_path, train=False, transform=transform, download=True)
        testloader = DataLoader(testset, batch_size=batch_test, shuffle=False, **kwargs)
    
    return trainloader, testloader




