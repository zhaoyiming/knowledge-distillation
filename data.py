from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def data():
    batch_size = 64
    mean = [0.49139968, 0.48215841, 0.44653091]
    stdv = [0.24703223, 0.24348513, 0.26158784]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    testset = torchvision.datasets.CIFAR10('./data', train=False,
                                           download=True, transform=test_transforms)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    trainset = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transforms, download=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    return train_loader, testloader
