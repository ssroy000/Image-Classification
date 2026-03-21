import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(base_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train = datasets.ImageFolder(f"{base_path}/train", transform=transform)
    val = datasets.ImageFolder(f"{base_path}/test", transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train