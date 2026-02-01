from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=8):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transform)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,num_workers=0)

    return train_loader, val_loader, test_loader, train_ds
