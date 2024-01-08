import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import os

if __name__ == '__main__':
    print("I am executing make_dataset.py")
    print(f"Current working directory is: {os.getcwd()}")

    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"{os.getcwd()}/data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"{os.getcwd()}/data/raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print(f"train_data.shape: {train_data.shape}")

    # Calculate mean and std
    mean = train_data.mean()
    std = train_data.std()
    normalize_transform = transforms.Normalize(mean=[mean], std=[std])

    test_data = torch.load(f"{os.getcwd()}/data/raw/test_images.pt")
    test_labels = torch.load(f"{os.getcwd()}/data/raw/test_target.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    print(f"train_data.shape: {train_data.shape}")

    # Apply normalization
    train_data_normalized = normalize_transform(train_data)
    test_data_normalized = normalize_transform(test_data)

    print(f"train_data_normalized.shape: {train_data_normalized.shape}")
    
    torch.save(torch.utils.data.TensorDataset(train_data_normalized, train_labels), f'{os.getcwd()}/data/processed/train.pt')
    torch.save(torch.utils.data.TensorDataset(test_data_normalized, test_labels), f'{os.getcwd()}/data/processed/test.pt')