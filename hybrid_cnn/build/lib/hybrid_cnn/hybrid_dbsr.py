import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

class DBSR(nn.Module):
    def __init__(self, num_blocks=4):
        super(DBSR, self).__init__()
        self.blocks = nn.Sequential(*[self._make_block() for _ in range(num_blocks)])

    def _make_block(self):
        block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )
        return block

    def forward(self, x):
        return self.blocks(x)

class CustomDBSRDataset(Dataset):
    def __init__(self, folder_path, dbsr_blocks=4, augment=True):
        self.dataset = ImageFolder(root=folder_path)
        self.dbsr_model = DBSR(num_blocks=dbsr_blocks)
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_tensor = transforms.ToTensor()(image)
        enhanced_image_tensor = self.dbsr_model(image_tensor.unsqueeze(0)).squeeze(0)
        if self.augment:
            enhanced_image_tensor = self.transform(enhanced_image_tensor)
        return enhanced_image_tensor, label

def get_dataloaders(folder_path, batch_size=64, dbsr_blocks=4, augment=True, split_ratio=0.8, num_workers=4):
    dataset = CustomDBSRDataset(folder_path, dbsr_blocks=dbsr_blocks, augment=augment)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader