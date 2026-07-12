# Provides CIFAR-10 Dataset and Image Classifier (fully-connected MLP) and helpers
# To be used as a black-boxed predictor

import torch
import torchvision
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, output_dim=10):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

def load_data(batch_size=64, *, paper_acp_split=False, calibration_size=100):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(42)
    if paper_acp_split:
        if not 0 < calibration_size < len(testset):
            raise ValueError("calibration_size must be between 1 and test-set size - 1")
        trainset = full_trainset
        calset, testset = random_split(
            testset,
            [calibration_size, len(testset) - calibration_size],
            generator=split_generator,
        )
    else:
        trainset, calset = random_split(
            full_trainset,
            [45000, 5000],
            generator=split_generator,
        )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    calloader = DataLoader(
        calset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, calloader, testloader

# Train and Eval for one epoch

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)
