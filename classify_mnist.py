import argparse
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import torchvision.datasets
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on MNIST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--batch-size", default=128)
parser.add_argument("--learning-rate", default=1e-3, type=float)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=10,
            kernel_size=5)
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=20,
            kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(20*20*20,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.dropout(torch.flatten(x,start_dim=1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(args):
    transform = transforms.ToTensor()

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        args.dataset_root, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        args.dataset_root, train=False, download=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    model = CNN()

    model = model.to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    def compute_accuracy(preds,labels):
        return 100 * sum([1 if x == y else 0 for x,y in zip(labels,preds)]) / len(preds)

    for epoch in range(1,11):
        model.train()
        # train
        for i, (data,labels) in enumerate(train_loader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model.forward(data)

            loss = criterion(logits,labels)

            loss.backward()

            optimiser.step()
            optimiser.zero_grad()

            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = compute_accuracy(preds,labels)

            print(f"epoch: {epoch}, step: {i+1}/{len(train_loader)}, loss: {loss:.5f}, accuracy: {accuracy:2.2f}%")

        model.eval()
        # test
        total_loss = 0
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for i, (data,labels) in enumerate(test_loader):
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model.forward(data)

                loss = criterion(logits, labels)
                total_loss += loss

                preds = logits.argmax(-1)
                preds_list.extend(preds)
                labels_list.extend(labels)

        accuracy = compute_accuracy(preds_list, labels_list)

        average_loss = total_loss / len(test_loader)

        print(f"testing loss: {average_loss:.5f}, accuracy: {accuracy:2.2f}")

if __name__ == '__main__':
    main(parser.parse_args())
