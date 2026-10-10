import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_full = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset, _ = random_split(
    train_full,
    [10000, 2000, len(train_full) - 12000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


configs = [
    ("Baseline-Dropout-0.3", 0.3),
    ("No-Dropout", 0.0)
]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fashion-MNIST-CNN-Ablation")

for name, dropout in configs:

    print("\n" + "=" * 50)
    print(f"EXPERIMENT: {name}")
    print(f"Dropout: {dropout}")
    print("=" * 50)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = CNN(dropout).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    with mlflow.start_run(run_name=name):

        mlflow.log_params({
            "model": "CNN",
            "dataset": "Fashion-MNIST",
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "dropout": dropout,
            "epochs": EPOCHS,
            "optimizer": "Adam"
        })

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch + 1)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        print(
            f"{name} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

print("\n" + "=" * 50)
print("ABLATION STUDY SELESAI")
print("=" * 50)