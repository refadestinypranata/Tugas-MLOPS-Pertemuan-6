import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# =========================================================
# 1. KONFIGURASI EKSPERIMEN
# =========================================================

SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5
DROPOUT = 0.3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. SET RANDOM SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. LOAD DATASET FASHION-MNIST
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.FashionMNIST(
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


# =========================================================
# 4. PEMBAGIAN DATA TRAIN DAN VALIDATION
# =========================================================

train_size = 10000
val_size = 2000

train_dataset, val_dataset, _ = random_split(
    dataset,
    [
        train_size,
        val_size,
        len(dataset) - train_size - val_size
    ],
    generator=torch.Generator().manual_seed(SEED)
)


# Menggunakan sebagian data test
test_size = 2000

test_dataset, _ = random_split(
    test_dataset,
    [
        test_size,
        len(test_dataset) - test_size
    ],
    generator=torch.Generator().manual_seed(SEED)
)


# =========================================================
# 5. DATA LOADER
# =========================================================

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


# =========================================================
# 6. MODEL CNN
# =========================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


model = CNN().to(DEVICE)


# =========================================================
# 7. LOSS FUNCTION DAN OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 8. FUNGSI EVALUASI
# =========================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


# =========================================================
# 9. KONFIGURASI MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Fashion-MNIST-CNN"
)


# =========================================================
# 10. TRAINING + MLFLOW TRACKING
# =========================================================

with mlflow.start_run(
    run_name="CNN-Baseline"
):

    # -----------------------------------------------------
    # Log parameter eksperimen
    # -----------------------------------------------------

    mlflow.log_params({

        "model": "CNN",

        "dataset": "Fashion-MNIST",

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "epochs": EPOCHS,

        "dropout": DROPOUT,

        "seed": SEED,

        "optimizer": "Adam",

        "train_size": train_size,

        "validation_size": val_size,

        "test_size": test_size

    })


    mlflow.log_param(
        "device",
        str(DEVICE)
    )


    # -----------------------------------------------------
    # Mulai menghitung waktu training
    # -----------------------------------------------------

    start_time = time.time()


    # -----------------------------------------------------
    # Training setiap epoch
    # -----------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            # Reset gradient

            optimizer.zero_grad()


            # Forward pass

            outputs = model(images)


            # Hitung loss

            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation

            loss.backward()


            # Update parameter

            optimizer.step()


            # Akumulasi loss

            running_loss += (
                loss.item()
                * images.size(0)
            )


            # Hitung akurasi

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        # -------------------------------------------------
        # Training metrics
        # -------------------------------------------------

        train_loss = running_loss / total

        train_accuracy = correct / total


        # -------------------------------------------------
        # Validation metrics
        # -------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader
        )


        # -------------------------------------------------
        # Tampilkan hasil epoch
        # -------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


        # -------------------------------------------------
        # Log metric setiap epoch ke MLflow
        # -------------------------------------------------

        mlflow.log_metrics({

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "val_loss": val_loss,

            "val_accuracy": val_accuracy

        }, step=epoch + 1)


    # =====================================================
    # 11. EVALUASI PADA TEST SET
    # =====================================================

    test_loss, test_accuracy = evaluate(
        model,
        test_loader
    )


    # =====================================================
    # 12. WAKTU TRAINING
    # =====================================================

    training_time = (
        time.time() - start_time
    )


    # =====================================================
    # 13. LOG HASIL AKHIR KE MLFLOW
    # =====================================================

    mlflow.log_metrics({

        "test_loss": test_loss,

        "test_accuracy": test_accuracy,

        "training_time_seconds": training_time

    })


    # =====================================================
    # 14. SIMPAN MODEL KE MLFLOW
    # =====================================================

    mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle"
    )


# =========================================================
# 15. TAMPILKAN HASIL
# =========================================================

print("\nTraining selesai.")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Training Time : "
    f"{training_time:.2f} detik"
)