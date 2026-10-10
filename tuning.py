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
# 1. KONFIGURASI
# =========================================================

SEED = 42
EPOCHS = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. RANDOM SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. DATASET
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


# =========================================================
# 4. TRAIN / VALIDATION SPLIT
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


# =========================================================
# 5. MODEL CNN
# =========================================================

class CNN(nn.Module):

    def __init__(self, dropout):
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

            nn.Dropout(dropout),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# 6. FUNGSI EVALUASI
# =========================================================

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

    return (
        total_loss / total,
        correct / total
    )


# =========================================================
# 7. DAFTAR 20 TRIAL
# =========================================================

trials = [

    {"learning_rate": 0.0001, "batch_size": 64,  "dropout": 0.2},
    {"learning_rate": 0.0001, "batch_size": 128, "dropout": 0.2},
    {"learning_rate": 0.0001, "batch_size": 256, "dropout": 0.2},

    {"learning_rate": 0.0003, "batch_size": 64,  "dropout": 0.2},
    {"learning_rate": 0.0003, "batch_size": 128, "dropout": 0.2},
    {"learning_rate": 0.0003, "batch_size": 256, "dropout": 0.2},

    {"learning_rate": 0.0005, "batch_size": 64,  "dropout": 0.3},
    {"learning_rate": 0.0005, "batch_size": 128, "dropout": 0.3},
    {"learning_rate": 0.0005, "batch_size": 256, "dropout": 0.3},

    {"learning_rate": 0.0007, "batch_size": 64,  "dropout": 0.3},
    {"learning_rate": 0.0007, "batch_size": 128, "dropout": 0.3},
    {"learning_rate": 0.0007, "batch_size": 256, "dropout": 0.3},

    {"learning_rate": 0.001, "batch_size": 64,  "dropout": 0.3},
    {"learning_rate": 0.001, "batch_size": 128, "dropout": 0.3},
    {"learning_rate": 0.001, "batch_size": 256, "dropout": 0.3},

    {"learning_rate": 0.001, "batch_size": 64,  "dropout": 0.5},
    {"learning_rate": 0.001, "batch_size": 128, "dropout": 0.5},
    {"learning_rate": 0.001, "batch_size": 256, "dropout": 0.5},

    {"learning_rate": 0.002, "batch_size": 128, "dropout": 0.3},
    {"learning_rate": 0.003, "batch_size": 128, "dropout": 0.5},
]


# =========================================================
# 8. MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Fashion-MNIST-CNN-Tuning"
)


# =========================================================
# 9. JALANKAN SEMUA TRIAL
# =========================================================

best_accuracy = 0.0
best_trial = None


for trial_number, config in enumerate(trials, start=1):

    print("\n" + "=" * 60)

    print(
        f"TRIAL {trial_number}/"
        f"{len(trials)}"
    )

    print(
        f"Learning Rate : "
        f"{config['learning_rate']}"
    )

    print(
        f"Batch Size    : "
        f"{config['batch_size']}"
    )

    print(
        f"Dropout       : "
        f"{config['dropout']}"
    )

    print("=" * 60)


    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = CNN(
        dropout=config["dropout"]
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )


    # -----------------------------------------------------
    # MLflow Run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name=f"Trial-{trial_number:02d}"
    ):

        # Log parameters

        mlflow.log_params({

            "model": "CNN",

            "dataset": "Fashion-MNIST",

            "trial": trial_number,

            "learning_rate":
                config["learning_rate"],

            "batch_size":
                config["batch_size"],

            "dropout":
                config["dropout"],

            "epochs": EPOCHS,

            "optimizer": "Adam",

            "train_size": train_size,

            "validation_size": val_size

        })


        mlflow.log_param(
            "device",
            str(DEVICE)
        )


        # -------------------------------------------------
        # Training
        # -------------------------------------------------

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


                loss = criterion(
                    outputs,
                    labels
                )


                loss.backward()

                optimizer.step()


                running_loss += (
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


            train_loss = (
                running_loss / total
            )

            train_accuracy = (
                correct / total
            )


            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion
            )


            # -------------------------------------------------
            # Print
            # -------------------------------------------------

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Acc: "
                f"{train_accuracy:.4f} | "
                f"Val Acc: "
                f"{val_accuracy:.4f}"
            )


            # -------------------------------------------------
            # MLflow Metrics
            # -------------------------------------------------

            mlflow.log_metrics({

                "train_loss": train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss": val_loss,

                "val_accuracy":
                    val_accuracy

            }, step=epoch + 1)


        # -------------------------------------------------
        # Simpan hasil akhir trial
        # -------------------------------------------------

        mlflow.log_metric(
            "final_val_accuracy",
            val_accuracy
        )


        # -------------------------------------------------
        # Cek trial terbaik
        # -------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            best_trial = trial_number


    print(
        f"Trial {trial_number} selesai | "
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )


# =========================================================
# 10. HASIL AKHIR
# =========================================================

print("\n" + "=" * 60)

print("TUNING SELESAI")

print(
    f"Best Trial       : "
    f"{best_trial}"
)

print(
    f"Best Validation Accuracy : "
    f"{best_accuracy:.4f}"
)

print("=" * 60)