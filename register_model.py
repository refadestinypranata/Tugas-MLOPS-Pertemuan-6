import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS = 3
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "Fashion-MNIST-CNN-Registry"
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================
# SET SEED
# =========================
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# =========================
# MODEL
# =========================
class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
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
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# DATASET
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# MODEL
# =========================
model = CNN(DROPOUT).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================
with mlflow.start_run(run_name="Best-Model-Trial-19"):

    mlflow.log_param("seed", SEED)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("dropout", DROPOUT)

    for epoch in range(EPOCHS):

        model.train()

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

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total


        # =========================
        # VALIDATION
        # =========================
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)


    # =========================
    # LOG MODEL
    # =========================

    mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle"
    )

    run_id = mlflow.active_run().info.run_id


# =========================
# MODEL REGISTRY
# =========================

model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="Fashion-MNIST-CNN"
)


# =========================
# CHAMPION ALIAS
# =========================

client = mlflow.MlflowClient()

client.set_registered_model_alias(
    name="Fashion-MNIST-CNN",
    alias="champion",
    version=registered_model.version
)


# =========================
# DONE
# =========================

print()
print("======================================")
print("MODEL REGISTRY SELESAI")
print("======================================")
print(f"Run ID       : {run_id}")
print(f"Model        : Fashion-MNIST-CNN")
print(f"Version      : {registered_model.version}")
print(f"Alias        : champion")
print("======================================")