import mlflow
import mlflow.pytorch
import torch
from torchvision import datasets, transforms


# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/Fashion-MNIST-CNN@champion"


# =========================
# LOAD MODEL
# =========================
model = mlflow.pytorch.load_model(MODEL_URI)

model.eval()


# =========================
# LOAD DATA
# =========================
dataset = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# =========================
# TAKE ONE SAMPLE
# =========================
image, label = dataset[0]

image = image.unsqueeze(0)


# =========================
# PREDICTION
# =========================
with torch.no_grad():

    output = model(image)

    predicted = torch.argmax(output, dim=1).item()


# =========================
# RESULT
# =========================
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("======================================")
print("MODEL PREDICTION")
print("======================================")
print(f"Predicted : {class_names[predicted]}")
print(f"Actual    : {class_names[label]}")
print("======================================")