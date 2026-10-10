import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fashion-MNIST-CNN")

with mlflow.start_run():
    mlflow.log_param("model", "CNN")
    mlflow.log_param("dataset", "Fashion-MNIST")
    mlflow.log_metric("test_accuracy", 0.85)

print("Eksperimen berhasil dicatat ke MLflow.")
