import joblib
from pathlib import Path
import mlflow.sklearn

class LocalModelSaver:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, model_name: str):
        model_path = self.save_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        return model_path

    def load(self, model_name: str):
        model_path = self.save_dir / f"{model_name}_model.pkl"
        return joblib.load(model_path)

class MLflowModelSaver:
    def __init__(self, mlflow_logger, artifact_path: str = "model"):
        self.mlflow_logger = mlflow_logger
        self.artifact_path = artifact_path

    def save(self, model, model_name: str, conda_env: str = None):
        mlflow.sklearn.log_model(model, artifact_path=model_name, conda_env=conda_env)
        # Optionally return the model URI if needed
        model_uri = f"models:/{model_name}/1"
        return model_uri

    def load(self, model_uri: str):
        return mlflow.sklearn.load_model(model_uri)
