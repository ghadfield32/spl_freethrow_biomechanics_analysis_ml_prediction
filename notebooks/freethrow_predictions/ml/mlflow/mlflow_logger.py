import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas
import os
import logging
import pandas as pd

class MLflowLogger:
    def __init__(self, tracking_uri=None, experiment_name="Default Experiment", enable_mlflow=True):
        self.enable_mlflow = enable_mlflow
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if self.enable_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                self.logger.info(f"MLflow Tracking URI set to: {tracking_uri}")
            else:
                self.logger.warning("MLflow Tracking URI not provided. Using default.")
            self.experiment_name = experiment_name
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow Experiment set to: {self.experiment_name}")

    def run_context(self, run_name, nested=False, tags=None):
        """
        Returns a context manager wrapping mlflow.start_run(...).
        """
        if self.enable_mlflow:
            return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
        else:
            # Dummy context manager when MLflow is disabled.
            from contextlib import nullcontext
            return nullcontext()

    def log_run(self, run_name, params, metrics, artifacts=None, tags=None, datasets=None, nested=False):
        """
        Log a single run with MLflow or basic logging.

        Args:
            run_name (str): Name of the run.
            params (dict): Parameters to log.
            metrics (dict): Metrics to log.
            artifacts (str or list): Paths to artifacts (files or directories).
            tags (dict): Tags to set for the run.
            datasets (list): List of datasets to log.
            nested (bool): Whether this run is nested under an active run.
        """
        if self.enable_mlflow:
            try:
                with mlflow.start_run(run_name=run_name, nested=nested):
                    self.logger.info(f"Started MLflow run: {run_name}")
                    
                    # Log parameters
                    if params:
                        mlflow.log_params(params)
                        self.logger.debug(f"Logged parameters: {params}")
                    
                    # Log metrics
                    if metrics:
                        for key, value in metrics.items():
                            if isinstance(value, (list, tuple)):  # Handle multiple values (e.g., per epoch)
                                for step, metric_value in enumerate(value):
                                    mlflow.log_metric(key, metric_value, step=step)
                                    self.logger.debug(f"Logged metric '{key}' at step {step}: {metric_value}")
                            else:
                                mlflow.log_metric(key, value)
                                self.logger.debug(f"Logged metric '{key}': {value}")
                    
                    # Log tags
                    if tags:
                        mlflow.set_tags(tags)
                        self.logger.debug(f"Set tags: {tags}")
                    
                    # Log artifacts
                    if artifacts:
                        if isinstance(artifacts, str):  # Single file or directory
                            mlflow.log_artifact(artifacts)
                            self.logger.debug(f"Logged artifact: {artifacts}")
                        elif isinstance(artifacts, list):  # Multiple paths
                            for artifact_path in artifacts:
                                mlflow.log_artifact(artifact_path)
                                self.logger.debug(f"Logged artifact: {artifact_path}")
                    
                    # Log datasets
                    if datasets:
                        for dataset in datasets:
                            self.log_datasets(dataset["dataframe"], dataset["source"], dataset["name"])
                    
            except Exception as e:
                self.logger.error(f"Failed to log MLflow run '{run_name}': {e}")
                raise
        else:
            # Basic Logging
            self.logger.info(f"Run Name: {run_name}")
            if params:
                self.logger.info(f"Parameters: {params}")
            if metrics:
                self.logger.info(f"Metrics: {metrics}")
            if tags:
                self.logger.info(f"Tags: {tags}")
            if artifacts:
                self.logger.info(f"Artifacts: {artifacts}")
            if datasets:
                for dataset in datasets:
                    self.logger.info(f"Dataset Logged: {dataset['name']} from {dataset['source']}")

    def log_model(self, model, model_name, conda_env=None):
        """
        Log a model to MLflow or perform basic logging.

        Args:
            model: Trained model object.
            model_name (str): Name of the model.
            conda_env (str): Path to a Conda environment file.
        """
        if self.enable_mlflow:
            try:
                mlflow.sklearn.log_model(model, artifact_path=model_name, conda_env=conda_env)
                self.logger.info(f"Model '{model_name}' logged to MLflow.")
            except Exception as e:
                self.logger.error(f"Failed to log model '{model_name}' to MLflow: {e}")
                raise
        else:
            # Basic Logging
            self.logger.info(f"Model '{model_name}' training complete. (MLflow logging disabled)")

    def log_datasets(self, dataframe, source, name):
        """
        Create a dataset log entry or perform basic logging.

        Args:
            dataframe (pd.DataFrame): The dataset.
            source (str): Data source (e.g., file path, S3 URI).
            name (str): Dataset name.

        Returns:
            Dataset object logged to MLflow or None.
        """
        if self.enable_mlflow:
            try:
                dataset = from_pandas(dataframe, source=source, name=name)
                mlflow.log_input(dataset)
                self.logger.info(f"Dataset '{name}' logged to MLflow from source '{source}'.")
                return dataset
            except Exception as e:
                self.logger.error(f"Failed to log dataset '{name}' to MLflow: {e}")
                raise
        else:
            # Basic Logging
            self.logger.info(f"Dataset '{name}' from source '{source}' logged. (MLflow logging disabled)")
            return None

    def get_active_run(self):
        """
        Retrieve the current active MLflow run or log a message.

        Returns:
            Active run info object or None.
        """
        if self.enable_mlflow:
            return mlflow.active_run()
        else:
            self.logger.info("No active MLflow run. MLflow logging is disabled.")
            return None

    def get_last_run(self):
        """
        Retrieve the last active MLflow run or log a message.

        Returns:
            Last run info object or None.
        """
        if self.enable_mlflow:
            return mlflow.last_active_run()
        else:
            self.logger.info("No last MLflow run available. MLflow logging is disabled.")
            return None
