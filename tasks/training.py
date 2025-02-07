# training.py:

from invoke import task
from pathlib import Path
import logging

# Import your MLflowLogger and TrainingJob classes
from notebooks.freethrow_predictions.ml.mlflow.mlflow_logger import MLflowLogger
from notebooks.freethrow_predictions.ml.jobs.training_job import TrainingJob

@task(help={
    "config": "Path to the training configuration file (YAML/JSON)",
})
def training(ctx, config="ml/model/preprocessor_config/preprocessor_config.yaml"):
    """
    Run the Training Job.
    """
    logging.basicConfig(level=logging.INFO)
    
    abs_tracking_uri = "file:///" + str(Path("../../data/model/mlruns").resolve())
    mlflow_logger = MLflowLogger(tracking_uri=abs_tracking_uri, experiment_name="SPL Feedback Experiment")
    
    config_path = Path(config)
    if not config_path.exists():
        print(f"Configuration file {config_path} does not exist!")
        return
    
    job = TrainingJob(config_path, mlflow_logger)
    job.run()
