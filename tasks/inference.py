#inference.py:

from invoke import task
from pathlib import Path
import logging

# Import your MLflowLogger and InferenceJob classes
from notebooks.freethrow_predictions.ml.mlflow.mlflow_logger import MLflowLogger
from notebooks.freethrow_predictions.ml.jobs.inference_job import InferenceJob

@task(help={
    "config": "Path to the inference configuration file (YAML/JSON)",
})
def inference(ctx, config="ml/model/preprocessor_config/preprocessor_config.yaml"):
    """
    Run the Inference Job.
    """
    logging.basicConfig(level=logging.INFO)
    
    abs_tracking_uri = "file:///" + str(Path("../../data/model/mlruns").resolve())
    mlflow_logger = MLflowLogger(tracking_uri=abs_tracking_uri, experiment_name="SPL Feedback Experiment")
    
    config_path = Path(config)
    if not config_path.exists():
        print(f"Configuration file {config_path} does not exist!")
        return

    job = InferenceJob(config_path, mlflow_logger)
    job.run()
