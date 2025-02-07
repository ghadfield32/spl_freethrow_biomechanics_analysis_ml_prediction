#tuning.py:

from invoke import task
from pathlib import Path
import logging

# Import your MLflowLogger and TuningJob classes
from notebooks.freethrow_predictions.ml.mlflow.mlflow_logger import MLflowLogger
from notebooks.freethrow_predictions.ml.jobs.tuning_job import TuningJob

@task(help={
    "config": "Path to the tuning configuration file (YAML/JSON)",
})
def tuning(ctx, config="ml/model/preprocessor_config/preprocessor_config.yaml"):
    """
    Run the Tuning Job.
    """
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Build an absolute tracking URI (adjust this as needed)
    abs_tracking_uri = "file:///" + str(Path("../../data/model/mlruns").resolve())
    mlflow_logger = MLflowLogger(tracking_uri=abs_tracking_uri, experiment_name="SPL Feedback Experiment")
    
    # Create an absolute path to the configuration file
    config_path = Path(config)
    if not config_path.exists():
        print(f"Configuration file {config_path} does not exist!")
        return
    
    # Instantiate and run the tuning job
    job = TuningJob(config_path, mlflow_logger)
    job.run()
