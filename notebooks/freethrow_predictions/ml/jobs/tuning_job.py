import logging
from pathlib import Path
import json
import pandas as pd

from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
from ml.train_utils.train_utils import bayes_best_model_train
from ml.mlflow.mlflow_logger import MLflowLogger

# Import the feature metadata loader
from ml.feature_selection.feature_importance_calculator import manage_features

class TuningJob:
    def __init__(self, config_path: Path, mlflow_logger: MLflowLogger):
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.mlflow_logger = mlflow_logger

    def run(self):
        # Extract configurations (paths, models, etc.)
        paths_config = self.config.paths
        model_save_dir = Path(paths_config.model_save_base_dir).resolve()
        classification_report_path = model_save_dir / "classification_report.txt"
        tuning_results_save_path = model_save_dir / "tuning_results.json"

        # Load your training data – adjust as needed
        raw_data_file = Path(paths_config.data_dir).resolve() / paths_config.raw_data
        try:
            df = pd.read_csv(raw_data_file)
            self.logger.info(f"Loaded data from {raw_data_file}.")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return

        # Load feature metadata using manage_features (same as in train.py)
        feature_paths = {
            'features': '../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl',
            'ordinal_categoricals': '../../data/preprocessor/features_info/ordinal_categoricals.pkl',
            'nominal_categoricals': '../../data/preprocessor/features_info/nominal_categoricals.pkl',
            'numericals': '../../data/preprocessor/features_info/numericals.pkl',
            'y_variable': '../../data/preprocessor/features_info/y_variable.pkl'
        }
        loaded = manage_features(mode='load', paths=feature_paths)
        if loaded:
            y_var = loaded.get('y_variable')
        else:
            self.logger.error("Failed to load feature metadata.")
            return

        # Now initialize the DataPreprocessor with actual feature metadata (instead of blank lists)
        from datapreprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(
            model_type="Tree Based Classifier",
            y_variable=y_var,
            ordinal_categoricals=loaded.get('ordinal_categoricals'),
            nominal_categoricals=loaded.get('nominal_categoricals'),
            numericals=loaded.get('numericals'),
            mode='train',
            debug=self.config.logging.debug,
            normalize_debug=False,
            normalize_graphs_output=False,
            graphs_output_dir=Path(paths_config.plots_output_dir),
            transformers_dir=Path(paths_config.transformers_save_base_dir)
        )
        try:
            X_train, X_test, y_train, y_test, *_ = preprocessor.final_preprocessing(df)
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            return

        # Wrap the tuning job in an MLflow run context
        with self.mlflow_logger.run_context("Tuning Job"):
            try:
                bayes_best_model_train(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    selection_metric=self.config.models.selection_metric,
                    model_save_dir=model_save_dir,
                    classification_save_path=classification_report_path,
                    tuning_results_save=tuning_results_save_path,
                    selected_models=self.config.models.selected_models,
                    use_pca=True
                )
            except Exception as e:
                self.logger.error(f"Tuning failed: {e}")
                raise

        self.logger.info("Tuning job completed successfully.")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import logging
    logging.basicConfig(level=logging.INFO)
    # Change the tracking URI from "databricks" to the local folder.
    abs_tracking_uri = "file:///" + str(Path("../../data/model/mlruns").resolve())
    mlflow_logger = MLflowLogger(tracking_uri=abs_tracking_uri, experiment_name="SPL Feedback Experiment")
    print("Using absolute MLflow Tracking URI:", abs_tracking_uri)

    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    job = TuningJob(config_path, mlflow_logger)
    job.run()
