import logging
from pathlib import Path
import json
import pandas as pd

from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
from ml.train_utils.train_utils import save_model, load_model, bayes_best_model_train
from ml.mlflow.mlflow_logger import MLflowLogger

# Import feature metadata loader
from ml.feature_selection.feature_importance_calculator import manage_features
from datapreprocessor import DataPreprocessor

class TrainingJob:
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

        # Load the complete training data
        raw_data_file = Path(paths_config.data_dir).resolve() / paths_config.raw_data
        try:
            df = pd.read_csv(raw_data_file)
            self.logger.info(f"Loaded data from {raw_data_file}. Shape: {df.shape}")
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
        loaded_features = manage_features(mode='load', paths=feature_paths)
        if loaded_features:
            y_var = loaded_features.get('y_variable')
        else:
            self.logger.error("Failed to load feature metadata.")
            return

        # Initialize the DataPreprocessor with feature metadata (for training mode)
        preprocessor = DataPreprocessor(
            model_type="Tree Based Classifier",
            y_variable=y_var,
            ordinal_categoricals=loaded_features.get('ordinal_categoricals'),
            nominal_categoricals=loaded_features.get('nominal_categoricals'),
            numericals=loaded_features.get('numericals'),
            mode='train',
            debug=self.config.logging.debug,
            normalize_debug=False,
            normalize_graphs_output=False,
            graphs_output_dir=Path(paths_config.plots_output_dir),
            transformers_dir=Path(paths_config.transformers_save_base_dir)
        )
        try:
            # Process the data to obtain training and test splits
            X_train, X_test, y_train, y_test, *_ = preprocessor.final_preprocessing(df)
            self.logger.info(f"Preprocessing complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            return

        # Load tuning results to get best model info
        try:
            with tuning_results_save_path.open("r") as f:
                tuning_results = json.load(f)
            best_model_info = tuning_results.get("Best Model")
            if not best_model_info:
                self.logger.error("No best model found in tuning results.")
                return
            best_model_name = best_model_info.get("model_name")
            self.logger.info(f"Best model selected from tuning: {best_model_name}")
        except Exception as e:
            self.logger.error(f"Error loading tuning results: {e}")
            return

        # Load the best model from disk
        try:
            model = load_model(best_model_name, model_save_dir)
            self.logger.info(f"Loaded best model '{best_model_name}' from disk.")
        except Exception as e:
            self.logger.error(f"Error loading model '{best_model_name}': {e}")
            return

        # Retrain the best model on the full training data
        try:
            model.fit(X_train, y_train)
            self.logger.info(f"Model '{best_model_name}' retrained on full training data.")
        except Exception as e:
            self.logger.error(f"Error during retraining of model '{best_model_name}': {e}")
            return

        # Save the retrained model
        try:
            save_model(model, best_model_name, save_dir=model_save_dir)
            self.logger.info(f"Retrained model '{best_model_name}' saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save retrained model '{best_model_name}': {e}")
            return

        # Optionally, log the retrained model to MLflow
        with self.mlflow_logger.run_context("Training Job"):
            try:
                self.mlflow_logger.log_model(model, best_model_name)
            except Exception as e:
                self.logger.error(f"Failed to log retrained model '{best_model_name}' to MLflow: {e}")
                return

        self.logger.info("Training job completed successfully.")

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
    job = TrainingJob(config_path, mlflow_logger)
    job.run()
