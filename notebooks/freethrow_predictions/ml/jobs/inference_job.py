import logging
from pathlib import Path
import pandas as pd
import json

from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
from ml.mlflow.mlflow_logger import MLflowLogger
from ml.train_utils.train_utils import load_model

# Import feature metadata loader
from ml.feature_selection.feature_importance_calculator import manage_features
from datapreprocessor import DataPreprocessor
from ml.predict.predict import predict_and_attach_predict_probs

class InferenceJob:
    def __init__(self, config_path: Path, mlflow_logger: MLflowLogger):
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.mlflow_logger = mlflow_logger

    def run(self):
        paths_config = self.config.paths
        model_save_dir = Path(paths_config.model_save_base_dir).resolve()
        tuning_results_path = model_save_dir / "tuning_results.json"
        try:
            with tuning_results_path.open("r") as f:
                tuning_results = json.load(f)
            best_model_info = tuning_results.get("Best Model")
            if not best_model_info:
                self.logger.error("No best model info found in tuning results.")
                return
            best_model_name = best_model_info.get("model_name")
            self.logger.info(f"Best model for inference: {best_model_name}")
        except Exception as e:
            self.logger.error(f"Error loading tuning results: {e}")
            return

        # Load the prediction dataset
        raw_data_file = Path(paths_config.data_dir).resolve() / paths_config.raw_data
        try:
            df = pd.read_csv(raw_data_file)
            self.logger.info(f"Loaded prediction data from {raw_data_file}.")
        except Exception as e:
            self.logger.error(f"Error loading prediction data: {e}")
            return

        # Load feature metadata using manage_features for preprocessing in prediction mode
        feature_paths = {
            'features': '../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl',
            'ordinal_categoricals': '../../data/preprocessor/features_info/ordinal_categoricals.pkl',
            'nominal_categoricals': '../../data/preprocessor/features_info/nominal_categoricals.pkl',
            'numericals': '../../data/preprocessor/features_info/numericals.pkl',
            'y_variable': '../../data/preprocessor/features_info/y_variable.pkl'
        }
        loaded_features = manage_features(mode='load', paths=feature_paths)
        if not loaded_features:
            self.logger.error("Failed to load feature metadata.")
            return

        # Initialize DataPreprocessor for prediction (using the loaded feature metadata)
        preprocessor = DataPreprocessor(
            model_type="Tree Based Classifier",
            y_variable=loaded_features.get('y_variable'),  # may be optional in prediction mode
            ordinal_categoricals=loaded_features.get('ordinal_categoricals'),
            nominal_categoricals=loaded_features.get('nominal_categoricals'),
            numericals=loaded_features.get('numericals'),
            mode='predict',
            options={},  # any extra options for prediction
            debug=self.config.logging.debug,
            normalize_debug=False,
            normalize_graphs_output=False,
            graphs_output_dir=Path(paths_config.plots_output_dir),
            transformers_dir=Path(paths_config.transformers_save_base_dir)
        )
        try:
            X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df)
            self.logger.info(f"Preprocessing complete for inference. Processed data shape: {X_preprocessed.shape}")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            return

        # Load the best model from disk
        try:
            trained_model = load_model(best_model_name, model_save_dir)
            self.logger.info(f"Loaded model '{best_model_name}' successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model '{best_model_name}': {e}")
            return

        # Make predictions and optionally log them via MLflow
        with self.mlflow_logger.run_context("Inference Job"):
            try:
                predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(
                    trained_model, X_preprocessed, X_inversed
                )
            except Exception as e:
                self.logger.error(f"Error during prediction: {e}")
                return

            # Save predictions to a CSV file
            predictions_output_dir = Path(paths_config.predictions_output_dir).resolve()
            predictions_output_dir.mkdir(parents=True, exist_ok=True)
            predictions_filename = predictions_output_dir / f'predictions_{best_model_name.replace(" ", "_")}.csv'
            try:
                X_inversed.to_csv(predictions_filename, index=False)
                self.logger.info(f"Predictions saved to {predictions_filename}")
            except Exception as e:
                self.logger.error(f"Error saving predictions: {e}")

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
    job = InferenceJob(config_path, mlflow_logger)
    job.run()
