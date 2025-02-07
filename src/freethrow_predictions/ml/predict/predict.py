
import pandas as pd
import logging
import os
import joblib
import json
from pathlib import Path
from typing import Any, Dict

# Local imports - Adjust based on your project structure
from ml.train_utils.train_utils import load_model  # Ensure correct import path
from datapreprocessor import DataPreprocessor  # Adjust as necessary

# Import new configuration loader
from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig  # To use as type annotation

# Set up logger
logger = logging.getLogger("PredictAndAttachLogger")
logger.setLevel(logging.DEBUG)

# Set up console and file handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler("predictions.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def predict_and_attach_predict_probs(trained_model, X_preprocessed, X_inversed):
    # [Existing prediction code unchanged...]
    try:
        predictions = trained_model.predict(X_preprocessed)
        logger.info("✅ Predictions made successfully.")
        logger.debug(f"Predictions sample: {predictions[:5]}")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return

    try:
        prediction_probs = trained_model.predict_proba(X_preprocessed)
        logger.info("✅ Prediction probabilities computed successfully.")
        logger.debug(f"Prediction probabilities sample:\n{prediction_probs[:2]}")
    except Exception as e:
        logger.error(f"❌ Prediction probabilities computation failed: {e}")
        return

    if hasattr(trained_model, 'classes_'):
        class_labels = trained_model.classes_
    else:
        class_labels = [f'class_{i}' for i in range(prediction_probs.shape[1])]

    try:
        if X_inversed is not None:
            if 'Prediction' not in X_inversed.columns:
                X_inversed['Prediction'] = predictions
                logger.debug("Predictions attached to inverse-transformed DataFrame.")
            if 'Prediction_Probabilities' not in X_inversed.columns:
                X_inversed['Prediction_Probabilities'] = prediction_probs.tolist()
                logger.debug("Prediction probabilities attached to inverse-transformed DataFrame.")
            for idx, label in enumerate(class_labels):
                col_name = f'Probability_{label}'
                X_inversed[col_name] = prediction_probs[:, idx]
                logger.debug(f"Attached column: {col_name}")
            X_inversed.drop(columns=['Prediction_Probabilities'], inplace=True)
            logger.debug(f"Final shape of X_inversed: {X_inversed.shape}")
            logger.debug(f"Final columns in X_inversed: {X_inversed.columns.tolist()}")
        else:
            logger.warning("X_inversed is None. Creating a new DataFrame with predictions.")
            data = {'Prediction': predictions, 'Prediction_Probabilities': prediction_probs.tolist()}
            for idx, label in enumerate(class_labels):
                col_name = f'Probability_{label}'
                data[col_name] = prediction_probs[:, idx]
            X_inversed = pd.DataFrame(data)
            X_inversed.drop(columns=['Prediction_Probabilities'], inplace=True)
            logger.debug(f"Created new X_inversed DataFrame with shape: {X_inversed.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to attach predictions to inverse-transformed DataFrame: {e}")
        X_inversed = pd.DataFrame({'Prediction': predictions})
        return

    return predictions, prediction_probs, X_inversed

def main():
    # ----------------------------
    # Step 1: Load Configuration
    # ----------------------------
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    # ----------------------------
    # Step 2: Extract Paths from Configuration
    # ----------------------------
    data_dir = Path(config.paths.data_dir).resolve()
    raw_data_path = data_dir / config.paths.raw_data
    processed_data_dir = data_dir / config.paths.processed_data_dir
    transformers_dir = Path(config.paths.transformers_save_base_dir).resolve()
    predictions_output_dir = Path(config.paths.predictions_output_dir).resolve()
    log_dir = Path(config.paths.log_dir).resolve()
    model_save_dir = Path(config.paths.model_save_base_dir).resolve()
    log_file = config.paths.log_file

    logger.info("✅ Starting prediction module.")

    # ----------------------------
    # Step 3: Extract Feature Assets
    # ----------------------------
    features_config = config.features
    column_assets = {
        'y_variable': features_config.y_variable,
        'ordinal_categoricals': features_config.ordinal_categoricals,
        'nominal_categoricals': features_config.nominal_categoricals,
        'numericals': features_config.numericals
    }

    # ----------------------------
    # Step 4: Load Tuning Results to Find Best Model
    # ----------------------------
    tuning_results_path = model_save_dir / "tuning_results.json"
    if not tuning_results_path.exists():
        logger.error(f"❌ Tuning results not found at '{tuning_results_path}'. Cannot determine the best model.")
        return

    try:
        with open(tuning_results_path, 'r') as f:
            tuning_results = json.load(f)
        best_model_info = tuning_results.get("Best Model")
        if not best_model_info:
            logger.error("❌ Best model information not found in tuning results.")
            return
        best_model_name = best_model_info.get("model_name")
        if not best_model_name:
            logger.error("❌ Best model name not found in tuning results.")
            return
        logger.info(f"Best model identified: {best_model_name}")
    except Exception as e:
        logger.error(f"❌ Failed to load tuning results: {e}")
        return

    # ----------------------------
    # Step 5: Load the Prediction Dataset
    # ----------------------------
    if not raw_data_path.exists():
        logger.error(f"❌ Prediction input dataset not found at '{raw_data_path}'.")
        return

    try:
        df_predict = pd.read_csv(raw_data_path)
        logger.info(f"✅ Prediction input data loaded from '{raw_data_path}'.")
    except Exception as e:
        logger.error(f"❌ Failed to load prediction input data: {e}")
        return

    # ----------------------------
    # Step 6: Initialize DataPreprocessor
    # ----------------------------
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",  # Adjust if needed based on best_model_name
        y_variable=column_assets.get('y_variable', []),
        ordinal_categoricals=column_assets.get('ordinal_categoricals', []),
        nominal_categoricals=column_assets.get('nominal_categoricals', []),
        numericals=column_assets.get('numericals', []),
        mode='predict',
        options={},
        debug=False,
        normalize_debug=False,
        normalize_graphs_output=False,
        graphs_output_dir=Path(config.paths.plots_output_dir).resolve(),
        transformers_dir=transformers_dir
    )

    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_predict)
        print("X_new_preprocessed type =", type(X_preprocessed), "X_new_inverse type =", type(X_inversed))
        logger.info("✅ Preprocessing completed successfully in predict mode.")
    except Exception as e:
        logger.error(f"❌ Preprocessing failed in predict mode: {e}")
        return

    # ----------------------------
    # Step 7: Load the Best Model
    # ----------------------------
    try:
        trained_model = load_model(best_model_name, model_save_dir)
        model_path = model_save_dir / best_model_name.replace(" ", "_") / 'trained_model.pkl'
        logger.info(f"✅ Trained model loaded from '{model_path}'.")
    except Exception as e:
        logger.error(f"❌ Failed to load the best model '{best_model_name}': {e}")
        return

    # ----------------------------
    # Step 8: Make Predictions and Attach Probabilities
    # ----------------------------
    predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(trained_model, X_preprocessed, X_inversed)
    print(X_inversed)

    # ----------------------------
    # Step 9: Save Predictions
    # ----------------------------
    try:
        predictions_output_dir.mkdir(parents=True, exist_ok=True)
        predictions_filename = predictions_output_dir / f'predictions_{best_model_name.replace(" ", "_")}.csv'
        X_inversed.to_csv(predictions_filename, index=False)
        logger.info(f"✅ Predictions saved to '{predictions_filename}'.")
    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {e}")
        return

    logger.info(f"✅ All prediction tasks completed successfully for model '{best_model_name}'.")

if __name__ == "__main__":
    main()
