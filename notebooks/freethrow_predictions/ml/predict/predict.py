
import pandas as pd
import logging
import os
import yaml
import joblib
import json
from pathlib import Path
from typing import Any, Dict

# Local imports - Adjust based on your project structure
from ml.train_utils.train_utils import load_model  # Ensure correct import path
from datapreprocessor import DataPreprocessor  # Uncomment and adjust as necessary

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Set up logger
logger = logging.getLogger("PredictAndAttachLogger")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs, adjust as needed

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Adjust this level for console output

# File handler
file_handler = logging.FileHandler("predictions.log")
file_handler.setLevel(logging.DEBUG)  # Log detailed information to a file

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def predict_and_attach_predict_probs(trained_model, X_preprocessed, X_inversed):
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

    # Determine class labels. Use trained_model.classes_ if available.
    if hasattr(trained_model, 'classes_'):
        class_labels = trained_model.classes_
    else:
        # Create generic class names if the attribute does not exist
        class_labels = [f'class_{i}' for i in range(prediction_probs.shape[1])]
    
    try:
        if X_inversed is not None:
            # Attach the basic predictions
            if 'Prediction' not in X_inversed.columns:
                X_inversed['Prediction'] = predictions
                logger.debug("Predictions attached to inverse-transformed DataFrame.")

            # Optionally, retain the list of probabilities in a column
            if 'Prediction_Probabilities' not in X_inversed.columns:
                X_inversed['Prediction_Probabilities'] = prediction_probs.tolist()
                logger.debug("Prediction probabilities (list) attached to inverse-transformed DataFrame.")

            # -----------------------------------------
            # Attach probability for each class as a column
            # -----------------------------------------
            for idx, label in enumerate(class_labels):
                col_name = f'Probability_{label}'
                X_inversed[col_name] = prediction_probs[:, idx]
                logger.debug(f"Attached column: {col_name}")

            # drop the 'Prediction_Probabilities' column
            X_inversed.drop(columns=['Prediction_Probabilities'], inplace=True)
            print("Dropped column = ", X_inversed)
            
            logger.debug(f"Final shape of X_inversed: {X_inversed.shape}")
            logger.debug(f"Final columns in X_inversed: {X_inversed.columns.tolist()}")
        else:
            logger.warning("X_inversed is None. Creating a new DataFrame with predictions.")
            # Build a dictionary with predictions, prediction probabilities and separate probability columns
            data = {'Prediction': predictions, 'Prediction_Probabilities': prediction_probs.tolist()}
            for idx, label in enumerate(class_labels):
                col_name = f'Probability_{label}'
                data[col_name] = prediction_probs[:, idx]
            

            X_inversed = pd.DataFrame(data)
            # drop the 'Prediction_Probabilities' column
            X_inversed.drop(columns=['Prediction_Probabilities'], inplace=True)
            print("Dropped column = ", X_inversed)
            
            logger.debug(f"Created new X_inversed DataFrame with shape: {X_inversed.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to attach predictions to inverse-transformed DataFrame: {e}")
        X_inversed = pd.DataFrame({'Prediction': predictions})  # fallback in case of error
        return

    return predictions, prediction_probs, X_inversed



def main():
    # ----------------------------
    # Step 1: Load Configuration
    # ----------------------------
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')  # Adjust as needed
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return  # Exit if config loading fails

    # ----------------------------
    # Step 2: Extract Paths from Configuration
    # ----------------------------
    paths = config.get('paths', {})
    data_dir = Path(paths.get('data_dir', '../../data/processed')).resolve()
    raw_data_path = data_dir / paths.get('raw_data', 'final_ml_dataset.csv')  # Corrected key
    processed_data_dir = data_dir / paths.get('processed_data_dir', 'preprocessor/processed')
    transformers_dir = Path(paths.get('transformers_save_base_dir', '../preprocessor/transformers')).resolve()  # Corrected key
    predictions_output_dir = Path(paths.get('predictions_output_dir', 'preprocessor/predictions')).resolve()
    log_dir = Path(paths.get('log_dir', '../preprocessor/logs')).resolve()
    model_save_dir = Path(paths.get('model_save_base_dir', '../preprocessor/models')).resolve()  # Corrected key
    log_file = paths.get('log_file', 'prediction.log')  # Ensure this key exists in config

    # ----------------------------
    # Step 3: Setup Logging
    # ----------------------------
    logger.info("✅ Starting prediction module.")

    # ----------------------------
    # Step 4: Extract Feature Assets
    # ----------------------------
    features_config = config.get('features', {})
    column_assets = {
        'y_variable': features_config.get('y_variable', []),
        'ordinal_categoricals': features_config.get('ordinal_categoricals', []),
        'nominal_categoricals': features_config.get('nominal_categoricals', []),
        'numericals': features_config.get('numericals', [])
    }

    # ----------------------------
    # Step 5: Load Tuning Results to Find Best Model
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
    # Step 6: Preprocess the Data
    # ----------------------------
    # Load Prediction Dataset
    if not raw_data_path.exists():
        logger.error(f"❌ Prediction input dataset not found at '{raw_data_path}'.")
        return

    try:
        df_predict = load_dataset(raw_data_path)
        logger.info(f"✅ Prediction input data loaded from '{raw_data_path}'.")
    except Exception as e:
        logger.error(f"❌ Failed to load prediction input data: {e}")
        return

    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",  # Or dynamically set based on best_model_name if necessary
        y_variable=column_assets.get('y_variable', []),
        ordinal_categoricals=column_assets.get('ordinal_categoricals', []),
        nominal_categoricals=column_assets.get('nominal_categoricals', []),
        numericals=column_assets.get('numericals', []),
        mode='predict',
        options={},  # Adjust based on config or load from somewhere
        debug=False,  # Can be parameterized
        normalize_debug=False,  # As per hardcoded paths
        normalize_graphs_output=False,  # As per hardcoded paths
        graphs_output_dir=Path(paths.get('plots_output_dir', '../preprocessor/plots')).resolve(),
        transformers_dir=transformers_dir
    )

    # Execute Preprocessing for Prediction
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_predict)
        print("X_new_preprocessed type = ", type(X_preprocessed), "X_new_inverse type = ", type(X_inversed))
        logger.info("✅ Preprocessing completed successfully in predict mode.")
    except Exception as e:
        logger.error(f"❌ Preprocessing failed in predict mode: {e}")
        return

    # ----------------------------
    # Step 7: Load the Best Model
    # ----------------------------
    try:
        trained_model = load_model(best_model_name, model_save_dir)
        logger.info(f"✅ Trained model loaded from '{model_save_dir / best_model_name.replace(' ', '_') / 'trained_model.pkl'}'.")
    except Exception as e:
        logger.error(f"❌ Failed to load the best model '{best_model_name}': {e}")
        return

    # ----------------------------
    # Step 8/9: Make Predictions/probs
    # ----------------------------
    predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(trained_model, X_preprocessed, X_inversed)
    print(X_inversed)

    # ----------------------------
    # Step 10: Save Predictions
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
