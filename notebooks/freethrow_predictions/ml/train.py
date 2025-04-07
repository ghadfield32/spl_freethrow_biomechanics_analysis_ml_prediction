import logging
from pathlib import Path
import pandas as pd
import os
import sys
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import json
from ml.config.config_loader import load_config  # New import for config
from ml.config.config_models import AppConfig
from datapreprocessor import DataPreprocessor
from ml.feature_selection.feature_importance_calculator import manage_features
from ml.train_utils.train_utils import (bayes_best_model_train
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print("WE'RE IN THIS DIRECTORY =", os.getcwd())
print("WE'RE IN THIS sys.path =", sys.path)

def main():
    # --- 1. Load Configuration via our new module ---
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    # load_config now returns an AppConfig instance (verified and typed)
    config: AppConfig = load_config(config_path)
    
    # --- 2. Use Config Values in the Code ---
    # Extract paths from configuration (using our typed model)
    paths_config = config.paths
    base_data_dir = Path(paths_config.data_dir).resolve()
    raw_data_file = base_data_dir / paths_config.raw_data

    # Output directories
    log_dir = Path(paths_config.log_dir).resolve()
    model_save_dir = Path(paths_config.model_save_base_dir).resolve()
    transformers_save_dir = Path(paths_config.transformers_save_base_dir).resolve()
    plots_output_dir = Path(paths_config.plots_output_dir).resolve()
    training_output_dir = Path(paths_config.training_output_dir).resolve()

    # Ensure output directories exist if needed
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    CLASSIFICATION_REPORT_PATH = model_save_dir / "classification_report.txt"
    TUNING_RESULTS_SAVE_PATH = model_save_dir / "tuning_results.json"

    # Extract model settings from the config
    selected_models = config.models.selected_models
    print(f"Selected Models: {selected_models}")
    selection_metric = config.models.selection_metric

    # Load the dataset
    try:
        filtered_df = pd.read_csv(raw_data_file)
        logger.info(f"✅ Loaded dataset from {raw_data_file}. Shape: {filtered_df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return
    base_dir = Path("../../data") / "preprocessor" / "features_info"
    # Load feature metadata using manage_features
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
        numericals = loaded.get('numericals')
        print(f"Numericals: {numericals}")
        print(f"Ordinal Categoricals: {loaded.get('ordinal_categoricals')}")
        print(f"Nominal Categoricals: {loaded.get('nominal_categoricals')}")
        print(f"Y Variable: {y_var}")
    else:
        logger.error("❌ Failed to load feature metadata.")
        return

    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=y_var,
        ordinal_categoricals=loaded.get('ordinal_categoricals'),
        nominal_categoricals=loaded.get('nominal_categoricals'),
        numericals=loaded.get('numericals'),
        mode='train',
        debug=config.logging.debug,
        normalize_debug=False,
        normalize_graphs_output=False,
        graphs_output_dir=plots_output_dir,
        transformers_dir=transformers_save_dir
    )

    try:
        X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(filtered_df)
        logger.info(f"✅ Preprocessing complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {e}")
        return

    # Proceed with training (tuning and model saving) using your existing function.
    try:
        bayes_best_model_train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selection_metric=selection_metric,
            model_save_dir=model_save_dir,
            classification_save_path=CLASSIFICATION_REPORT_PATH,
            tuning_results_save=TUNING_RESULTS_SAVE_PATH,
            selected_models=selected_models,
            use_pca=True  
        )
    except Exception as e:
        logger.error(f"❌ Model training/tuning failed: {e}")
        return

    logger.info("✅ Training workflow completed successfully.")

if __name__ == "__main__":
    main()
