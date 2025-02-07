
import pandas as pd
import logging
import os
import yaml
import joblib
import json
from pathlib import Path
from typing import Any, Dict

from ml.predict.predict import predict_and_attach_predict_probs
from datapreprocessor import DataPreprocessor
from ml.feature_selection.feature_importance_calculator import manage_features  # New Import

# Local imports for SHAP and model loading should be uncommented and adjusted as needed
from ml.shap.shap_utils import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    generate_global_recommendations,
    generate_individual_feedback
)
from ml.train_utils.train_utils import load_model  # Ensure 'train_utils.py' contains load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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





def main():
    # ----------------------------
    # Step 1: Load Configuration
    # ----------------------------
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')  # Adjust as needed
    try:
        config = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return  # Exit if config loading fails

    # ----------------------------
    # Step 2: Extract Paths from Configuration
    # ----------------------------
    paths = config.get('paths', {})
    feature_paths = {
        'features': Path('../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl'),
        'ordinal_categoricals': Path('../../data/preprocessor/features_info/ordinal_categoricals.pkl'),
        'nominal_categoricals': Path('../../data/preprocessor/features_info/nominal_categoricals.pkl'),
        'numericals': Path('../../data/preprocessor/features_info/numericals.pkl'),
        'y_variable': Path('../../data/preprocessor/features_info/y_variable.pkl')
    }

    # Define other necessary paths
    data_dir = Path(paths.get('data_dir', '../../data/processed')).resolve()
    raw_data_path = data_dir / paths.get('raw_data', 'final_ml_dataset.csv')  # Corrected key
    processed_data_dir = data_dir / paths.get('processed_data_dir', 'preprocessor/processed')
    transformers_dir = Path(paths.get('transformers_save_base_dir', '../preprocessor/transformers')).resolve()  # Corrected key
    predictions_output_path = Path(paths.get('predictions_output_dir', 'preprocessor/predictions')).resolve()
    log_dir = Path(paths.get('log_dir', '../preprocessor/logs')).resolve()
    model_save_dir = Path(paths.get('model_save_base_dir', '../preprocessor/models')).resolve()  # Corrected key
    log_file = paths.get('log_file', 'prediction.log')  # Ensure this key exists in config

    # ----------------------------
    # Step 3: Setup Logging
    # ----------------------------
    logger.info("✅ Starting prediction module.")
    logger.debug(f"Configuration paths extracted: {paths}")

    # ----------------------------
    # Step 4: Load Feature Lists Using manage_features
    # ----------------------------
    try:
        feature_lists = manage_features(
            mode='load',
            paths=feature_paths
        )
        y_variable = feature_lists.get('y_variable', [])
        ordinal_categoricals = feature_lists.get('ordinal_categoricals', [])
        nominal_categoricals = feature_lists.get('nominal_categoricals', [])
        numericals = feature_lists.get('numericals', [])

        logger.debug(f"Loaded Feature Lists: y_variable={y_variable}, "
                     f"ordinal_categoricals={ordinal_categoricals}, "
                     f"nominal_categoricals={nominal_categoricals}, "
                     f"numericals={numericals}")
    except Exception as e:
        logger.error(f"❌ Failed to load feature lists: {e}")
        return

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
        logger.debug(f"Best model details: {best_model_info}")
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
        logger.debug(f"Prediction input data shape: {df_predict.shape}")
        logger.debug(f"Prediction input data columns: {df_predict.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Failed to load prediction input data: {e}")
        return

    # Initialize DataPreprocessor with Loaded Feature Lists
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",  # Adjust if model type varies
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode='predict',
        options={},  # Populate based on specific requirements or configurations
        debug=True,  # Enable debug mode for detailed logs
        normalize_debug=False,  # As per requirements
        normalize_graphs_output=False,  # As per requirements
        graphs_output_dir=Path(paths.get('plots_output_dir', '../preprocessor/plots')).resolve(),
        transformers_dir=transformers_dir
    )

    # Execute Preprocessing for Prediction
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_predict)
        print("X_new_preprocessed type = ", type(X_preprocessed), "X_new_inverse type = ", type(X_inversed))
        logger.info("✅ Preprocessing completed successfully in predict mode.")
        logger.debug(f"Shape of X_preprocessed: {X_preprocessed.shape}")
        logger.debug(f"Columns in X_preprocessed: {X_preprocessed.columns.tolist()}")
        logger.debug(f"Shape of X_inversed: {X_inversed.shape}")
        logger.debug(f"Columns in X_inversed: {X_inversed.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Preprocessing failed in predict mode: {e}")
        return

    # ----------------------------
    # Step 7: Load the Best Model
    # ----------------------------
    try:
        trained_model = load_model(best_model_name, model_save_dir)
        model_path = model_save_dir / best_model_name.replace(' ', '_') / 'trained_model.pkl'
        logger.info(f"✅ Trained model loaded from '{model_path}'.")
        logger.debug(f"Trained model type: {type(trained_model)}")
    except Exception as e:
        logger.error(f"❌ Failed to load the best model '{best_model_name}': {e}")
        return

    # ----------------------------
    # Step 8: Make Predictions + Prediction Probabilities and add them to the inverse-transformed DataFrame
    # ----------------------------
    predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(trained_model, X_preprocessed, X_inversed)


    # ----------------------------
    # Step 12: Compute SHAP Values for Interpretability
    # ----------------------------
    try:
        logger.info("📊 Computing SHAP values for interpretability...")
        explainer, shap_values = compute_shap_values(
            trained_model, 
            X_preprocessed,  # Compute SHAP on preprocessed data
            debug=config.get('logging', {}).get('debug', False), 
            logger=logger
        )
        logger.info("✅ SHAP values computed successfully.")
        
        # Additional Debugging
        logger.debug(f"Type of shap_values: {type(shap_values)}")
        logger.debug(f"Shape of shap_values: {shap_values.shape}")
        if hasattr(shap_values, 'feature_names'):
            logger.debug(f"SHAP feature names: {shap_values.feature_names}")
        else:
            logger.debug("shap_values does not have 'feature_names' attribute.")
        
        logger.debug(f"Type of X_preprocessed: {type(X_preprocessed)}")
        logger.debug(f"Shape of X_preprocessed: {X_preprocessed.shape}")
        logger.debug(f"Columns in X_preprocessed: {X_preprocessed.columns.tolist()}")
        
        # Check feature alignment
        if hasattr(shap_values, 'feature_names'):
            shap_feature_names = shap_values.feature_names
        else:
            shap_feature_names = X_preprocessed.columns.tolist()
        
        if list(shap_feature_names) != list(X_preprocessed.columns):
            logger.error("Column mismatch between SHAP values and X_preprocessed.")
            logger.error(f"SHAP feature names ({len(shap_feature_names)}): {shap_feature_names}")
            logger.error(f"X_preprocessed columns ({len(X_preprocessed.columns)}): {X_preprocessed.columns.tolist()}")
            raise ValueError("Column mismatch between SHAP values and X_preprocessed.")
        else:
            logger.debug("Column alignment verified between SHAP values and X_preprocessed.")
        
        # Ensure all features are numeric
        X_preprocessed = X_preprocessed.apply(pd.to_numeric, errors='coerce')
        logger.debug("Converted all features in X_preprocessed to numeric types.")
        
        # Check for non-numeric columns
        non_numeric_cols = X_preprocessed.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_cols:
            logger.error(f"Non-numeric columns found in X_preprocessed: {non_numeric_cols}")
            raise ValueError("All features must be numeric for SHAP plotting.")
        else:
            logger.debug("All features in X_preprocessed are numeric.")
        
        # Check for missing values in shap_values
        if hasattr(shap_values, 'values'):
            if pd.isnull(shap_values.values).any():
                logger.warning("SHAP values contain NaNs.")
            else:
                logger.debug("SHAP values do not contain NaNs.")
        else:
            logger.warning("shap_values do not have 'values' attribute.")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute SHAP values: {e}")
        return

    # ----------------------------
    # Step 13: Generate SHAP Summary Plot
    # ----------------------------
    try:
        shap_summary_path = predictions_output_path / 'shap_summary.png'
        logger.debug(f"SHAP summary plot will be saved to: {shap_summary_path}")
        
        # Pass X_preprocessed to SHAP plotting functions
        plot_shap_summary(
            shap_values=shap_values, 
            X_original=X_preprocessed,  # Use preprocessed data that matches SHAP values
            save_path=shap_summary_path, 
            debug=config.get('logging', {}).get('debug', False), 
            logger=logger
        )
        logger.info(f"✅ SHAP summary plot saved to '{shap_summary_path}'.")
    except Exception as e:
        logger.error(f"❌ Failed to generate SHAP summary plot: {e}")
        return

    # ----------------------------
    # Step 14: Generate SHAP Dependence Plots for Top Features
    # ----------------------------
    try:
        top_n = len(X_preprocessed.columns)  # Define how many top features you want
        logger.debug(f"Generating SHAP dependence plots for top {top_n} features.")
        recommendations = generate_global_recommendations(
            shap_values=shap_values, 
            X_original=X_preprocessed,  # Use preprocessed data
            top_n=top_n, 
            debug=config.get('logging', {}).get('debug', False), 
            use_mad=False,  # Set to True if using MAD for range definitions
            logger=logger
        )
        shap_dependence_dir = predictions_output_path / 'shap_dependence_plots'
        os.makedirs(shap_dependence_dir, exist_ok=True)
        logger.debug(f"SHAP dependence plots will be saved to: {shap_dependence_dir}")
        for feature in recommendations.keys():
            shap_dependence_filename = f"shap_dependence_{feature}.png"
            shap_dependence_path = shap_dependence_dir / shap_dependence_filename
            logger.debug(f"Generating SHAP dependence plot for feature '{feature}' at '{shap_dependence_path}'.")
            plot_shap_dependence(
                shap_values=shap_values, 
                feature=feature, 
                X_original=X_preprocessed,  # Use preprocessed data
                save_path=shap_dependence_path, 
                debug=config.get('logging', {}).get('debug', False), 
                logger=logger
            )
        logger.info(f"✅ SHAP dependence plots saved to '{shap_dependence_dir}'.")
    except Exception as e:
        logger.error(f"❌ Failed to generate SHAP dependence plots: {e}")
        return

    # ----------------------------
    # Step 15: Annotate Final Dataset with SHAP Recommendations
    # ----------------------------
    try:
        logger.info("🔍 Annotating final dataset with SHAP recommendations...")
        feature_metadata = {}
        for feature in X_preprocessed.columns:
            if 'angle' in feature.lower():
                unit = 'degrees'
            else:
                unit = 'meters'
            feature_metadata[feature] = {'unit': unit}

        specific_feedback_list = []
        for idx in X_inversed.index:
            trial = X_inversed.loc[idx]
            shap_values_trial = shap_values[idx]  # Removed .values
            feedback = generate_individual_feedback(trial, shap_values_trial, feature_metadata, logger=logger)
            specific_feedback_list.append(feedback)
            if config.get('logging', {}).get('debug', False):
                logger.debug(f"Generated feedback for trial {idx}: {feedback}")
        X_inversed['specific_feedback'] = specific_feedback_list

        logger.info("✅ Specific feedback generated for all trials.")
    except Exception as e:
        logger.error(f"❌ Failed to generate specific feedback: {e}")
        return

    # ----------------------------
    # Step 16: Save the Final Dataset with Predictions and SHAP Annotations
    # ----------------------------
    try:
        os.makedirs(predictions_output_path, exist_ok=True)
        final_output_path = predictions_output_path / 'final_predictions_with_shap.csv'
        X_inversed.to_csv(final_output_path, index=False)
        logger.info(f"✅ Final dataset with predictions and SHAP annotations saved to '{final_output_path}'.")
        logger.debug(f"Final dataset shape: {X_inversed.shape}")
        logger.debug(f"Final dataset columns: {X_inversed.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ Failed to save the final dataset: {e}")
        return

    # ----------------------------
    # Step 17: Save Global Recommendations as JSON
    # ----------------------------
    try:
        recommendations_filename = "global_shap_recommendations.json"
        recommendations_save_path = predictions_output_path / recommendations_filename
        with open(recommendations_save_path, "w") as f:
            json.dump(recommendations, f, indent=4)
        logger.info(f"✅ Global SHAP recommendations saved to '{recommendations_save_path}'.")
        logger.debug(f"Global SHAP recommendations saved.")
    except Exception as e:
        logger.error(f"❌ Failed to save global SHAP recommendations: {e}")
        return

    # ----------------------------
    # Step 18: Generate Specific Feedback for Each Trial
    # ----------------------------
    # Note: This step was moved above in the script for logical flow.

    # ----------------------------
    # Step 19: Display Minimal Outputs
    # ----------------------------
    try:
        # Display the first few rows of the final dataset
        print("\nInverse Transformed Prediction DataFrame with Predictions and SHAP Annotations:")
        print(X_inversed.head())

        # Display global recommendations
        print("\nGlobal SHAP Recommendations:")
        for feature, rec in recommendations.items():
            print(f"{feature}: Range={rec['range']}, Importance={rec['importance']}, Direction={rec['direction']}")

        # Display specific feedback for each trial
        print("\nSpecific Feedback for Each Trial:")
        for idx, row in X_inversed.iterrows():
            print(f"Trial {idx + 1}:")
            for feature, feedback in row['specific_feedback'].items():
                print(f"  - {feedback}")
            print()
    except Exception as e:
        logger.error(f"❌ Error during displaying outputs: {e}")
        return

    logger.info("✅ Prediction pipeline with SHAP analysis completed successfully.")


if __name__ == "__main__":
    main()
