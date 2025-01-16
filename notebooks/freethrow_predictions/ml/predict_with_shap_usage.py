
import pandas as pd
import logging
import os
import yaml
import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from ml.predict.predict import predict_and_attach_predict_probs
from datapreprocessor import DataPreprocessor
from ml.feature_selection.feature_importance_calculator import manage_features

# Make sure to import your SHAP helpers:
from ml.shap.shap_utils import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    generate_global_recommendations,
    generate_individual_feedback,
    plot_shap_force
)
from ml.train_utils.train_utils import load_model

from jinja2 import Environment, FileSystemLoader
import logging.config
import shap
import matplotlib.pyplot as plt
import pickle


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


def setup_logging(config: Dict[str, Any], log_file_path: Path) -> logging.Logger:
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': str(log_file_path),
                'maxBytes': 10**6,
                'backupCount': 5,
                'encoding': 'utf8',
            },
        },
        'loggers': {
            '__main__': {
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)


def load_preprocessor_from_path(transformer_path: Path):
    with open(transformer_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor


# Unified function that orchestrates prediction and SHAP.
def predict_and_shap(
      config: Dict[str, Any],
      df_input: pd.DataFrame,
      save_dir: Path,
      generate_summary_plot: bool = True,
      generate_dependence_plots: bool = False,
      generate_force_plots_or_feedback_indices: Optional[List[Any]] = None,
      top_n_features: int = 10,
      use_mad: bool = False,
      generate_feedback: bool = False,
      index_column: Optional[str] = None,
      logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    results = {}

    # --- Determine Paths from Configuration ---
    paths = config.get('paths', {})
    feature_paths = {
        'features': Path('../../data/model/pipeline/final_ml_df_selected_features_columns_test.pkl'),
        'ordinal_categoricals': Path('../../data/model/pipeline/features_info/ordinal_categoricals.pkl'),
        'nominal_categoricals': Path('../../data/model/pipeline/features_info/nominal_categoricals.pkl'),
        'numericals': Path('../../data/model/pipeline/features_info/numericals.pkl'),
        'y_variable': Path('../../data/model/pipeline/features_info/y_variable.pkl')
    }
    data_dir = Path(paths.get('data_dir', '../../data/processed')).resolve()
    model_save_dir = Path(paths.get('model_save_base_dir', '../preprocessor/models')).resolve()
    transformers_dir = Path(paths.get('transformers_save_base_dir', '../preprocessor/transformers')).resolve()

    # --- Load Tuning Results and Determine Best Model ---
    tuning_results_path = model_save_dir / "tuning_results.json"
    if not tuning_results_path.exists():
        raise FileNotFoundError(f"Tuning results not found at '{tuning_results_path}'.")
    with open(tuning_results_path, 'r') as f:
        tuning_results = json.load(f)
    best_model_info = tuning_results.get("Best Model")
    if not best_model_info:
        raise ValueError("Best model information not found in tuning results.")
    best_model_name = best_model_info.get("model_name")
    if not best_model_name:
        raise ValueError("Best model name not found in tuning results.")
    if logger:
        logger.info(f"Best model identified: {best_model_name}")
        logger.debug(f"Best model details: {best_model_info}")
    model_path = model_save_dir / best_model_name.replace(' ', '_') / "trained_model.pkl"

    # --- Set Transformer Path ---
    transformer_path = transformers_dir / "transformers.pkl"

    # --- Load Feature Lists ---
    try:
        feature_lists = manage_features(mode='load', paths=feature_paths)
        y_variable = feature_lists.get('y_variable', [])
        ordinal_categoricals = feature_lists.get('ordinal_categoricals', [])
        nominal_categoricals = feature_lists.get('nominal_categoricals', [])
        numericals = feature_lists.get('numericals', [])
        if logger:
            logger.debug(f"Loaded Feature Lists: y_variable={y_variable}, "
                         f"ordinal_categoricals={ordinal_categoricals}, "
                         f"nominal_categoricals={nominal_categoricals}, "
                         f"numericals={numericals}")
    except Exception as e:
        if logger:
            logger.warning(f"Feature lists could not be loaded: {e}")
        y_variable, ordinal_categoricals, nominal_categoricals, numericals = [], [], [], []

    # --- Use DataPreprocessor ---
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode='predict',
        options={},
        debug=True,
        normalize_debug=False,
        normalize_graphs_output=False,
        graphs_output_dir=Path(paths.get('plots_output_dir', '../preprocessor/plots')).resolve(),
        transformers_dir=transformers_dir
    )
    
    # --- Execute Preprocessing ---
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_input)
        if logger:
            logger.info("Preprocessing completed successfully in predict mode.")
    except Exception as e:
        if logger:
            logger.error(f"Preprocessing failed: {e}")
        raise

    # --- Load Best Model Once ---
    try:
        model = load_model(best_model_name, model_save_dir)
        if logger:
            logger.info(f"Trained model loaded from '{model_path}'.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to load the best model '{best_model_name}': {e}")
        raise

    # --- Make Predictions and Attach Probabilities ---
    try:
        predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(
            trained_model=model, 
            X_preprocessed=X_preprocessed, 
            X_inversed=X_inversed
        )
        results['predictions'] = predictions
        results['prediction_probs'] = prediction_probs
        if logger:
            logger.info("Predictions generated and attached to the dataset.")
    except Exception as e:
        if logger:
            logger.error(f"Prediction failed: {e}")
        raise

    # --- Set the unique ID as the index (for row-level feedback and lookup) ---
    if index_column is not None:
        if index_column in df_input.columns:
            # Add the ID column into the processed DataFrames if not already present.
            X_inversed[index_column] = df_input[index_column].values
            X_preprocessed[index_column] = df_input[index_column].values
            # Set the DataFrame index to the unique identifier.
            X_inversed.set_index(index_column, inplace=True)
            X_preprocessed.set_index(index_column, inplace=True)
            if logger:
                logger.info(f"Using '{index_column}' as the index for both preprocessed and inverse-transformed data.")
                logger.debug(f"X_inversed index: {X_inversed.index.unique().tolist()}")
            # Create the specific_feedback column if it does not exist.
            if 'specific_feedback' not in X_inversed.columns:
                X_inversed['specific_feedback'] = None
        else:
            if logger:
                logger.warning(f"Specified index column '{index_column}' not found in df_input; using default index.")

    # --- Compute SHAP Values ---
    try:
        explainer, shap_values = compute_shap_values(
            model,
            X_preprocessed,
            debug=config.get('logging', {}).get('debug', False),
            logger=logger
        )
        results['shap_values'] = shap_values
        results['explainer'] = explainer
        if logger:
            logger.info("SHAP values computed successfully.")
            logger.debug(f"Type of shap_values: {type(shap_values)}")
            if hasattr(shap_values, 'shape'):
                logger.debug(f"shap_values.shape: {shap_values.shape}")
    except Exception as e:
        if logger:
            logger.error(f"SHAP computation failed: {e}")
        raise

    # --- Generate SHAP Summary Plot ---
    if generate_summary_plot:
        shap_summary_path = save_dir / "shap_summary.png"
        try:
            plot_shap_summary(shap_values, X_preprocessed, str(shap_summary_path),
                              debug=config.get('logging', {}).get('debug', False), logger=logger)
            results['shap_summary_plot'] = str(shap_summary_path)
            if logger:
                logger.info(f"SHAP summary plot saved at {shap_summary_path}.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to generate SHAP summary plot: {e}")

    # --- Generate SHAP Dependence Plots ---
    recommendations_dict = {}
    if generate_dependence_plots:
        try:
            recommendations_dict = generate_global_recommendations(
                shap_values=shap_values,
                X_original=X_preprocessed,
                top_n=top_n_features,
                use_mad=use_mad,
                logger=logger
            )
            results['recommendations'] = recommendations_dict
            shap_dependence_dir = save_dir / "shap_dependence_plots"
            os.makedirs(shap_dependence_dir, exist_ok=True)
            for feature in recommendations_dict.keys():
                dep_path = shap_dependence_dir / f"shap_dependence_{feature}.png"
                plot_shap_dependence(shap_values, feature, X_preprocessed, str(dep_path),
                                     debug=config.get('logging', {}).get('debug', False), logger=logger)
            if logger:
                logger.info(f"SHAP dependence plots saved at {shap_dependence_dir}.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to generate SHAP dependence plots: {e}")

    # --- Optionally Generate SHAP Force Plots for Specified IDs ---
    if generate_force_plots_or_feedback_indices is not None:
        try:
            force_plots_dir = save_dir / "shap_force_plots"
            force_plots_dir.mkdir(exist_ok=True, parents=True)
            for trial_id in generate_force_plots_or_feedback_indices:
                force_path = force_plots_dir / f"shap_force_trial_{trial_id}.png"
                plot_shap_force(explainer, shap_values, X_preprocessed, trial_id, force_path,
                                logger=logger, debug=config.get('logging', {}).get('debug', False))
            if logger:
                logger.info(f"SHAP force plots saved at {force_plots_dir}.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to generate SHAP force plots: {e}")

    # --- Generate Individual Feedback for the Entire Dataset if Requested ---
    if generate_feedback:
        try:
            # Build feature metadata based on column names
            feature_metadata = {}
            for feature in X_inversed.columns:
                unit = 'degrees' if 'angle' in feature.lower() else 'meters'
                feature_metadata[feature] = {'unit': unit}
            
            feedback_dict = {}
            # Adjust SHAP values if needed (same logic as before)
            adjusted_shap_values = shap_values
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    logger.info("Detected shap_values as a list of two arrays; using shap_values[1] for positive class.")
                    adjusted_shap_values = shap_values[1]
                else:
                    logger.warning("shap_values is a list but not of length 2. Using the first element.")
                    adjusted_shap_values = shap_values[0]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                logger.info(f"shap_values is a 3D array with shape {shap_values.shape}. Attempting extraction based on axis.")
                if shap_values.shape[0] == 2:  # shape (2, n_samples, n_features)
                    adjusted_shap_values = shap_values[1, :, :]
                elif shap_values.shape[1] == 2:  # shape (n_samples, 2, n_features)
                    adjusted_shap_values = shap_values[:, 1, :]
                else:
                    logger.warning("Unexpected shape for shap_values 3D array. Using the first slice along the first axis.")
                    adjusted_shap_values = shap_values[0, :, :]
            else:
                logger.info("shap_values is assumed to be a 2D array already.")

            logger.debug(f"Type of adjusted_shap_values: {type(adjusted_shap_values)}")

            # Loop over every row (each trial id) in X_inversed
            for trial_id in X_inversed.index:
                try:
                    trial_features = X_inversed.loc[trial_id]
                except Exception as err:
                    if logger:
                        logger.warning(f"Cannot extract features for trial '{trial_id}': {err}")
                    continue

                try:
                    pos = X_inversed.index.get_loc(trial_id)
                    # If pos is a multi-index location, choose the first element
                    if isinstance(pos, (list, pd.Index, np.ndarray)):
                        logger.warning(f"Trial id '{trial_id}' yielded a multi-index location {pos}. Using the first element.")
                        pos = pos[0]
                except Exception as err:
                    if logger:
                        logger.warning(f"Error determining position for trial '{trial_id}': {err}")
                    continue

                try:
                    # Extract the SHAP values corresponding to that row.
                    if isinstance(adjusted_shap_values, (pd.Series, pd.DataFrame)):
                        shap_values_trial = adjusted_shap_values.iloc[pos]
                    else:
                        shap_values_trial = adjusted_shap_values[pos]
                    logger.debug(f"shap_values_trial for trial '{trial_id}' has shape: {getattr(shap_values_trial, 'shape', 'N/A')}")
                except Exception as err:
                    if logger:
                        logger.warning(f"Error extracting SHAP values for trial '{trial_id}' at position {pos}: {err}")
                    continue

                try:
                    # Generate feedback using your helper function.
                    fb = generate_individual_feedback(trial_features, shap_values_trial, feature_metadata, logger=logger)
                    feedback_dict[trial_id] = fb
                    # Store feedback in the DataFrame so every row has its feedback.
                    X_inversed.at[trial_id, 'specific_feedback'] = fb
                except Exception as err:
                    if logger:
                        logger.warning(f"Error generating feedback for trial '{trial_id}': {err}")
                    continue

            results['individual_feedback'] = feedback_dict
            if logger:
                logger.info("Individual feedback generated for the entire dataset.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to generate individual feedback: {e}")

    # --- Save Final Dataset and Global Recommendations ---
    try:
        save_dir.mkdir(exist_ok=True, parents=True)
        # Save the final DataFrame with the index (trial id) preserved.
        final_dataset_path = save_dir / "final_predictions_with_shap.csv"
        X_inversed.to_csv(final_dataset_path, index=True)
        results['final_dataset'] = str(final_dataset_path)
        if recommendations_dict:
            recs_path = save_dir / "global_shap_recommendations.json"
            with open(recs_path, "w") as f:
                json.dump(recommendations_dict, f, indent=4)
            results['recommendations_file'] = str(recs_path)
        if logger:
            logger.info("Final dataset and global recommendations saved.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save outputs: {e}")
        raise

    if logger:
        logger.info("Predict+SHAP pipeline completed successfully.")
    return results


def main():
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    paths = config.get('paths', {})
    data_dir = Path(paths.get('data_dir', '../../data/processed')).resolve()
    raw_data_path = data_dir / paths.get('raw_data', 'final_ml_dataset.csv')
    predictions_output_path = Path(paths.get('predictions_output_dir', 'preprocessor/predictions')).resolve()
    log_dir = Path(paths.get('log_dir', '../preprocessor/logs')).resolve()
    log_file = paths.get('log_file', 'prediction.log')

    logger = setup_logging(config, log_dir / log_file)
    logger.info("Starting prediction module (unified predict_and_shap).")
    logger.debug(f"Paths: {paths}")

    try:
        df_predict = load_dataset(raw_data_path)
        print("Columns in input data:", df_predict.columns.tolist())
        logger.info(f"Prediction input data loaded from {raw_data_path}.")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    try:
        # In this example, we also pass a list to generate force plots (if desired)
        # but note that individual feedback will be generated for the entire dataset.
        results = predict_and_shap(
            config=config,
            df_input=df_predict,
            save_dir=predictions_output_path,
            generate_summary_plot=True,
            generate_dependence_plots=True,
            generate_force_plots_or_feedback_indices=['T0125'],  # Optional: used for force plots
            top_n_features=len(df_predict.columns),
            use_mad=False,
            generate_feedback=True,
            index_column="trial_id",
            logger=logger
        )

        logger.info("Unified predict_and_shap function executed successfully.")
    except Exception as e:
        logger.error(f"Unified predict_and_shap function failed: {e}")
        return

    try:
        print("\nFinal Predictions with SHAP annotations (preview):")
        # Read the final dataset; the index is preserved so you can do:
        final_df = pd.read_csv(results['final_dataset'], index_col=0)
        print(final_df.head())
        # For example, pulling the specific feedback for a particular trial:
        trial_id = 'T0125'  # or any valid id present
        if trial_id in final_df.index:
            print(f"\nFeedback for trial {trial_id}:")
            print(final_df.loc[trial_id, 'specific_feedback'])
        print("\nResults keys:")
        print(results.keys())
    except Exception as e:
        logger.error(f"Failed to display outputs: {e}")


if __name__ == "__main__":
    main()
