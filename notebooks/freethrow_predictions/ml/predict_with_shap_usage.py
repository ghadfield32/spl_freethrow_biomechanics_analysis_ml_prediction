import pandas as pd
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pickle
import logging.config
import ast

# Import configuration loader and models
from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
# Import other necessary modules
from ml.train_utils.train_utils import load_model
from ml.feature_selection.feature_importance_calculator import manage_features

# Optionally, import SHAP helpers if needed:
# Local imports for SHAP and model loading should be uncommented and adjusted as needed
from ml.shap.shap_utils import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    generate_global_recommendations,
    generate_individual_feedback,
    plot_shap_force,
    expand_specific_feedback
)

# Assume these are imported in the modules that call them
from datapreprocessor import DataPreprocessor
from ml.predict.predict import predict_and_attach_predict_probs


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def setup_logging(config: AppConfig, log_file_path: Path) -> logging.Logger:
    log_level = config.logging.level.upper()
    # Option 1: Remove file handler entirely (for ease)
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
        },
        'loggers': {
            '__main__': {
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            },
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)


def unpack_feedback(feedback: any) -> None:
    if feedback is None:
        print("No feedback available.")
        return

    if isinstance(feedback, dict):
        feedback_dict = feedback
    elif isinstance(feedback, str):
        try:
            # Print raw feedback for debugging
            print("Raw feedback string (repr):", repr(feedback))
            feedback_dict = ast.literal_eval(feedback)
        except Exception as e:
            print(f"ast.literal_eval failed: {e}")
            return
    else:
        print("Feedback is not in a recognized format.")
        return

    print("Trial Specific Feedback:")
    for metric, suggestion in feedback_dict.items():
        print(f"  {metric}: {suggestion}")


def predict_and_shap(
      config: AppConfig,
      df_input: pd.DataFrame,
      save_dir: Path,
      generate_summary_plot: bool = True,
      generate_dependence_plots: bool = False,
      generate_force_plots_or_feedback_indices: Optional[List[Any]] = None,
      top_n_features: int = 10,
      use_mad: bool = False,
      generate_feedback: bool = False,
      index_column: Optional[str] = None,
      logger: Optional[logging.Logger] = None,
      # Optional overrides for feature file paths:
      features_file: Optional[Path] = None,
      ordinal_file: Optional[Path] = None,
      nominal_file: Optional[Path] = None,
      numericals_file: Optional[Path] = None,
      y_variable_file: Optional[Path] = None,
      model_save_dir_override: Optional[Path] = None,
      transformers_dir_override: Optional[Path] = None
) -> Dict[str, Any]:
    results = {}
    # Use dot‑notation to access configuration values
    data_dir = Path(config.paths.data_dir).resolve()
    model_save_dir = Path(config.paths.model_save_base_dir).resolve() if model_save_dir_override is None else model_save_dir_override.resolve()
    transformers_dir = Path(config.paths.transformers_save_base_dir).resolve() if transformers_dir_override is None else transformers_dir_override.resolve()
    # Use the configuration for feature paths:
    features_file = Path(config.paths.features_metadata_file) if features_file is None else features_file
    # [Set other feature file defaults similarly]

    # Load tuning results, select best model, etc.
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
    model_path = model_save_dir / best_model_name.replace(' ', '_') / "trained_model.pkl"

    # Load feature lists (via manage_features) and initialize DataPreprocessor
    # Build the feature paths dictionary:
    feature_paths = {
        'features': features_file,
        'ordinal_categoricals': ordinal_file,
        'nominal_categoricals': nominal_file,
        'numericals': numericals_file,
        'y_variable': y_variable_file
    }
    try:
        feature_lists = manage_features(mode='load', paths=feature_paths)
        y_variable_list = feature_lists.get('y_variable', [])
        ordinal_categoricals = feature_lists.get('ordinal_categoricals', [])
        nominal_categoricals = feature_lists.get('nominal_categoricals', [])
        numericals = feature_lists.get('numericals', [])
        if logger:
            logger.debug(f"Loaded Feature Lists: y_variable={y_variable_list}, ordinal_categoricals={ordinal_categoricals}, nominal_categoricals={nominal_categoricals}, numericals={numericals}")
    except Exception as e:
        if logger:
            logger.warning(f"Feature lists could not be loaded: {e}")
        y_variable_list, ordinal_categoricals, nominal_categoricals, numericals = [], [], [], []

    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=y_variable_list,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode='predict',
        options={},
        debug=True,
        normalize_debug=False,
        normalize_graphs_output=False,
        graphs_output_dir=Path(config.paths.plots_output_dir).resolve(),
        transformers_dir=transformers_dir
    )
    
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_input)
        if logger:
            logger.info("Preprocessing completed successfully in predict mode.")
    except Exception as e:
        if logger:
            logger.error(f"Preprocessing failed: {e}")
        raise

    duplicates = X_inversed.index.duplicated()
    if duplicates.any():
        print("Duplicate trial IDs found:", X_inversed.index[duplicates].tolist())
    else:
        print("Trial IDs are unique.")

    try:
        model = load_model(best_model_name, model_save_dir)
        if logger:
            logger.info(f"Trained model loaded from '{model_path}'.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to load the best model '{best_model_name}': {e}")
        raise

    try:
        predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(model, X_preprocessed, X_inversed)
        results['predictions'] = predictions
        results['prediction_probs'] = prediction_probs
        if logger:
            logger.info("Predictions generated and attached to the dataset.")
    except Exception as e:
        if logger:
            logger.error(f"Prediction failed: {e}")
        raise

    if index_column is not None:
        if index_column in df_input.columns:
            X_inversed[index_column] = df_input[index_column].values
            X_preprocessed[index_column] = df_input[index_column].values
            X_inversed.set_index(index_column, inplace=True)
            # if index_column in X_inversed.columns:
            #     X_inversed.drop(index_column, axis=1, inplace=True)
            X_preprocessed.set_index(index_column, inplace=True)
            if logger:
                logger.info(f"Using '{index_column}' as the index for both preprocessed and inverse-transformed data.")
        else:
            if logger:
                logger.warning(f"Specified index column '{index_column}' not found in df_input; using default index.")

    try:
        explainer, shap_values = compute_shap_values(model, X_preprocessed, debug=config.logging.debug, logger=logger)
        results['shap_values'] = shap_values
        results['explainer'] = explainer
        results['X_preprocessed'] = X_preprocessed
        if logger:
            logger.info("SHAP values computed successfully.")
            logger.debug(f"Type of shap_values: {type(shap_values)}")
            if hasattr(shap_values, 'shape'):
                logger.debug(f"shap_values.shape: {shap_values.shape}")
            # Log a sample of shap_values
            if isinstance(shap_values, (pd.DataFrame, pd.Series)):
                logger.debug(f"shap_values sample:\n{shap_values.head()}")
            elif isinstance(shap_values, np.ndarray):
                logger.debug(f"shap_values sample:\n{shap_values[:5]}")
    except Exception as e:
        if logger:
            logger.error(f"SHAP computation failed: {e}")
        raise

    if generate_summary_plot:
        shap_summary_path = save_dir / "shap_summary.png"
        try:
            plot_shap_summary(shap_values, X_preprocessed, str(shap_summary_path), debug=config.logging.debug, logger=logger)
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
                                     debug=config.logging.debug, logger=logger)
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
                force_path = force_plots_dir / f"shap_force_plot_{trial_id}.html"  # Unique filename per trial
                plot_shap_force(explainer, shap_values, X_preprocessed, trial_id, force_path,
                                logger=logger, debug=config.logging.debug)

            if logger:
                logger.info(f"SHAP force plots saved at {force_plots_dir}.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to generate SHAP force plots: {e}")

    # --- Generate Individual Feedback for the Entire Dataset if Requested ---
    if generate_feedback:
        try:
            feature_metadata = {}
            for feature in X_inversed.columns:
                unit = 'degrees' if 'angle' in feature.lower() else 'meters'
                feature_metadata[feature] = {'unit': unit}
            
            feedback_dict = {}
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
                if shap_values.shape[0] == 2:
                    adjusted_shap_values = shap_values[1, :, :]
                elif shap_values.shape[1] == 2:
                    adjusted_shap_values = shap_values[:, 1, :]
                else:
                    logger.warning("Unexpected shape for shap_values 3D array. Using the first slice along the first axis.")
                    adjusted_shap_values = shap_values[0, :, :]
            else:
                logger.info("shap_values is assumed to be a 2D array already.")

            logger.debug(f"Type of adjusted_shap_values: {type(adjusted_shap_values)}; shape: {getattr(adjusted_shap_values, 'shape', 'N/A')}")

            # Check if the number of shap_values rows matches X_inversed
            if hasattr(adjusted_shap_values, 'shape'):
                if adjusted_shap_values.shape[0] != len(X_inversed):
                    logger.warning(f"Number of shap values ({adjusted_shap_values.shape[0]}) does not match number of trials ({len(X_inversed)}).")
                else:
                    logger.debug("Number of shap values matches number of trials.")
            else:
                logger.warning(f"adjusted_shap_values does not have a 'shape' attribute.")

            # Reindex shap_values if they are a DataFrame or Series
            if isinstance(adjusted_shap_values, (pd.DataFrame, pd.Series)):
                adjusted_shap_values = adjusted_shap_values.reindex(X_inversed.index)
                logger.debug("Reindexed adjusted_shap_values to match X_inversed index.")

            for trial_id in X_inversed.index:
                try:
                    # Specific Debugging for T0001
                    if trial_id == 'T0001':
                        logger.debug("---- Debugging Trial T0001 ----")
                        logger.debug(f"Trial Features: {X_inversed.loc[trial_id].to_dict()}")
                    
                    # Extract SHAP values using .loc
                    if isinstance(adjusted_shap_values, pd.Series) or isinstance(adjusted_shap_values, pd.DataFrame):
                        shap_values_trial = adjusted_shap_values.loc[trial_id]
                        logger.debug(f"SHAP values for trial_id={trial_id} accessed via .loc.")
                    elif isinstance(adjusted_shap_values, np.ndarray):
                        pos = X_inversed.index.get_loc(trial_id)
                        shap_values_trial = adjusted_shap_values[pos]
                        logger.debug(f"SHAP values for trial_id={trial_id} accessed via numpy indexing at pos={pos}.")
                    else:
                        logger.warning(f"Unsupported type for adjusted_shap_values: {type(adjusted_shap_values)}")
                        raise TypeError(f"Unsupported type for adjusted_shap_values: {type(adjusted_shap_values)}")

                    # Extract trial features
                    trial_features_raw = X_inversed.loc[[trial_id]]
                    logger.debug(f"Processing trial '{trial_id}': trial_features_raw shape={trial_features_raw.shape}, columns={trial_features_raw.columns.tolist()}")

                    trial_features = trial_features_raw.iloc[0]
                    logger.debug(f"Trial '{trial_id}' converted to Series; trial_features shape={trial_features.shape}, index={trial_features.index.tolist()}")

                    # Generate individual feedback
                    fb = generate_individual_feedback(trial_features, shap_values_trial, feature_metadata, logger=logger)
                    feedback_dict[trial_id] = fb
                    X_inversed.at[trial_id, 'specific_feedback'] = fb
                    logger.debug(f"Feedback for trial '{trial_id}' generated successfully: {fb}")

                except Exception as err:
                    logger.warning(f"Error generating feedback for trial '{trial_id}': {err}")
                    # Ensure that even if feedback generation fails, 'specific_feedback' is populated to prevent NaN
                    X_inversed.at[trial_id, 'specific_feedback'] = {}
                    continue

            results['individual_feedback'] = feedback_dict
            if logger:
                logger.info("Individual feedback generated for the entire dataset.")

            # --- Handle 'specific_feedback' Column Dtype Before Filling NaNs ---
            X_inversed['specific_feedback'] = X_inversed['specific_feedback'].astype(object)
            X_inversed.fillna('No feedback available', inplace=True)
            logger.debug("'specific_feedback' column dtype set to object and NaNs filled with 'No feedback available'.")

            # --- Expand 'specific_feedback' into Separate Columns ---
            try:
                X_inversed = expand_specific_feedback(X_inversed, logger=logger)
                results['final_dataset'] = X_inversed
                if logger:
                    logger.info("'specific_feedback' column expanded into separate feedback columns.")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to expand 'specific_feedback': {e}")
                raise

        except Exception as e:
            if logger:
                logger.error("Failed to generate individual feedback: %s", e)

    # --- Save Final Dataset and Global Recommendations ---
    try:
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
        config: AppConfig = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    data_dir = Path(config.paths.data_dir).resolve()
    raw_data_path = data_dir / config.paths.raw_data
    predictions_output_path = Path(config.paths.predictions_output_dir).resolve()
    predictions_output_path = predictions_output_path / "shap_results"

    log_dir = Path(config.paths.log_dir).resolve()
    log_file = config.paths.log_file

    logger = setup_logging(config, log_dir / log_file)
    logger.info("Starting prediction module (unified predict_and_shap).")
    logger.debug(f"Paths: {config.paths}")

    try:
        df_predict = load_dataset(raw_data_path)
        print("Columns in input data:", df_predict.columns.tolist())
        logger.info(f"Prediction input data loaded from {raw_data_path}.")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return
    base_dir = Path("../../data") / "preprocessor" / "features_info"
    try:
        results = predict_and_shap(
            config=config,
            df_input=df_predict,
            save_dir=predictions_output_path,
            generate_summary_plot=True,
            generate_dependence_plots=True,
            generate_force_plots_or_feedback_indices=['T0001'],
            top_n_features=len(df_predict.columns),
            use_mad=False,
            generate_feedback=True,
            index_column="trial_id",
            logger=logger,
            features_file = (Path(config.paths.data_dir) / config.paths.features_metadata_file).resolve(),
            ordinal_file=Path(f'{base_dir}/ordinal_categoricals.pkl'),
            nominal_file=Path(f'{base_dir}/nominal_categoricals.pkl'),
            numericals_file=Path(f'{base_dir}/numericals.pkl'),
            y_variable_file=Path(f'{base_dir}/y_variable.pkl'),
            model_save_dir_override=Path(config.paths.model_save_base_dir),
            transformers_dir_override=Path(config.paths.transformers_save_base_dir)
        )
        logger.info("Unified predict_and_shap function executed successfully.")
    except Exception as e:
        logger.error(f"Unified predict_and_shap function failed: {e}")
        return

    try:
        print("\nFinal Predictions with SHAP annotations (preview):")
        final_df = pd.read_csv(results['final_dataset'], index_col=0)
        print(final_df.head())

        # Debug: Print columns in final_df
        logger.debug(f"Final DataFrame columns: {final_df.columns.tolist()}")

        trial_id = 'T0002'
        if trial_id in final_df.index:
            # Select all 'shap_' columns
            shap_columns = [col for col in final_df.columns if col.startswith('shap_')]
            logger.debug(f"'shap_' columns for feedback: {shap_columns}")

            # Ensure there are 'shap_' columns
            if not shap_columns:
                logger.error("No 'shap_' columns found in the final DataFrame.")
                print("No feedback columns found in the final DataFrame.")
            else:
                feedback_entry = final_df.loc[trial_id, shap_columns].to_dict()
                logger.debug(f"Feedback entry for {trial_id}: {feedback_entry}")

                print(f"Feedback for trial {trial_id}:")
                unpack_feedback(feedback_entry)
        else:
            print(f"No feedback found for trial {trial_id}.")

    except Exception as e:
        logger.error(f"Failed to display outputs: {e}")

    # Check for missing feedback across all 'shap_' columns
    try:
        shap_columns = [col for col in final_df.columns if col.startswith('shap_')]
        if shap_columns:
            null_feedback = final_df[final_df[shap_columns].isnull().any(axis=1)]
            if not null_feedback.empty:
                print(f"Trials with null feedback: {null_feedback.index.tolist()}")
            else:
                print("All trials have complete feedback.")
        else:
            print("No 'shap_' columns found to check for feedback completeness.")
    except KeyError:
        print("No 'shap_' columns found in the final DataFrame.")

    # Additional Debugging: Check Feedback for T0001
    try:
        trial_id = 'T0001'
        if trial_id in final_df.index:
            shap_columns = [col for col in final_df.columns if col.startswith('shap_')]
            feedback_entry = final_df.loc[trial_id, shap_columns].to_dict()
            logger.debug(f"Feedback entry for {trial_id}: {feedback_entry}")

            print(f"\nFeedback for trial {trial_id}:")
            unpack_feedback(feedback_entry)
        else:
            print(f"No feedback found for trial {trial_id}.")
    except Exception as e:
        logger.error(f"Failed to display feedback for trial {trial_id}: {e}")


if __name__ == "__main__":
    main()
