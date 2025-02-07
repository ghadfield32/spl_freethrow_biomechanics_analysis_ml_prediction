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

# Assume these are imported in the modules that call them
from datapreprocessor import DataPreprocessor
from ml.predict.predict import predict_and_attach_predict_probs

# Import utility functions from our SHAP modules
from ml.shap.predict_with_shap_usage_utils import compute_original_metric_error, generate_feedback_and_expand
from ml.shap.shap_utils import load_dataset, setup_logging, load_configuration, initialize_logger

# Import SHAP helper classes
from ml.shap.shap_calculator import ShapCalculator
from ml.shap.shap_visualizer import ShapVisualizer
from ml.shap.feedback_generator import FeedbackGenerator

def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def predict_and_shap(
      config: AppConfig,
      df_input: pd.DataFrame,
      save_dir: Path,
      columns_to_add: Optional[List[str]] = None,
      generate_summary_plot: bool = True,
      generate_dependence_plots: bool = False,
      generate_force_plots: bool = False,
      force_plot_indices: Optional[List[int]] = None,
      top_n_features: int = 10,
      use_mad: bool = False,
      logger: Optional[logging.Logger] = None,
      # Optional overrides for feature file paths:
      features_file: Optional[Path] = None,
      ordinal_file: Optional[Path] = None,
      nominal_file: Optional[Path] = None,
      numericals_file: Optional[Path] = None,
      y_variable_file: Optional[Path] = None,
      model_save_dir_override: Optional[Path] = None,
      transformers_dir_override: Optional[Path] = None,
      metrics_percentile: float = 10,
      override_model_name: Optional[str] = None  # <<-- NEW parameter to override the best model name
) -> Dict[str, Any]:
    """
    Perform prediction and SHAP analysis on the input DataFrame.
    (Docstring unchanged for brevity.)
    """
    results = {}
    # Use dot‑notation to access configuration values.
    data_dir = Path(config.paths.data_dir).resolve()
    model_save_dir = (Path(config.paths.model_save_base_dir).resolve() 
                      if model_save_dir_override is None 
                      else model_save_dir_override.resolve())
    transformers_dir = (Path(config.paths.transformers_save_base_dir).resolve() 
                        if transformers_dir_override is None 
                        else transformers_dir_override.resolve())
    # Use configuration for feature paths.
    features_file = Path(config.paths.features_metadata_file) if features_file is None else features_file
    ordinal_file = Path(config.paths.ordinal_categoricals_file) if ordinal_file is None else ordinal_file
    nominal_file = Path(config.paths.nominal_categoricals_file) if nominal_file is None else nominal_file
    numericals_file = Path(config.paths.numericals_file) if numericals_file is None else numericals_file
    y_variable_file = Path(config.paths.y_variable_file) if y_variable_file is None else y_variable_file

    # Load tuning results and select the best model.
    tuning_results_path = model_save_dir / "tuning_results.json"
    print(f"tuning_results_path: {tuning_results_path}")
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
    # <<-- If an override is provided, use that model name instead.
    if override_model_name:
        best_model_name = override_model_name
        if logger:
            logger.info(f"Overriding best model selection; using model: {best_model_name}")
    else:
        if logger:
            logger.info(f"Best model identified from tuning results: {best_model_name}")

    # Load the best model from the consistent save directory.
    try:
        model = load_model(best_model_name, model_save_dir)
        if logger:
            logger.info(f"Trained model loaded from '{model_save_dir}' using model name '{best_model_name}'.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to load the best model '{best_model_name}': {e}")
        raise

    # Load feature lists via manage_features.
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

    # Initialize the SHAP helper classes.
    from ml.shap.shap_calculator import ShapCalculator  # import here if needed
    from ml.shap.feedback_generator import FeedbackGenerator
    from ml.shap.shap_visualizer import ShapVisualizer
    shap_calculator = ShapCalculator(model=model, logger=logger)
    feedback_generator = FeedbackGenerator(logger=logger)
    shap_visualizer = ShapVisualizer(logger=logger)

    # Initialize DataPreprocessor.
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
    
    # Preprocess the input DataFrame.
    X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df_input)
    logger.info("Preprocessing completed successfully in predict mode.")

    # Reindex inverse-transformed data to match the preprocessed data.
    X_inversed = X_inversed.reindex(X_preprocessed.index)
    logger.debug(f"X_preprocessed index: {X_preprocessed.index.tolist()}")
    logger.debug(f"X_inversed index after reindexing: {X_inversed.index.tolist()}")

    # Compute predictions and attach them.
    predictions, prediction_probs, X_inversed = predict_and_attach_predict_probs(model, X_preprocessed, X_inversed)
    results['predictions'] = predictions
    results['prediction_probs'] = prediction_probs
    logger.info("Predictions generated and attached to the dataset.")

    # Compute SHAP values using our updated SHAP calculator.
    explainer, shap_values = shap_calculator.compute_shap_values(X_preprocessed, debug=config.logging.debug)
    results['shap_values'] = shap_values
    results['explainer'] = explainer
    results['X_preprocessed'] = X_preprocessed
    logger.info("SHAP values computed successfully.")
    logger.debug(f"SHAP values shape: {shap_values.shape}")
    
    # -----------------------------------------------------------
    # IMPORTANT UPDATE:
    # Instead of using X_inversed (17 features) for expected_features,
    # we use X_preprocessed (14 features) to match the SHAP computation.
    # -----------------------------------------------------------
    expected_features = X_preprocessed.columns.tolist()
    logger.debug(f"SHAP features count: {len(X_preprocessed.columns)}")
    logger.debug(f"Expected features count (from X_preprocessed): {len(expected_features)}")
    # Optionally, add an assertion during development:
    # assert set(X_preprocessed.columns) == set(expected_features), "Feature mismatch!"
    
    # Generate and expand feedback.
    X_inversed = generate_feedback_and_expand(
        X_inversed=X_inversed,
        shap_values=shap_values,
        logger=logger,
        feedback_generator=feedback_generator,
        metrics_percentile=metrics_percentile,
        expected_features=expected_features,  # now using the correct features list
        reference_index=X_preprocessed.index
    )
    results['final_dataset'] = X_inversed
    logger.info("Feedback generation and metric threshold application completed.")

    # Generate plots if required.
    if generate_summary_plot:
        shap_summary_path = save_dir / "shap_summary.png"
        try:
            shap_visualizer.plot_summary(shap_values, X_preprocessed, shap_summary_path, debug=config.logging.debug)
            results['shap_summary_plot'] = str(shap_summary_path)
            logger.info(f"SHAP summary plot saved at {shap_summary_path}.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary plot: {e}")

    if generate_dependence_plots:
        try:
            recommendations_dict = feedback_generator.generate_global_recommendations(
                shap_values=shap_values,
                X_original=X_preprocessed,
                top_n=top_n_features,
                use_mad=use_mad,
                debug=config.logging.debug
            )
            results['recommendations'] = recommendations_dict
            shap_dependence_dir = save_dir / "shap_dependence_plots"
            shap_dependence_dir.mkdir(parents=True, exist_ok=True)
            for feature in recommendations_dict.keys():
                dep_path = shap_dependence_dir / f"shap_dependence_{feature}.png"
                shap_visualizer.plot_dependence(shap_values, feature, X_preprocessed, dep_path, interaction_index=None, debug=config.logging.debug)
            logger.info(f"SHAP dependence plots saved at {shap_dependence_dir}.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP dependence plots: {e}")

    if generate_force_plots and force_plot_indices:
        try:
            force_plots_dir = save_dir / "shap_force_plots"
            force_plots_dir.mkdir(parents=True, exist_ok=True)
            for idx in force_plot_indices:
                if idx < 0 or idx >= X_preprocessed.shape[0]:
                    logger.warning(f"Index {idx} is out of bounds. Skipping.")
                    continue
                force_path = force_plots_dir / f"shap_force_plot_{idx}.html"
                shap_visualizer.plot_force(explainer, shap_values, X_preprocessed, idx, force_path, debug=config.logging.debug)
            logger.info(f"SHAP force plots saved at {force_plots_dir}.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP force plots: {e}")

    # Optionally add extra columns from the input DataFrame.
    if columns_to_add:
        try:
            logger.info(f"Adding columns from df_input to final_df: {columns_to_add}")
            for column in columns_to_add:
                if column not in df_input.columns:
                    logger.warning(f"Column '{column}' not found in input DataFrame.")
                    continue
                if len(df_input) != len(X_inversed):
                    logger.error("Length mismatch between df_input and X_inversed.")
                    raise ValueError("Length mismatch between df_input and X_inversed.")
                X_inversed[column] = df_input[column].values
            logger.info(f"Columns added successfully: {columns_to_add}")
        except Exception as e:
            logger.error(f"Failed to add columns to final_df: {e}")
            raise

    # Save the final dataset.
    try:
        final_dataset_path = save_dir / "final_predictions_with_shap.csv"
        logger.info(f"Saving final dataset with SHAP annotations to {final_dataset_path}.")
        X_inversed.to_csv(final_dataset_path, index=True)
        results['final_dataset'] = str(final_dataset_path)
        logger.info("Final dataset and global recommendations saved.")
    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        raise

    results = convert_np_types(results)
    logger.info("Predict+SHAP pipeline completed successfully.")
    return results





if __name__ == "__main__":
    # Main testing block for trying multiple models.
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
        print(f"Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        exit(1)

    data_dir = Path(config.paths.data_dir).resolve()
    raw_data_path = data_dir / config.paths.raw_data
    predictions_output_path = Path(config.paths.predictions_output_dir).resolve() / "shap_results"

    log_dir = Path(config.paths.log_dir).resolve()
    log_file = Path(config.paths.log_file).resolve()

    try:
        logger = setup_logging(config, log_file)
        logger.info("Starting prediction module (unified predict_and_shap).")
        logger.debug(f"Paths: {config.paths}")
    except Exception as e:
        print(f"❌ Failed to set up logging: {e}")
        exit(1)

    try:
        df_predict = load_dataset(raw_data_path)
        print("Columns in input data:", df_predict.columns.tolist())
        logger.info(f"Prediction input data loaded from {raw_data_path}.")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        exit(1)

    # List of model names to test.
    test_model_names = ["XGBoost", "Random Forest", "CatBoost"]

    for model_name in test_model_names:
        print(f"\n--- Running pipeline for model: {model_name} ---")
        # Create a separate output subdirectory for each model.
        model_output_dir = predictions_output_path / model_name.replace(" ", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = predict_and_shap(
                config=config,
                df_input=df_predict,
                save_dir=model_output_dir,
                columns_to_add=['trial_id'],
                generate_summary_plot=True,
                generate_dependence_plots=True,
                generate_force_plots=True,
                force_plot_indices=[0],
                top_n_features=10,
                use_mad=False,
                logger=logger,
                features_file=(Path(config.paths.data_dir) / config.paths.features_metadata_file).resolve(),
                ordinal_file=Path(f'{Path("../../data") / "preprocessor" / "features_info"}/ordinal_categoricals.pkl'),
                nominal_file=Path(f'{Path("../../data") / "preprocessor" / "features_info"}/nominal_categoricals.pkl'),
                numericals_file=Path(f'{Path("../../data") / "preprocessor" / "features_info"}/numericals.pkl'),
                y_variable_file=Path(f'{Path("../../data") / "preprocessor" / "features_info"}/y_variable.pkl'),
                model_save_dir_override=Path(config.paths.model_save_base_dir),
                transformers_dir_override=Path(config.paths.transformers_save_base_dir),
                metrics_percentile=10,
                override_model_name=model_name  # <<-- override to use the current model in the loop, comment out if not needed
            )
            logger.info(f"Unified predict_and_shap function executed successfully for model {model_name}.")
        except Exception as e:
            logger.error(f"Unified predict_and_shap function failed for model {model_name}: {e}")
            continue

        try:
            print(f"\nFinal Predictions with SHAP annotations for model {model_name} (preview):")
            final_df = pd.read_csv(results['final_dataset'], index_col=0)
            print(final_df.head())
            logger.debug(f"Final DataFrame columns for {model_name}: {final_df.columns.tolist()}")
            trial_index = 0
            if trial_index < final_df.shape[0]:
                shap_columns = [col for col in final_df.columns if col.startswith('shap_')]
                logger.debug(f"'shap_' columns for feedback in {model_name}: {shap_columns}")
                if not shap_columns:
                    logger.error("No 'shap_' columns found in the final DataFrame.")
                    print("No feedback columns found in the final DataFrame.")
                else:
                    print(f"\nFeedback for trial at index {trial_index} for model {model_name}:")
                    feedback = final_df.iloc[trial_index][shap_columns].to_dict()
                    for metric, suggestion in feedback.items():
                        print(f"  - {metric}: {suggestion}")
            else:
                print(f"No feedback found for trial at index {trial_index} in model {model_name}.")
        except Exception as e:
            logger.error(f"Failed to display outputs for model {model_name}: {e}")

    print("All tests completed.")
    print("ATTENTION HERE=", config.paths.model_save_base_dir)
    print("Current working directory:", os.getcwd())
