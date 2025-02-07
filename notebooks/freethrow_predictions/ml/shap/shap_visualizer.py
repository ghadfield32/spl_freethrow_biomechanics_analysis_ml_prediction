
import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import os
import numpy as np

from ml.shap.shap_calculator import ShapCalculator
from ml.shap.shap_utils import load_dataset, setup_logging, load_configuration, initialize_logger

class ShapVisualizer:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the ShapVisualizer with an optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def plot_summary(self, shap_values, X_original: pd.DataFrame, save_path: Path, debug: bool = False):
        """
        Generate and save a SHAP summary plot.

        :param shap_values: SHAP values array.
        :param X_original: Original feature DataFrame.
        :param save_path: Path to save the plot.
        :param debug: Enable detailed debug logs.
        """
        self.logger.info("Generating SHAP summary plot...")
        self._log_shap_values(shap_values, X_original, debug)
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_original, show=False)
            plt.tight_layout()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.debug(f"SHAP summary plot saved to {save_path}")
            self.logger.info("SHAP summary plot generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP summary plot: {e}")
            raise

    def plot_dependence(self, shap_values, feature: str, X_original: pd.DataFrame, save_path: Path, interaction_index: Optional[str] = None, debug: bool = False):
        self.logger.info(f"Generating SHAP dependence plot for feature '{feature}'...")
        try:
            # Instead of slicing the SHAP values to a 1D vector,
            # verify that the full shap_values array has the expected number of columns.
            if not (isinstance(shap_values, np.ndarray) and shap_values.ndim == 2):
                msg = f"'shap_values' must be a 2D array, but got shape {np.shape(shap_values)}"
                self.logger.error(msg)
                raise ValueError(msg)
            
            if shap_values.shape[1] != len(X_original.columns):
                msg = f"'shap_values' has {shap_values.shape[1]} columns but 'features' has {len(X_original.columns)} columns!"
                self.logger.error(msg)
                raise ValueError(msg)
            
            # Log the shapes for debugging.
            self.logger.debug(f"Full shap_values shape: {shap_values.shape}")
            self.logger.debug(f"X_original shape: {X_original.shape}")
            
            # Call the dependence plot using the full 2D array and feature name.
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature, shap_values, X_original, interaction_index=interaction_index, show=False)
            plt.tight_layout()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.debug(f"SHAP dependence plot for '{feature}' saved to {save_path}")
            self.logger.info(f"SHAP dependence plot for feature '{feature}' generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP dependence plot for feature '{feature}': {e}")
            raise



    def plot_force(self, shap_explainer, shap_values, X_original: pd.DataFrame, trial_id: str, save_path: Path, debug: bool = False):
        self.logger.info(f"Generating SHAP force plot for trial {trial_id}...")
        try:
            trial_index = X_original.index.get_loc(trial_id) if trial_id in X_original.index else None
            if trial_index is None:
                self.logger.error(f"Trial ID '{trial_id}' not found in X_original index.")
                raise ValueError(f"Trial ID '{trial_id}' not found.")
            
            # Log and inspect the expected value from the explainer.
            base_val = shap_explainer.expected_value
            self.logger.debug(f"explainer.expected_value type: {type(base_val)}")
            if isinstance(base_val, (list, np.ndarray)):
                self.logger.debug(f"explainer.expected_value shape: {np.shape(base_val)}")
                base_val = base_val[0]  # For multi-output, select the first element.
            self.logger.debug(f"Selected base value for force plot: {base_val}")
            
            # Extract and squeeze SHAP values for the trial.
            trial_shap = shap_values[trial_index]
            self.logger.debug(f"Trial SHAP values shape before squeezing: {np.shape(trial_shap)}")
            if trial_shap.ndim > 1:
                trial_shap = np.squeeze(trial_shap)
                self.logger.debug(f"Trial SHAP values shape after squeezing: {np.shape(trial_shap)}")
            
            # Call the new force plot API.
            shap_plot = shap.force_plot(
                base_val,
                trial_shap,
                X_original.iloc[trial_index],
                matplotlib=False
            )
            shap.save_html(str(save_path), shap_plot)
            self.logger.debug(f"SHAP force plot saved to {save_path}")
            self.logger.info(f"SHAP force plot for trial {trial_id} generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP force plot for trial '{trial_id}': {e}")
            raise


    def plot_interaction(self, shap_values, feature: str, X_original: pd.DataFrame, save_path: Path, debug: bool = False):
        """
        Generate and save a SHAP interaction plot for a specific feature.

        :param shap_values: SHAP values array.
        :param feature: Feature name for interaction plot.
        :param X_original: Original feature DataFrame.
        :param save_path: Path to save the plot.
        :param debug: Enable detailed debug logs.
        """
        self.logger.info(f"Generating SHAP interaction plot for feature '{feature}'...")
        try:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature, shap_values, X_original, interaction_index=feature, show=False)
            plt.tight_layout()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.debug(f"SHAP interaction plot for '{feature}' saved to {save_path}")
            self.logger.info(f"SHAP interaction plot for feature '{feature}' generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP interaction plot for feature '{feature}': {e}")
            raise

    def generate_all_plots(self, shap_values, explainer, X_original: pd.DataFrame, save_dir: Path, debug: bool = False):
        """
        Generate and save all relevant SHAP plots.

        :param shap_values: SHAP values array.
        :param explainer: SHAP explainer object.
        :param X_original: Original feature DataFrame.
        :param save_dir: Directory to save all plots.
        :param debug: Enable detailed debug logs.
        """
        self.logger.info("Generating all SHAP plots...")
        try:
            # Ensure save directory exists
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Summary Plot
            summary_plot_path = save_dir / "shap_summary.png"
            self.plot_summary(shap_values, X_original, summary_plot_path, debug=debug)

            # Dependence and Interaction Plots for Top Features
            top_features = self.get_top_features(shap_values, X_original, top_n=5)
            for feature in top_features:
                dependence_plot_path = save_dir / f"shap_dependence_{feature}.png"
                self.plot_dependence(shap_values, feature, X_original, dependence_plot_path, interaction_index=None, debug=debug)

                # Interaction Plot (Optional)
                interaction_plot_path = save_dir / f"shap_interaction_{feature}.png"
                self.plot_interaction(shap_values, feature, X_original, interaction_plot_path, debug=debug)

            # Force Plot for a Specific Trial (e.g., first trial)
            trial_id = X_original.index[0]
            force_plot_path = save_dir / f"shap_force_plot_{trial_id}.html"
            self.plot_force(explainer, shap_values, X_original, trial_id, force_plot_path, debug=debug)

            self.logger.info("All SHAP plots generated and saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to generate all SHAP plots: {e}")
            raise

    def get_top_features(self, shap_values, X_original: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        Identify top N features based on mean absolute SHAP values.

        :param shap_values: SHAP values array.
        :param X_original: Original feature DataFrame.
        :param top_n: Number of top features to identify.
        :return: List of top feature names.
        """
        if isinstance(shap_values, list):
            # Multiclass: average over classes
            shap_values_avg = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values_avg = np.mean(np.abs(shap_values), axis=0)
        
        feature_importance = pd.Series(shap_values_avg, index=X_original.columns)
        top_features = feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
        self.logger.debug(f"Top {top_n} features based on SHAP values: {top_features}")
        return top_features

    def _log_shap_values(self, shap_values, X_original: pd.DataFrame, debug: bool):
        """
        Log details about SHAP values and feature alignment.

        :param shap_values: SHAP values array.
        :param X_original: Original feature DataFrame.
        :param debug: Enable detailed debug logs.
        """
        if debug:
            self.logger.debug(f"Type of shap_values: {type(shap_values)}")
            self.logger.debug(f"Shape of shap_values: {np.shape(shap_values)}")
            if isinstance(shap_values, list):
                self.logger.debug(f"Number of class SHAP arrays: {len(shap_values)}")
                if len(shap_values) > 0 and hasattr(shap_values[0], 'shape'):
                    self.logger.debug(f"Shape of first class SHAP array: {shap_values[0].shape}")
            elif isinstance(shap_values, np.ndarray):
                self.logger.debug(f"Shape of shap_values: {shap_values.shape}")
            if hasattr(shap_values, 'values'):
                self.logger.debug(f"Sample SHAP values:\n{shap_values.values[:2]}")
            if hasattr(shap_values, 'feature_names'):
                self.logger.debug(f"SHAP feature names: {shap_values.feature_names}")

            self.logger.debug(f"Type of X_original: {type(X_original)}")
            self.logger.debug(f"Shape of X_original: {X_original.shape}")
            self.logger.debug(f"Columns in X_original: {X_original.columns.tolist()}")

            # Verify column alignment
            shap_feature_names = self._get_shap_feature_names(shap_values, X_original)
            if list(shap_feature_names) != list(X_original.columns):
                self.logger.error("Column mismatch between SHAP values and X_original.")
                self.logger.error(f"SHAP feature names ({len(shap_feature_names)}): {shap_feature_names}")
                self.logger.error(f"X_original columns ({len(X_original.columns)}): {X_original.columns.tolist()}")
                raise ValueError("Column mismatch between SHAP values and X_original.")
            else:
                self.logger.debug("Column alignment verified between SHAP values and X_original.")

    def _get_shap_feature_names(self, shap_values, X_original: pd.DataFrame):
        """
        Extract feature names from shap_values.

        :param shap_values: SHAP values array.
        :param X_original: Original feature DataFrame.
        :return: List of feature names.
        """
        if isinstance(shap_values, shap.Explainer):
            return shap_values.feature_names
        elif hasattr(shap_values, 'feature_names'):
            return shap_values.feature_names
        else:
            return X_original.columns.tolist()  # Fallback


if __name__ == "__main__":
    # Test code to verify the ShapVisualizer class
    print("Testing SHAPvisualizer module...")

    from ml.train_utils.train_utils import load_model
    from datapreprocessor import DataPreprocessor
    from ml.predict.predict import predict_and_attach_predict_probs
    from ml.feature_selection.feature_importance_calculator import manage_features

    # from ml.shap.shap_utils import (
    #     load_dataset,
    #     setup_logging, load_configuration, initialize_logger
    # )


    # **Load Configuration and Initialize Logger**
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config = load_configuration(config_path)
        print(f"✅ Configuration loaded successfully from {config_path}.")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        exit(1)

    log_file = Path(config.paths.log_file).resolve()
    try:
        logger = initialize_logger(config, log_file)
    except Exception as e:
        print(f"❌ Failed to set up logging: {e}")
        exit(1)

    # **Load Dataset**
    raw_data_path = Path(config.paths.data_dir).resolve() / config.paths.raw_data
    try:
        df = load_dataset(raw_data_path)
        print(f"✅ Dataset loaded successfully from {raw_data_path}.")
        print(f"📊 Dataset Columns: {df.columns.tolist()}")
        logger.info(f"Dataset loaded with shape: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        exit(1)

    # **Load Model**
    try:
        model = load_model('CatBoost', Path(config.paths.model_save_base_dir).resolve())
        print("✅ Model loaded successfully.")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        exit(1)

    # **Initialize ShapCalculator**
    shap_calculator = ShapCalculator(model=model, logger=logger)
    print("✅ ShapCalculator initialized successfully.")
    logger.info("ShapCalculator initialized successfully.")

    # **Load Feature Lists (via manage_features)**
    base_dir = Path("../../data") / "preprocessor" / "features_info"
    features_file = (Path(config.paths.data_dir) / config.paths.features_metadata_file).resolve()
    ordinal_file = Path(f'{base_dir}/ordinal_categoricals.pkl')
    nominal_file = Path(f'{base_dir}/nominal_categoricals.pkl')
    numericals_file = Path(f'{base_dir}/numericals.pkl')
    y_variable_file = Path(f'{base_dir}/y_variable.pkl')
    model_save_dir_override = Path(config.paths.model_save_base_dir)
    transformers_dir_override = Path(config.paths.transformers_save_base_dir)
    best_model_name = "CatBoost"
    save_dir = Path(config.paths.predictions_output_dir).resolve() / "shap_results"
    model_path = model_save_dir_override / best_model_name.replace(' ', '_') / "trained_model.pkl"
    results = {}
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

    # **Initialize DataPreprocessor**
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
        transformers_dir=transformers_dir_override
    )

    # **Preprocess Data**
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df)
        if logger:
            logger.info("Preprocessing completed successfully in predict mode.")
    except Exception as e:
        if logger:
            logger.error(f"Preprocessing failed: {e}")
        raise

    # **Validate Preprocessed Data**
    try:
        non_numeric_features = X_preprocessed.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_features:
            logger.error(f"Non-numeric features detected in preprocessed data: {non_numeric_features}")
            raise ValueError(f"Preprocessed data contains non-numeric features: {non_numeric_features}")
        else:
            logger.debug("All features in X_preprocessed are numeric.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

    duplicates = X_inversed.index.duplicated()
    if duplicates.any():
        print("Duplicate trial IDs found:", X_inversed.index[duplicates].tolist())
    else:
        print("Trial IDs are unique.")

    try:
        model = load_model(best_model_name, model_save_dir_override)
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
    shap_visualizer = ShapVisualizer(logger=logger)
    generate_summary_plot = True
    if generate_summary_plot:
        shap_summary_path = save_dir / "shap_summary.png"
        try:
            shap_visualizer.plot_summary(shap_values, X_preprocessed, shap_summary_path, debug=config.logging.debug)
            results['shap_summary_plot'] = str(shap_summary_path)
            logger.info(f"SHAP summary plot saved at {shap_summary_path}.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary plot: {e}")
