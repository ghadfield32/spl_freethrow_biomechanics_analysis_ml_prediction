
import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_shap_values(model, X, debug: bool = False, logger: logging.Logger = None):
    if logger:
        logger.info("Initializing SHAP explainer...")

    try:
        explainer = shap.Explainer(model, X)
        if logger and debug:
            logger.debug(f"SHAP Explainer initialized: {type(explainer)}")
            logger.debug(f"Explainer details: {explainer}")

        shap_values = explainer(X)
        if logger and debug:
            logger.debug(f"SHAP values computed: {type(shap_values)}")
            logger.debug(f"Shape of shap_values: {shap_values.shape}")
            if hasattr(shap_values, 'values'):
                logger.debug(f"Sample SHAP values:\n{shap_values.values[:2]}")
            if hasattr(shap_values, 'feature_names'):
                logger.debug(f"SHAP feature names: {shap_values.feature_names}")

        # Determine number of classes
        n_classes = len(model.classes_)
        logger.debug(f"Number of classes in the model: {n_classes}")

        if shap_values.values.ndim == 3:
            if n_classes > 1:
                # For multi-class classification, select SHAP values for the positive class
                shap_values_class = shap_values.values[:, :, 1]
                logger.debug(f"Extracted SHAP values for class 1: Shape {shap_values_class.shape}")
            else:
                # For single-class models, retain SHAP values as is
                shap_values_class = shap_values.values[:, :, 0]
                logger.debug(f"Extracted SHAP values for single class: Shape {shap_values_class.shape}")
        elif shap_values.values.ndim == 2:
            if n_classes > 1:
                # For binary classification, SHAP returns 2D array for the positive class
                shap_values_class = shap_values.values
                logger.debug(f"Extracted SHAP values for positive class: Shape {shap_values_class.shape}")
            else:
                shap_values_class = shap_values.values
                logger.debug(f"Extracted SHAP values for single class: Shape {shap_values_class.shape}")
        else:
            logger.error(f"Unexpected SHAP values dimensions: {shap_values.values.ndim}")
            raise ValueError("Unexpected SHAP values dimensions.")

        return explainer, shap_values_class
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute SHAP values: {e}")
        raise


def plot_shap_summary(shap_values, X_original: pd.DataFrame, save_path: str, debug: bool = False, 
                     logger: logging.Logger = None):
    """
    Generate and save a SHAP summary plot.

    :param shap_values: SHAP values computed for the dataset.
    :param X_original: Original (preprocessed) feature DataFrame.
    :param save_path: Full file path to save the plot.
    :param debug: If True, enable detailed debug logs.
    :param logger: Logger instance for logging.
    """
    if logger:
        logger.info("Generating SHAP summary plot...")
        logger.debug(f"Type of shap_values: {type(shap_values)}")
        logger.debug(f"Shape of shap_values: {shap_values.shape}")
        
        # Check if shap_values is a shap.Explanation object
        if isinstance(shap_values, shap.Explanation):
            logger.debug(f"SHAP feature names: {shap_values.feature_names}")
        else:
            logger.debug("shap_values is not a shap.Explanation object. Attempting to extract feature names.")
            if hasattr(shap_values, 'feature_names'):
                logger.debug(f"shap_values feature names: {shap_values.feature_names}")
            else:
                logger.debug("Cannot extract feature names from shap_values.")
        
        logger.debug(f"Type of X_original: {type(X_original)}")
        logger.debug(f"Shape of X_original: {X_original.shape}")
        logger.debug(f"Columns in X_original: {X_original.columns.tolist()}")
        
        # Verify column alignment
        if isinstance(shap_values, shap.Explanation):
            shap_feature_names = shap_values.feature_names
        elif hasattr(shap_values, 'feature_names'):
            shap_feature_names = shap_values.feature_names
        else:
            shap_feature_names = X_original.columns.tolist()  # Fallback
        
        if list(shap_feature_names) != list(X_original.columns):
            logger.error("Column mismatch between SHAP values and X_original.")
            logger.error(f"SHAP feature names ({len(shap_feature_names)}): {shap_feature_names}")
            logger.error(f"X_original columns ({len(X_original.columns)}): {X_original.columns.tolist()}")
            raise ValueError("Column mismatch between SHAP values and X_original.")
        else:
            logger.debug("Column alignment verified between SHAP values and X_original.")
    
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_original, show=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        if logger and debug:
            logger.debug(f"SHAP summary plot saved to {save_path}")
        if logger:
            logger.info("SHAP summary plot generated successfully.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate SHAP summary plot: {e}")
        raise




def plot_shap_dependence(shap_values, feature: str, X_original: pd.DataFrame, save_path: str, debug: bool = False, 
                         logger: logging.Logger = None):
    """
    Generate and save a SHAP dependence plot for a specific feature.

    :param shap_values: SHAP values computed for the dataset.
    :param feature: Feature name to generate the dependence plot for.
    :param X_original: Original (untransformed) feature DataFrame.
    :param save_path: Full file path to save the plot.
    :param debug: If True, enable detailed debug logs.
    :param logger: Logger instance for logging.
    """
    if logger:
        logger.info(f"Generating SHAP dependence plot for feature '{feature}'...")
    try:
        plt.figure(figsize=(8, 6))
        # shap.dependence_plot(feature, shap_values.values, X_original, show=False)
        shap.dependence_plot(feature, shap_values, X_original, show=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        if logger and debug:
            logger.debug(f"SHAP dependence plot for '{feature}' saved to {save_path}")
        if logger:
            logger.info(f"SHAP dependence plot for feature '{feature}' generated successfully.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate SHAP dependence plot for feature '{feature}': {e}")
        raise


def generate_global_recommendations(shap_values, X_original: pd.DataFrame, top_n: int = 5, debug: bool = False, 
                                    use_mad: bool = False, logger: logging.Logger = None) -> dict:
    """
    Generate global recommendations based on SHAP values and feature distributions.

    :param shap_values: SHAP values computed for the dataset.
    :param X_original: Original (untransformed) feature DataFrame.
    :param top_n: Number of top features to generate recommendations for.
    :param debug: If True, enable detailed debug logs.
    :param use_mad: If True, use Median Absolute Deviation for range definition.
    :param logger: Logger instance for logging.
    :return: recommendations: Dictionary mapping features to recommended value ranges, importance, and direction.
    """
    if logger:
        logger.info("Generating feature importance based on SHAP values...")
    try:
        # shap_df = pd.DataFrame(shap_values.values, columns=X_original.columns)
        shap_df = pd.DataFrame(shap_values, columns=X_original.columns)

        # Calculate mean absolute SHAP values for importance
        feature_importance = pd.DataFrame({
            'feature': X_original.columns,
            'importance': np.abs(shap_df).mean(axis=0),
            'mean_shap': shap_df.mean(axis=0)
        }).sort_values(by='importance', ascending=False)
        
        if logger and debug:
            logger.debug(f"Feature importance (top {top_n}):\n{feature_importance.head(top_n)}")
        
        top_features = feature_importance.head(top_n)['feature'].tolist()
        recommendations = {}
        
        for feature in top_features:
            feature_values = X_original[feature]
            
            if use_mad:
                # Use Median and MAD for robust statistics
                median = feature_values.median()
                mad = feature_values.mad()
                lower_bound = median - 1.5 * mad
                upper_bound = median + 1.5 * mad
                range_str = f"{lower_bound:.1f}–{upper_bound:.1f}"
            else:
                # Default to Interquartile Range (IQR)
                lower_bound = feature_values.quantile(0.25)
                upper_bound = feature_values.quantile(0.75)
                range_str = f"{lower_bound:.1f}–{upper_bound:.1f}"
            
            # Determine direction based on mean SHAP value
            mean_shap = feature_importance.loc[feature_importance['feature'] == feature, 'mean_shap'].values[0]
            direction = 'positive' if mean_shap > 0 else 'negative'
            
            recommendations[feature] = {
                'range': range_str,
                'importance': round(feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0], 4),  # Rounded for readability
                'direction': direction
            }
            if logger and debug:
                logger.debug(f"Recommendation for {feature}: Range={range_str}, Importance={feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0]}, Direction={direction}")
        
        if logger and debug:
            logger.debug(f"Final Recommendations with Importance and Direction: {recommendations}")
        return recommendations
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate global recommendations: {e}")
        raise


def generate_individual_feedback(trial: pd.Series, shap_values_trial: np.ndarray, feature_metadata: dict = None, 
                                 logger: logging.Logger = None) -> dict:
    """
    Generate specific feedback for a single trial based on its SHAP values and feature metadata.

    Args:
        trial (pd.Series): A single trial's data.
        shap_values_trial (np.ndarray): SHAP values for the trial.
        feature_metadata (dict, optional): Additional metadata for features (e.g., units).
        logger (logging.Logger, optional): Logger instance for logging.

    Returns:
        dict: Feedback messages for each feature.
    """
    feedback = {}
    feature_names = trial.index.tolist()

    for feature, shap_value in zip(feature_names, shap_values_trial):
        if shap_value > 0:
            adjustment = "maintain or increase"
            direction = "positively"
        elif shap_value < 0:
            adjustment = "decrease"
            direction = "positively"
        else:
            feedback[feature] = f"{feature.replace('_', ' ').capitalize()} has no impact on the prediction."
            continue

        # Map SHAP values to meaningful adjustment magnitudes
        # Example: 10% of the current feature value
        current_value = trial[feature]
        adjustment_factor = 0.1
        adjustment_amount = adjustment_factor * abs(current_value)

        # Incorporate feature metadata if available
        if feature_metadata and feature in feature_metadata:
            unit = feature_metadata[feature].get('unit', '')
            adjustment_str = f"{adjustment_amount:.2f} {unit}" if unit else f"{adjustment_amount:.2f}"
        else:
            adjustment_str = f"{adjustment_amount:.2f}"

        # Construct feedback message
        feedback_message = (
            f"Consider to {adjustment} '{feature.replace('_', ' ')}' by approximately {adjustment_str} "
            f"to {direction} influence the result."
        )
        feedback[feature] = feedback_message

    return feedback


def compute_individual_shap_values(explainer, X_transformed: pd.DataFrame, trial_index: int, 
                                   logger: logging.Logger = None):
    """
    Compute SHAP values for a single trial.

    :param explainer: SHAP explainer object.
    :param X_transformed: Transformed features used for prediction.
    :param trial_index: Index of the trial.
    :param logger: Logger instance.
    :return: shap_values for the trial.
    """
    if logger:
        logger.info(f"Computing SHAP values for trial at index {trial_index}...")
    try:
        trial = X_transformed.iloc[[trial_index]]
        shap_values = explainer(trial)
        if logger:
            logger.debug(f"SHAP values for trial {trial_index} computed successfully.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to compute SHAP values for trial {trial_index}: {e}")
        raise
    return shap_values


def plot_individual_shap_force(shap_explainer, shap_values, X_original: pd.DataFrame, trial_index: int, 
                               save_path: str, logger: logging.Logger = None):
    """
    Generate and save a SHAP force plot for a specific trial.

    :param shap_explainer: SHAP explainer object.
    :param shap_values: SHAP values for the trial.
    :param X_original: Original feature DataFrame.
    :param trial_index: Index of the trial.
    :param save_path: Full file path to save the force plot.
    :param logger: Logger instance.
    """
    if logger:
        logger.info(f"Generating SHAP force plot for trial {trial_index}...")
    try:
        shap_plot = shap.force_plot(
            shap_explainer.expected_value, 
            shap_values.values[0], 
            X_original.iloc[trial_index],
            matplotlib=False
        )
        shap.save_html(save_path, shap_plot)
        if logger:
            logger.debug(f"SHAP force plot saved to {save_path}")
            logger.info(f"SHAP force plot for trial {trial_index} generated successfully.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate SHAP force plot for trial {trial_index}: {e}")
        raise


def extract_force_plot_values(shap_values, trial_index: int, logger: logging.Logger = None) -> dict:
    """
    Extract SHAP values and feature contributions for a specific trial.

    Args:
        shap_values (shap.Explanation): SHAP values object.
        trial_index (int): Index of the trial.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: Dictionary of feature contributions.
    """
    try:
        shap_values_instance = shap_values.values[trial_index]
        features_instance = shap_values.data[trial_index]
        feature_contributions = dict(zip(shap_values.feature_names, shap_values_instance))
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"SHAP values for trial {trial_index}: {feature_contributions}")
        return feature_contributions
    except Exception as e:
        if logger:
            logger.error(f"Error extracting SHAP values for trial {trial_index}: {e}")
        raise


def save_shap_values(shap_values, save_path: str, logger: logging.Logger = None):
    """
    Save SHAP values to a file using pickle.

    :param shap_values: SHAP values object to save.
    :param save_path: File path to save the SHAP values.
    :param logger: Logger instance.
    """
    if logger:
        logger.info(f"Saving SHAP values to {save_path}...")
    try:
        with open(save_path, "wb") as f:
            pickle.dump(shap_values, f)
        if logger:
            logger.info(f"SHAP values saved successfully to {save_path}.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save SHAP values: {e}")
        raise


def load_shap_values(load_path: str, logger: logging.Logger = None):
    """
    Load SHAP values from a pickle file.

    :param load_path: File path to load the SHAP values from.
    :param logger: Logger instance.
    :return: Loaded SHAP values object.
    """
    if logger:
        logger.info(f"Loading SHAP values from {load_path}...")
    try:
        with open(load_path, "rb") as f:
            shap_values = pickle.load(f)
        if logger:
            logger.info(f"SHAP values loaded successfully from {load_path}.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to load SHAP values: {e}")
        raise
    return shap_values

def get_shap_row(shap_values, df: pd.DataFrame, trial_id: Any, logger: Optional[logging.Logger] = None):
    """
    Retrieve the SHAP values by converting a string-based trial ID
    to its integer position (row index) in the DataFrame.

    :param shap_values: The array-like SHAP values (often shap.Explanation or np.ndarray).
    :param df: A pandas DataFrame indexed by trial IDs.
    :param trial_id: The ID (string or other) we want to map to a row position.
    :param logger: Optional logger for debug or warning messages.
    :return: The 1D array of SHAP values for the specified row, or None if not found.
    """
    if trial_id not in df.index:
        if logger:
            logger.warning(f"Trial ID '{trial_id}' not found in df.index: {df.index.tolist()}")
        return None

    # Convert the string-based index (e.g., "T0123") to its integer position
    pos = df.index.get_loc(trial_id)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Mapped trial ID '{trial_id}' to position {pos} in the DataFrame.")

    return shap_values[pos]


def plot_shap_force(shap_explainer, shap_values, X_original, trial_index, 
                    save_path: Path, debug: bool = False, logger: Optional[logging.Logger] = None):
    try:
        # Retrieve the SHAP values row for the specified trial.
        if isinstance(trial_index, int):
            shap_value = shap_values[trial_index]
            trial_features = X_original.iloc[trial_index]
            if logger and debug:
                logger.debug(f"Index {trial_index} is integer. Using it directly.")
        else:
            if logger and debug:
                logger.debug(f"Index {trial_index} is string; attempting to map it.")
            shap_value = get_shap_row(shap_values, X_original, trial_index, logger=logger)
            if shap_value is None:
                if logger:
                    logger.warning(f"Cannot generate force plot. SHAP row not found for '{trial_index}'.")
                return
            trial_features = X_original.loc[trial_index]

        # Determine the base value.
        if hasattr(shap_explainer.expected_value, '__iter__'):
            base_value = shap_explainer.expected_value[0]
            if logger and debug:
                logger.debug(f"Expected value is iterable; using first element: {base_value}")
        else:
            base_value = shap_explainer.expected_value
            if logger and debug:
                logger.debug(f"Expected value is scalar: {base_value}")

        # Generate the SHAP force plot.
        shap_plot = shap.force_plot(
            base_value,  # Use scalar base_value
            shap_value, 
            trial_features,
            matplotlib=False
        )
        
        # Convert save_path to a string before saving.
        shap.save_html(str(save_path), shap_plot)
        if logger and debug:
            logger.debug(f"SHAP force plot saved to {save_path}.")
        if logger:
            logger.info(f"✅ SHAP force plot generated for trial {trial_index}.")
    
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to generate SHAP force plot for trial {trial_index}: {e}")
        raise






