
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


def generate_individual_feedback(trial: pd.Series, shap_values_trial: np.ndarray,
                                 feature_metadata: dict = None, logger: logging.Logger = None) -> dict:
    """
    Generate individual feedback for a single trial and break it out into several pieces for each feature.
    
    For each feature, three keys will be generated:
      1. shap_{sign}_direction_{feature}: A recommendation such as "decrease" or "maintain or increase"
      2. shap_importance_{feature}: The absolute SHAP value (rounded) for that feature in this trial
      3. shap_{sign}_unit_change_{feature}: The computed unit–change (with unit) for that feature.
    
    :param trial: A pandas Series containing the trial's feature values.
    :param shap_values_trial: A numpy array with the SHAP values for the trial.
    :param feature_metadata: A dictionary with metadata per feature (e.g. units).
    :param logger: Optional logger for debugging.
    :return: A dictionary containing the feedback.
    """
    feedback = {}
    
    # Loop over each feature and its associated SHAP value.
    for feature, shap_val in zip(trial.index.tolist(), shap_values_trial):
        # Choose the sign and suggestion based on the SHAP value.
        if shap_val > 0:
            sign = "positive"
            suggestion = "increase"
        elif shap_val < 0:
            sign = "negative"
            suggestion = "decrease"
        else:
            # If SHAP is zero, mark it as no impact.
            feedback[f"shap_{feature}_impact"] = "no impact"
            continue

        # Compute the adjustment amount as (e.g.) 10% of the current value's absolute.
        current_value = trial[feature]
        adjustment_factor = 0.1
        unit_change_value = adjustment_factor * abs(current_value)
        
        # Retrieve unit from feature_metadata, if available.
        unit = ""
        if feature_metadata and feature in feature_metadata:
            unit = feature_metadata[feature].get('unit', '')
        # Format the unit change as a string.
        unit_change_str = f"{unit_change_value:.2f} {unit}".strip()

        # The importance for the feature in this trial is simply the absolute value of the SHAP value.
        importance = abs(shap_val)
        
        # Create key names.
        key_direction = f"shap_direction_{feature}"
        key_importance = f"shap_importance_{feature}"
        key_unit_change = f"shap_unit_change_{feature}"
        
        # Set values.
        feedback[key_direction] = suggestion
        feedback[key_importance] = round(importance, 4)
        feedback[key_unit_change] = unit_change_str

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"For feature '{feature}': {key_direction}='{suggestion}', "
                         f"{key_importance}={round(importance, 4)}, {key_unit_change}='{unit_change_str}'")
    return feedback



def expand_specific_feedback(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Expand the 'specific_feedback' column from dictionaries to separate columns.

    :param df: Original DataFrame containing the 'specific_feedback' column.
    :param logger: Optional logger for debugging.
    :return: Expanded DataFrame with separate feedback columns.
    """
    if 'specific_feedback' not in df.columns:
        logger.error("'specific_feedback' column not found in DataFrame.")
        raise KeyError("'specific_feedback' column not found.")
    
    logger.info("Expanding 'specific_feedback' into separate columns.")
    try:
        feedback_df = df['specific_feedback'].apply(pd.Series)
        logger.debug(f"Feedback DataFrame shape after expansion: {feedback_df.shape}")
        
        # Optional: Handle missing values
        feedback_df.fillna('No feedback available', inplace=True)
        
        # Merge with original DataFrame
        df_expanded = pd.concat([df.drop(columns=['specific_feedback']), feedback_df], axis=1)
        logger.info("'specific_feedback' expanded successfully.")
        return df_expanded
    except Exception as e:
        logger.error(f"Failed to expand 'specific_feedback': {e}")
        raise




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
        # Retrieve the correct SHAP row and trial features
        if isinstance(trial_index, int):
            shap_value = shap_values[trial_index]
            trial_features = X_original.iloc[[trial_index]]  # Pass as DataFrame (1-row)
            if logger and debug:
                logger.debug(f"Using integer index {trial_index} for force plot.")
        else:
            shap_value = get_shap_row(shap_values, X_original, trial_index, logger=logger)
            if shap_value is None:
                if logger:
                    logger.warning(f"SHAP row not found for trial '{trial_index}'.")
                return
            trial_features = X_original.loc[[trial_index]]  # keep as DataFrame

        # Determine the expected value and ensure it is a scalar (or select the first element if iterable)
        if hasattr(shap_explainer.expected_value, '__iter__'):
            base_value = shap_explainer.expected_value[0]
            if logger and debug:
                logger.debug(f"Expected value is iterable; using {base_value}.")
        else:
            base_value = shap_explainer.expected_value
            if logger and debug:
                logger.debug(f"Expected value is scalar: {base_value}")

        # **Key change:** Use matplotlib=False so that the returned object is an interactive Visualizer
        shap_plot = shap.force_plot(
            base_value, 
            shap_value, 
            trial_features,
            matplotlib=False  # Ensures an interactive (HTML/JS) plot is returned
        )
        
        # Save the interactive plot as an HTML file.
        # (It’s a best practice to convert save_path to string)
        shap.save_html(str(save_path), shap_plot)
        if logger and debug:
            logger.debug(f"Interactive SHAP force plot saved to {save_path}.")
        if logger:
            logger.info(f"✅ Interactive SHAP force plot generated for trial {trial_index}.")
    
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to generate SHAP force plot for trial {trial_index}: {e}")
        raise


# Below is an updated version of shap_utils.py that includes additional functionality.
# This functionality helps compute a percentile-based error limit (unit_change_error_limit)
# for each feature (metric), then uses that limit alongside the computed shap_unit_change_{metric}
# and shap_direction_{metric} to determine a trial-based feedback label such as "early", "good",
# or "late".
#
# The new function introduced at the end is `compute_feedback_with_thresholds`. Inside it, we:
# 1) Collect a data distribution for each metric from the entire dataset (or a relevant subset).
# 2) Compute a chosen percentile (like the 90th percentile) for each metric to determine an error limit.
# 3) Retrieve shap_unit_change_{metric} and shap_direction_{metric} from the feedback.
#    - If "increase" -> the "goal" is current_value + shap_unit_change.
#    - If "decrease" -> the "goal" is current_value - shap_unit_change.
# 4) Compare the difference (or ratio) between the goal and the actual distribution of that metric to see
#    if the difference is within or beyond the error limit.
# 5) Mark trial-based feedback as "early", "good", or "late" accordingly.
#
# NOTE: This is sample logic and can be adapted to match your exact domain definitions.
#       Additional logging or debugging can be added as needed.

import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Other functions omitted for brevity...

###########################################
# NEW FUNCTION: compute_feedback_with_thresholds
###########################################

def compute_feedback_with_thresholds(
    df: pd.DataFrame,
    features: List[str],
    percentile: float = 90,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    This function demonstrates how to add percentile-based thresholds for each metric.

    Steps:
    1) For each metric in 'features', calculate the chosen percentile (e.g. 90th) across all trials.
       This is our 'unit_change_error_limit' for that metric.
    2) We look for the shap_direction_{metric} and shap_unit_change_{metric} columns in df.
       If they say "increase" or "decrease", we can compute the 'goal' metric as:
         - If direction is 'increase':  goal = actual_value + shap_unit_change (numerical interpretation)
         - If direction is 'decrease':  goal = actual_value - shap_unit_change
    3) Compare the difference between this 'goal' and the actual_value to see if it is within
       the 'unit_change_error_limit'. Then assign feedback: 'early', 'good', or 'late'.
       (The user can define the exact rules for these categories. We'll show an example.)

    Returns:
       The same DataFrame 'df', but with new columns:
         - {feature}_threshold
         - shap_feedback_{feature}
       which store the threshold used and the final feedback label.
    """
    df_out = df.copy()

    # 1) Compute the percentile for each feature across the dataset.
    thresholds = {}
    for metric in features:
        if metric not in df_out.columns:
            if logger:
                logger.warning(f"Metric '{metric}' not found in df columns, skipping.")
            continue
        threshold_value = np.percentile(df_out[metric].dropna(), percentile)
        thresholds[metric] = threshold_value
        if logger:
            logger.info(f"{percentile}th percentile for '{metric}' is {threshold_value:.3f}.")

    # 2) For each metric, retrieve direction & unit_change columns if they exist.
    for metric in features:
        dir_col = f"shap_direction_{metric}"
        uc_col = f"shap_unit_change_{metric}"

        threshold_col = f"{metric}_threshold"
        feedback_col = f"shap_feedback_{metric}"

        # We'll store the threshold for reference.
        if metric in thresholds:
            df_out[threshold_col] = thresholds[metric]
        else:
            df_out[threshold_col] = np.nan

        # If direction or unit change columns are absent, skip.
        if dir_col not in df_out.columns or uc_col not in df_out.columns:
            if logger:
                logger.debug(f"Either {dir_col} or {uc_col} not found in df columns. Skipping feedback.")
            df_out[feedback_col] = "No feedback"
            continue

        # We'll parse the numeric portion from shap_unit_change_{metric} because that might be something like '0.45 meters'.
        # Let's define a helper to parse the numeric part.
        def parse_value_with_unit(value_str: Any) -> float:
            if isinstance(value_str, (int, float)):
                return float(value_str)
            try:
                # example: "0.45 meters", split by space, parse first item
                parts = str(value_str).split()
                return float(parts[0])
            except:
                return 0.0

        # Now define an inline function to compute feedback row by row.
        def compute_row_feedback(row):
            # If we have no actual metric col in the row, skip.
            if pd.isnull(row.get(metric, np.nan)):
                return "No actual metric"

            actual_val = row[metric]
            direction = row[dir_col]
            # shap_unit_change_{metric} might be numeric or str with unit, parse.
            shap_delta = parse_value_with_unit(row[uc_col])

            # compute "goal"
            if direction == "increase":
                goal_val = actual_val + shap_delta
            elif direction == "decrease":
                goal_val = actual_val - shap_delta
            else:
                # e.g. 'no feedback available' or something else.
                return "No direction"

            # difference from goal
            diff = abs(goal_val - actual_val)

            # compare with threshold
            limit = thresholds.get(metric, np.nan)
            if pd.isnull(limit) or limit == 0:
                # fallback if threshold not found.
                return "No threshold"

            # Define rules based on percentiles:
            if diff <= 0.05 * limit:
                return "good"
            elif diff > 0.05 * limit and goal_val > actual_val:
                return "early"
            elif diff > 0.05 * limit and goal_val < actual_val:
                return "late"
            else:
                return "No feedback"

        df_out[feedback_col] = df_out.apply(compute_row_feedback, axis=1)

    return df_out
