import pandas as pd
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List
from ml.config.config_models import AppConfig
from ml.config.config_loader import load_config

def compute_original_metric_error(df: pd.DataFrame, percentile: float, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    df_out = df.copy()
    logger.info("[Step C] Computing original metric error with corrected feature extraction.")

    shap_unit_change_cols = [c for c in df_out.columns if c.startswith("shap_") and c.endswith("_unit_change")]
    logger.debug(f"[Debug] Found shap_ columns with '_unit_change': {shap_unit_change_cols}")

    for col in shap_unit_change_cols:
        feature_name = col[len("shap_"): col.rfind("_unit_change")]
        goal_col = f"shap_{feature_name}_goal"
        min_col  = f"shap_{feature_name}_min"
        max_col  = f"shap_{feature_name}_max"
        class_col= f"shap_{feature_name}_classification"

        if goal_col not in df_out.columns:
            logger.debug(f"[Debug] {goal_col} missing => skipping min/max/classification for '{feature_name}'.")
            df_out[min_col] = np.nan
            df_out[max_col] = np.nan
            df_out[class_col] = "No data"
            continue

        df_out[goal_col] = pd.to_numeric(df_out[goal_col], errors='coerce')
        df_out[min_col] = pd.to_numeric(df_out[min_col], errors='coerce')
        df_out[max_col] = pd.to_numeric(df_out[max_col], errors='coerce')
        df_out[feature_name] = pd.to_numeric(df_out[feature_name], errors='coerce')

        if df_out[col].notna().any():
            tolerance = np.percentile(df_out[col].dropna(), percentile)
        else:
            tolerance = 0.0

        logger.debug(f"[Debug] For feature '{feature_name}': tolerance={tolerance:.3f}")
        df_out[min_col] = df_out[goal_col] - tolerance
        df_out[max_col] = df_out[goal_col] + tolerance

        def classify(row):
            try:
                current_val = row.get(feature_name, None)
                min_val = row.get(min_col, None)
                max_val = row.get(max_col, None)
            except Exception as e:
                logger.debug(f"Row {row.name}, feature {feature_name} => {row[feature_name]}")
                raise e

            try:
                val  = float(current_val) if pd.notnull(current_val) else np.nan
                vmin = float(min_val) if pd.notnull(min_val) else np.nan
                vmax = float(max_val) if pd.notnull(max_val) else np.nan
            except (ValueError, TypeError) as e:
                logger.debug(f"[Debug] Classification error for {feature_name}: {e}")
                return "No data"

            if pd.isnull(val) or pd.isnull(vmin) or pd.isnull(vmax):
                return "No data"
            if val < vmin:
                return "Early"
            elif val > vmax:
                return "Late"
            else:
                return "Good"

        df_out[class_col] = df_out.apply(classify, axis=1)
        if not pd.api.types.is_numeric_dtype(df_out[min_col]) or not pd.api.types.is_numeric_dtype(df_out[max_col]):
            logger.error(f"Min or Max columns for feature '{feature_name}' contain non-numeric data after conversion.")
            df_out[class_col] = "No data"

    return df_out

def generate_feedback_and_expand(
    X_inversed: pd.DataFrame,
    shap_values: np.ndarray,
    logger: logging.Logger,
    feedback_generator: Any,
    metrics_percentile: float = 10.0,
    expected_features: Optional[List[str]] = None,
    reference_index: Optional[pd.Index] = None
):
    logger.info("[Step A] Generating feedback for each trial with detailed debug information.")
    logger.debug(f"X_inversed.shape = {X_inversed.shape}, shap_values.shape = {shap_values.shape}")
    if reference_index is not None:
        logger.debug("Reindexing X_inversed to match the reference index.")
        X_inversed = X_inversed.reindex(reference_index)
        logger.debug(f"New X_inversed index: {X_inversed.index.tolist()}")

    if X_inversed.shape[0] != shap_values.shape[0]:
        logger.error("Mismatch between number of shap_values and number of trials.")
        raise ValueError("Mismatch between shap_values and trials.")

    feedback_list = []
    for pos in range(X_inversed.shape[0]):
        try:
            logger.debug(f"[feedback-loop] Processing row at position {pos}")
            shap_values_trial = shap_values[pos]
            logger.debug(f"[feedback-loop] Retrieved shap_values_trial for pos={pos}: {shap_values_trial}")
            trial_features = X_inversed.iloc[pos]
            logger.debug(f"[feedback-loop] Trial features (type {type(trial_features)}): {trial_features}")
            feedback = feedback_generator.generate_individual_feedback(
                trial=trial_features,
                shap_values_trial=shap_values_trial,
                percentile=metrics_percentile,
                expected_features=expected_features
            )
            logger.debug(f"[feedback-loop] Generated feedback for pos={pos}: {feedback}")
            feedback_list.append(feedback)
        except Exception as e:
            logger.warning(f"Error generating feedback for row at position {pos}: {e}")
            feedback_list.append({})
            continue

    feedback_df = pd.DataFrame(feedback_list)
    logger.debug(f"Feedback DataFrame shape: {feedback_df.shape}")
    # Convert columns that hold non-numeric feedback to object type to avoid future dtype warnings.
    for col in feedback_df.columns:
        if feedback_df[col].dtype.kind in 'if':
            feedback_df[col] = feedback_df[col].astype(object)

    existing_shap_cols = {col for col in X_inversed.columns if col.startswith("shap_")}
    new_shap_cols = {col for col in feedback_df.columns if col.startswith("shap_")}
    duplicate_shap_cols = existing_shap_cols.intersection(new_shap_cols)
    if duplicate_shap_cols:
        logger.warning(f"Duplicate shap_ columns detected: {duplicate_shap_cols}. Renaming new shap_ columns to avoid conflicts.")
        feedback_df = feedback_df.rename(columns=lambda x: f"{x}.1" if x in duplicate_shap_cols else x)

    X_inversed = pd.concat([X_inversed.reset_index(drop=True), feedback_df.reset_index(drop=True)], axis=1)
    logger.debug(f"X_inversed shape after merging feedback: {X_inversed.shape}")

    logger.info("[Step B] Computing original metric error with corrected feature extraction.")
    X_inversed = compute_original_metric_error(
        df=X_inversed,
        percentile=metrics_percentile,
        logger=logger
    )

    return X_inversed
