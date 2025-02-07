
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from ml.shap.shap_calculator import ShapCalculator
from ml.shap.shap_utils import load_dataset, setup_logging, load_configuration, initialize_logger

class FeedbackGenerator:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the FeedbackGenerator with an optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def generate_global_recommendations(
        self,
        shap_values: np.ndarray,
        X_original: pd.DataFrame,
        top_n: int = 5,
        use_mad: bool = False,
        debug: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Generating feature importance based on SHAP values...")
        self.logger.debug(f"Received SHAP values with shape: {np.shape(shap_values)}")
        
        # Handle 3D array by selecting one slice if necessary.
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            self.logger.warning(f"SHAP values are 3D with shape {shap_values.shape}; selecting one slice for global recommendations.")
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
                self.logger.debug("Selected the positive class slice (index 1) for SHAP values.")
            else:
                error_msg = f"Unexpected 3D shape for SHAP values: {shap_values.shape}. Cannot determine which slice to use."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        try:
            shap_df = pd.DataFrame(shap_values, columns=X_original.columns)
            feature_importance = pd.DataFrame({
                'feature': X_original.columns,
                'importance': np.abs(shap_df).mean(axis=0),
                'mean_shap': shap_df.mean(axis=0)
            }).sort_values(by='importance', ascending=False)
            if debug:
                self.logger.debug(f"Feature importance (top {top_n}):\n{feature_importance.head(top_n)}")
            top_features = feature_importance.head(top_n)['feature'].tolist()
            recommendations = {}
            for feature in top_features:
                feature_values = X_original[feature]
                range_str = self._compute_feature_range(feature_values, use_mad, debug)
                mean_shap = feature_importance.loc[feature_importance['feature'] == feature, 'mean_shap'].values[0]
                direction = 'increase' if mean_shap > 0 else 'decrease'
                recommendations[feature] = {
                    'range': range_str,
                    'importance': round(feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0], 4),
                    'direction': direction
                }
                if debug:
                    self.logger.debug(
                        f"Recommendation for {feature}: Range={range_str}, "
                        f"Importance={feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0]}, "
                        f"Direction={direction}"
                    )
            if debug:
                self.logger.debug(f"Final Recommendations with Importance and Direction: {recommendations}")
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to generate global recommendations: {e}")
            raise



    def _compute_feature_range(self, feature_values: pd.Series, use_mad: bool, debug: bool) -> str:
        if use_mad:
            median = feature_values.median()
            mad = feature_values.mad()
            lower_bound = median - 1.5 * mad
            upper_bound = median + 1.5 * mad
            range_str = f"{lower_bound:.1f}–{upper_bound:.1f}"
            if debug:
                self.logger.debug(f"Computed MAD-based range for feature: {range_str}")
        else:
            lower_bound = feature_values.quantile(0.25)
            upper_bound = feature_values.quantile(0.75)
            range_str = f"{lower_bound:.1f}–{upper_bound:.1f}"
            if debug:
                self.logger.debug(f"Computed IQR-based range for feature: {range_str}")
        return range_str

    def _normalize_feature_name(self, name: str) -> str:
        # Ensure the name is a string.
        return str(name).replace("num__", "").replace("cat__", "")

    def generate_individual_feedback(
        self,
        trial: pd.Series,
        shap_values_trial: np.ndarray,
        percentile: float = 10.0,
        expected_features: Optional[List[str]] = None,
        # New optional parameter: the reference dataset from which to compute the metric min and max.
        reference_dataset: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate individual feedback for a trial.
        This version normalizes feature names from both the expected list and trial index.
        In addition to numeric outputs, a detailed textual explanation is provided.

        Parameters:
            trial (pd.Series): The trial data (e.g. from the inverse transform).
            shap_values_trial (np.ndarray): The SHAP values for this trial.
            percentile (float): The input percentile used to scale the tolerance.
            expected_features (Optional[List[str]]): The list of expected features.
            reference_dataset (Optional[pd.DataFrame]): A DataFrame containing the reference data.
                If provided, for each feature the tolerance is computed as:
                    tolerance = ((dataset_max - dataset_min) * (percentile/100.0)) / 2
                Otherwise, the tolerance is computed using the default method.

        Returns:
            Dict[str, Any]: A dictionary with detailed feedback for each feature.
        """
        if expected_features is None:
            expected_features = trial.index.tolist()
        
        # Normalize expected features and trial keys.
        norm_expected = {self._normalize_feature_name(f) for f in expected_features}
        norm_trial = {self._normalize_feature_name(f) for f in trial.index.tolist()}
        
        self.logger.debug(f"Normalized expected features: {norm_expected}")
        self.logger.debug(f"Normalized trial features: {norm_trial}")
        
        # Log the raw trial keys for extra visibility.
        self.logger.debug(f"Original trial keys: {list(trial.index)}")
        
        missing = norm_expected - norm_trial
        if missing:
            self.logger.error(f"Critical feature mismatch after normalization: {missing}")
            raise ValueError(f"Feature alignment failed: {missing}")

        # Check if shap_values_trial is 2D; if so, select one column.
        shape_trial = np.shape(shap_values_trial)
        self.logger.debug(f"SHAP values for trial before processing: shape = {shape_trial}")
        if len(shape_trial) == 2 and shape_trial[1] == 2:
            self.logger.debug("Detected 2D SHAP values for individual trial; selecting positive class (index 1).")
            shap_values_trial = shap_values_trial[:, 1]
            self.logger.debug(f"New shape of SHAP values for trial: {np.shape(shap_values_trial)}")

        # If the SHAP output is a scalar, wrap it in a list.
        if np.isscalar(shap_values_trial):
            shap_values_trial = [shap_values_trial]
            self.logger.debug("SHAP output was a scalar; wrapped into a list.")

        num_expected = len(expected_features)
        num_shap = len(shap_values_trial)
        if num_expected != num_shap:
            self.logger.warning(f"Length mismatch: expected features length = {num_expected} but SHAP values length = {num_shap}. Iterating up to the minimum length.")
            min_len = min(num_expected, num_shap)
        else:
            min_len = num_expected

        feedback = {}
        for i in range(min_len):
            orig_feature = expected_features[i]
            shap_val = shap_values_trial[i]
            norm_feature = self._normalize_feature_name(orig_feature)
            
            # Try to match the normalized key; if not found, try with and without the 'num__' prefix.
            matching_keys = [k for k in trial.index if self._normalize_feature_name(k) == norm_feature]
            if not matching_keys:
                alt_key = "num__" + norm_feature
                if alt_key in trial.index:
                    matching_keys = [alt_key]
                else:
                    self.logger.debug(f"No matching key found in trial for normalized feature '{norm_feature}'. Skipping.")
                    continue
            
            trial_key = matching_keys[0]
            trial_value = trial.get(trial_key, None)
            
            self.logger.debug(f"Processing feature '{orig_feature}' (normalized: '{norm_feature}'): SHAP value = {shap_val}, Trial value = {trial_value}")
            
            # Updated type checks to accept numpy numeric types as well.
            if not isinstance(shap_val, (int, float, np.number)) or pd.isna(shap_val):
                self.logger.debug(f"Invalid SHAP value type {type(shap_val)} for '{orig_feature}' ({shap_val}). Skipping.")
                continue
            if not isinstance(trial_value, (int, float, np.number)) or pd.isna(trial_value):
                self.logger.debug(f"Invalid trial value type {type(trial_value)} for '{orig_feature}' ({trial_value}). Skipping.")
                continue
            
            suggestion = "increase" if shap_val > 0 else "decrease"
            adjustment_factor = 0.1
            unit_change_value = adjustment_factor * abs(trial_value)
            goal_value = trial_value + unit_change_value if suggestion == "increase" else trial_value - unit_change_value
            unit = "units"
            
            # --- New Min/Max Strategy Implementation ---
            if reference_dataset is not None:
                self.logger.debug(f"Reference dataset provided for feature '{orig_feature}'. Available columns: {list(reference_dataset.columns)}")
                # Try to locate the corresponding column in the reference dataset by matching normalized names.
                ref_col = None
                for col in reference_dataset.columns:
                    normalized_col = self._normalize_feature_name(col)
                    self.logger.debug(f"Checking reference dataset column '{col}' normalized as '{normalized_col}' against normalized feature '{norm_feature}'")
                    if normalized_col == norm_feature:
                        ref_col = col
                        break
                if ref_col is not None:
                    metric_min = reference_dataset[ref_col].min()
                    metric_max = reference_dataset[ref_col].max()
                    range_span = metric_max - metric_min
                    total_adjustment = range_span * (percentile / 100.0)
                    tolerance = total_adjustment / 2.0
                    self.logger.debug(
                        f"Using dataset-based tolerance for '{orig_feature}': "
                        f"ref_col='{ref_col}', metric_min={metric_min}, metric_max={metric_max}, "
                        f"range_span={range_span}, total_adjustment={total_adjustment}, tolerance={tolerance}"
                    )
                else:
                    self.logger.debug(f"Reference dataset provided but no matching column found for '{orig_feature}' (normalized as '{norm_feature}').")
                    tolerance = 0
                    self.logger.debug(f"Falling back to default tolerance: {tolerance}")
            else:
                tolerance = 0
                self.logger.debug(f"Reference dataset not provided. Using default tolerance for '{orig_feature}': tolerance={tolerance}")
            # Set the acceptable range around the goal.
            min_value = goal_value - tolerance
            max_value = goal_value + tolerance
            # --- End New Strategy ---

            # --- End New Strategy ---
            
            feedback_text = (
                f"For feature '{orig_feature}': The SHAP value of {shap_val:.2f} suggests to {suggestion} the value. "
                f"Current value is {trial_value:.2f}. A 10% adjustment equals {unit_change_value:.2f} {unit}, "
                f"which would set a target of {goal_value:.2f}. The acceptable range is [{min_value:.2f}, {max_value:.2f}], "
                f"classifying this metric as '{self.classify_metric(trial_value, min_value, max_value)}'."
            )

            feedback[f"shap_{norm_feature}_unit_change"] = round(unit_change_value, 2)
            feedback[f"shap_{norm_feature}_unit"] = unit
            feedback[f"shap_{norm_feature}_direction"] = suggestion
            feedback[f"shap_{norm_feature}_importance"] = round(abs(shap_val), 4)
            feedback[f"shap_{norm_feature}_goal"] = round(goal_value, 3)
            feedback[f"shap_{norm_feature}_min"] = round(min_value, 3)
            feedback[f"shap_{norm_feature}_max"] = round(max_value, 3)
            feedback[f"shap_{norm_feature}_classification"] = self.classify_metric(trial_value, min_value, max_value)
            feedback[f"shap_{norm_feature}_feedback_text"] = feedback_text
            
            self.logger.debug(
                f"For feature '{norm_feature}': suggestion={suggestion}, goal={goal_value:.3f}, "
                f"min={min_value:.3f}, max={max_value:.3f}, classification={self.classify_metric(trial_value, min_value, max_value)}, "
                f"trial value={trial_value}, unit_change={unit_change_value:.3f} {unit}. "
                f"Feedback text: {feedback_text}"
            )
        
        if not feedback:
            self.logger.warning("No individual feedback was generated; returning raw SHAP values for each feature.")
            feedback = {}
            for f in expected_features:
                norm_f = self._normalize_feature_name(f)
                value = trial.get(norm_f)
                if value is None:
                    value = trial.get("num__" + norm_f)
                feedback[norm_f] = f"Raw SHAP value: {value}"
        
        return feedback



    def classify_metric(self, value: float, min_val: float, max_val: float) -> str:
        if value < min_val:
            return "Early"
        elif value > max_val:
            return "Late"
        else:
            return "Good"


if __name__ == "__main__":
    # Test code to verify the FeedbackGenerator class
    print("Testing FeedbackGenerator module...")

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
    model = 'Random Forest' #XGBoost, CatBoost, Random Forest
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
        model = load_model(model, Path(config.paths.model_save_base_dir).resolve())
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
        logger.info("Preprocessing completed successfully in predict mode.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        exit(1)

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
        exit(1)

    # **Compute SHAP Values**
    try:
        explainer, shap_values = shap_calculator.compute_shap_values(X_preprocessed, debug=True)
        print("✅ SHAP values computed successfully.")
        logger.info(f"SHAP values computed with shape: {shap_values.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to compute SHAP values: {e}")
        exit(1)

    # **Initialize FeedbackGenerator**
    feedback_generator = FeedbackGenerator(logger=logger)
    print("✅ FeedbackGenerator initialized successfully.")
    logger.info("FeedbackGenerator initialized successfully.")

    # **Generate Global Recommendations**
    try:
        recommendations = feedback_generator.generate_global_recommendations(
            shap_values=shap_values,
            X_original=X_preprocessed,
            top_n=5,
            use_mad=False,
            debug=True
        )
        print("✅ Global recommendations generated successfully:")
        for feature, rec in recommendations.items():
            print(f"  - {feature}: {rec}")
        logger.info("Global recommendations generated successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to generate global recommendations: {e}")

    # **Generate Individual Feedback for a Specific Trial**
    try:
        trial_id = 1  # Intended trial id
        trial_index = X_preprocessed.index.get_loc(trial_id)
        shap_values_trial = shap_values[trial_index]  # Use the row corresponding to trial_id
        expected_features = X_preprocessed.columns.tolist()
        print("expected_features", expected_features)
        feedback = feedback_generator.generate_individual_feedback(trial=X_inversed.loc[trial_id],
                                                                shap_values_trial=shap_values_trial,
                                                                percentile=10,
                                                                expected_features=expected_features,
                                                                reference_dataset=X_inversed)


        print(f"\n✅ Individual feedback for trial '{trial_id}':")
        for metric, suggestion in feedback.items():
            print(f"  - {metric}: {suggestion}")
        logger.info(f"Individual feedback for trial '{trial_id}' generated successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to generate individual feedback for trial '{trial_id}': {e}")

    print("✅ All tests in feedback_generator.py passed successfully.")
