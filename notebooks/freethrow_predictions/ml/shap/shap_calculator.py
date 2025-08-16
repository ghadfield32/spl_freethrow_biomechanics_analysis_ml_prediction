import logging
import shap
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from ml.config.config_models import AppConfig
from ml.config.config_loader import load_config
from ml.shap.shap_utils import load_dataset, setup_logging, load_configuration, initialize_logger

class ShapCalculator:
    def __init__(self, model, model_type: Optional[str] = None, logger: logging.Logger = None):
        """        
        Initialize the ShapCalculator with a model and an optional logger.
        Detects the appropriate SHAP explainer based on model type.
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.explainer = None
        self.model_type = model_type or self._infer_model_type()
        self.logger.debug(f"ShapCalculator initialized with model type: {self.model_type}")
    
    def _infer_model_type(self) -> str:
        import xgboost as xgb
        if isinstance(self.model, xgb.XGBModel):
            self.logger.debug("XGBoost model detected via XGBModel interface.")
            return "xgboost"
        if isinstance(self.model, xgb.Booster):
            self.logger.debug("XGBoost Booster instance detected.")
            return "xgboost"
        if hasattr(self.model, "get_booster"):
            self.logger.debug("XGBoost model detected via get_booster().")
            return "xgboost"
        elif hasattr(self.model, 'tree_structure__'):
            return 'tree'
        elif hasattr(self.model, 'coef_'):
            return 'linear'
        elif hasattr(self.model, 'layers_'):
            return 'deep'
        else:
            self.logger.warning("Unable to infer model type. Defaulting to 'tree'.")
            return 'tree'

    def compute_shap_values(self, X: pd.DataFrame, debug: bool = False) -> Tuple[shap.Explainer, np.ndarray]:
        self.logger.info("Initializing SHAP explainer...")
        try:
            # Ensure all features are numeric.
            non_numeric_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if non_numeric_features:
                self.logger.error(f"Non-numeric features detected in input data: {non_numeric_features}")
                raise ValueError(f"Preprocessed data contains non-numeric features: {non_numeric_features}")
            else:
                self.logger.debug("All features are numeric.")

            # For XGBoost, align feature names if needed.
            if self.model_type == 'xgboost':
                booster = self.model.get_booster()
                expected_features = booster.feature_names
                if expected_features is None:
                    self.logger.warning("No feature names found in the booster; skipping feature alignment.")
                else:
                    missing = set(expected_features) - set(X.columns)
                    if missing:
                        self.logger.error(f"Missing features in SHAP input: {missing}")
                        raise ValueError("Feature mismatch between model and input data")
                    self.logger.debug(f"Reordering features to match expected order: {expected_features}")
                    X = X[expected_features]

            # Choose explainer.
            if self.model_type in ['tree', 'xgboost']:
                model_to_use = self.model if self.model_type == 'tree' else self.model.get_booster()
                self.logger.debug("Using SHAP TreeExplainer.")
                self.explainer = shap.TreeExplainer(model_to_use, feature_perturbation="tree_path_dependent")
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, X, feature_dependence="independent")
            elif self.model_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model, X)
            else:
                self.logger.warning(f"Unrecognized model type '{self.model_type}'. Defaulting to TreeExplainer.")
                self.explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")

            if debug:
                self.logger.debug(f"SHAP Explainer initialized: {type(self.explainer)}")
                self.logger.debug(f"Explainer details: {self.explainer}")

            # Compute SHAP values.
            shap_values = self.explainer.shap_values(X)
            self.logger.debug(f"Type of shap_values before conversion: {type(shap_values)}")
            if hasattr(shap_values, "values"):
                self.logger.debug("Converting shap_values to numpy array using .values attribute.")
                shap_values = shap_values.values

            if debug:
                self.logger.debug(f"SHAP values computed: {type(shap_values)}")
                self.logger.debug(f"Shape of SHAP values: {np.shape(shap_values)}")

            # If the SHAP values array is 3D, slice to 2D.
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                if shap_values.shape[2] == 2:
                    self.logger.warning(f"SHAP values are 3D with shape {shap_values.shape}; selecting slice index 1.")
                    shap_values = shap_values[:, :, 1]
                else:
                    self.logger.error(f"Unexpected SHAP values shape: {shap_values.shape}")
                    raise ValueError("Unexpected SHAP values shape.")

            # If the SHAP values come as a list (e.g., multiclass), select the appropriate element.
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    self.logger.debug("Multiclass detected; selecting positive class (index 1).")
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]

            if shap_values.ndim == 1:
                shap_values = np.atleast_2d(shap_values)
                self.logger.debug("Converted 1D SHAP output to 2D row vector.")

            if hasattr(self.model, 'classes_'):
                n_classes = len(self.model.classes_)
            elif hasattr(self.model, 'n_classes_'):
                n_classes = self.model.n_classes_
            else:
                n_classes = 1
            self.logger.debug(f"Number of classes in the model: {n_classes}")

            shap_values_class = self._process_shap_values(shap_values, n_classes, debug)
            return self.explainer, shap_values_class
        except Exception as e:
            self.logger.error(f"Failed to compute SHAP values: {e}")
            raise


    def _process_shap_values(self, shap_values, n_classes, debug: bool):
        if isinstance(shap_values, list):
            if n_classes > 1:
                shap_values_class = shap_values[1]
                self.logger.debug(f"Extracted SHAP values for class 1: Shape {shap_values_class.shape}")
            else:
                shap_values_class = shap_values[0]
                self.logger.debug(f"Extracted SHAP values for single-class: Shape {shap_values_class.shape}")
        elif isinstance(shap_values, np.ndarray):
            shap_values_class = shap_values
            self.logger.debug(f"SHAP values array shape: {shap_values_class.shape}")
        else:
            self.logger.error(f"Unexpected SHAP values type: {type(shap_values)}")
            raise ValueError("Unexpected SHAP values type.")
        return shap_values_class

    def compute_individual_shap_values(self, X_transformed: pd.DataFrame, trial_index: int, debug: bool = False) -> np.ndarray:
        """
        Compute SHAP values for a single trial.
        For models like XGBoost, the output may be a list; if so, select the positive class slice and squeeze the trial dimension.
        Returns a 1D numpy array of SHAP values for the trial.
        """
        self.logger.info(f"Computing SHAP values for trial at index {trial_index}...")
        try:
            trial = X_transformed.iloc[[trial_index]]  # Keep as DataFrame for SHAP explainer
            shap_values_trial = self.explainer.shap_values(trial)
            self.logger.debug(f"Raw SHAP values for trial (type: {type(shap_values_trial)}): {shap_values_trial}")
            
            # If the result is a list (e.g., for multiclass/binary classifiers)
            if isinstance(shap_values_trial, list):
                if len(shap_values_trial) > 1:
                    self.logger.debug("Multiclass detected in individual trial; selecting positive class (index 1).")
                    shap_values_trial = shap_values_trial[1]
                else:
                    shap_values_trial = shap_values_trial[0]
            
            # At this point, shap_values_trial should have shape (1, n_features)
            # Squeeze the trial dimension so that we have a 1D array of length n_features.
            shap_values_trial = np.squeeze(shap_values_trial, axis=0)
            self.logger.debug(f"Processed SHAP values for trial (after squeezing): {shap_values_trial} with shape {np.shape(shap_values_trial)}")
            
            # NEW: Log the dtype so we know it is numeric
            self.logger.debug(f"Data type of individual SHAP values: {shap_values_trial.dtype}")
            return shap_values_trial
        except Exception as e:
            self.logger.error(f"Failed to compute SHAP values for trial {trial_index}: {e}")
            raise



    def extract_force_plot_values(self, shap_values: np.ndarray, trial_id: Any, X_original: pd.DataFrame) -> Dict[str, Any]:
        self.logger.info(f"Extracting SHAP values for trial ID '{trial_id}'...")
        try:
            if trial_id not in X_original.index:
                self.logger.warning(f"Trial ID '{trial_id}' not found in X_original index.")
                return {}
            pos = X_original.index.get_loc(trial_id)
            self.logger.debug(f"Trial ID '{trial_id}' is at position {pos}.")
            shap_values_trial = shap_values[pos]
            feature_contributions = dict(zip(X_original.columns, shap_values_trial))
            self.logger.debug(f"Feature contributions for trial '{trial_id}': {feature_contributions}")
            return feature_contributions
        except Exception as e:
            self.logger.error(f"Error extracting SHAP values for trial '{trial_id}': {e}")
            raise

    def get_shap_row(self, shap_values: np.ndarray, df: pd.DataFrame, trial_id: Any) -> Optional[np.ndarray]:
        self.logger.info(f"Retrieving SHAP values for trial ID '{trial_id}'...")
        try:
            if trial_id not in df.index:
                self.logger.warning(f"Trial ID '{trial_id}' not found in DataFrame index.")
                return None
            pos = df.index.get_loc(trial_id)
            shap_row = shap_values[pos]
            self.logger.debug(f"SHAP values for trial ID '{trial_id}' at position {pos}: {shap_row}")
            return shap_row
        except Exception as e:
            self.logger.error(f"Error retrieving SHAP row for trial ID '{trial_id}': {e}")
            raise

# The main execution block remains largely the same but now benefits from the enhanced ShapCalculator.

if __name__ == "__main__":
    # Test code to verify the ShapCalculator class
    print("Testing ShapCalculator module...")
    from ml.train_utils.train_utils import load_model
    from datapreprocessor import DataPreprocessor
    from ml.predict.predict import predict_and_attach_predict_probs
    from ml.feature_selection.feature_importance_calculator import manage_features

    # Import utility functions
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
        model = load_model('XGBoost', Path(config.paths.model_save_base_dir).resolve())
        print("✅ Model loaded successfully.")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        exit(1)

    # **Initialize ShapCalculator with Dynamic Model Type**
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

    # **Compute Individual SHAP Values for a Specific Trial**
    try:
        trial_id = X_inversed.index[0]  # Example: first trial
        trial_index = X_preprocessed.index.get_loc(trial_id)
        shap_values_trial = shap_calculator.compute_individual_shap_values(X_preprocessed, trial_index, debug=True)
        print(f"✅ SHAP values for trial '{trial_id}' computed successfully.")
        logger.info(f"SHAP values for trial '{trial_id}' computed.")
    except Exception as e:
        logger.error(f"❌ Failed to compute SHAP values for trial '{trial_id}': {e}")

    # **Extract Force Plot Values for a Specific Trial**
    try:
        feature_contributions = shap_calculator.extract_force_plot_values(shap_values, trial_id, df)
        print(f"✅ Feature contributions for trial '{trial_id}' extracted successfully:")
        for feature, contribution in feature_contributions.items():
            print(f"  - {feature}: {contribution}")
        logger.info(f"Feature contributions for trial '{trial_id}' extracted.")
    except Exception as e:
        logger.error(f"❌ Failed to extract feature contributions for trial '{trial_id}': {e}")

    # **Retrieve SHAP Row for a Specific Trial**
    try:
        shap_row = shap_calculator.get_shap_row(shap_values, df, trial_id)
        if shap_row is not None:
            print(f"✅ Retrieved SHAP row for trial '{trial_id}': {shap_row}")
            logger.info(f"SHAP row for trial '{trial_id}' retrieved.")
        else:
            print(f"⚠️ SHAP row for trial '{trial_id}' not found.")
            logger.warning(f"SHAP row for trial '{trial_id}' not found.")
    except Exception as e:
        logger.error(f"❌ Failed to retrieve SHAP row for trial '{trial_id}': {e}")

    print("✅ All tests in shap_calculator.py passed successfully.")
