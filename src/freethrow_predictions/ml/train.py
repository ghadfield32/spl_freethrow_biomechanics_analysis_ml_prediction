import logging
import json
from typing import Any, Dict
from pathlib import Path
import yaml
import pandas as pd
import joblib  # Ensure joblib is imported
import xgboost as xgb

from ml.feature_selection.feature_importance_calculator import manage_features

import matplotlib.pyplot as plt

# Local imports - Adjust the import paths based on your project structure
from datapreprocessor import DataPreprocessor
from ml.train.train_utils import (
    evaluate_model, save_model, load_model, plot_decision_boundary,
    tune_random_forest, tune_xgboost, tune_decision_tree
)


logger = logging.getLogger(__name__)



def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with config_path.open('r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise

def bayes_best_model_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selection_metric: str,
    model_save_dir: Path,
    classification_save_path: Path,
    tuning_results_save: Path,
    selected_models: Any,
    use_pca: bool = False
):
    """
    Streamlined function for model tuning, evaluation, and saving the best model.
    """
    logger.info("Starting the Bayesian hyperparameter tuning process...")

    # Scoring metric selection
    scoring_metric = "neg_log_loss" if selection_metric.lower() == "log loss" else "accuracy"

    # Prepare model registry
    model_registry = {
        "XGBoost": tune_xgboost,
        "Random Forest": tune_random_forest,
        "Decision Tree": tune_decision_tree
    }

    # Normalize selected_models input
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    elif not selected_models:
        selected_models = list(model_registry.keys())
        logger.info(f"No models specified. Using all available: {selected_models}")

    tuning_results = {}
    best_model_name = None
    best_model = None
    best_metric_value = None

    # Ensure model_save_dir exists
    model_save_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured that the model save directory '{model_save_dir}' exists.")

    # Define metric key mapping
    metric_key_mapping = {
        "log loss": "Log Loss",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1 score": "F1 Score",
        "roc auc": "ROC AUC"
    }

    # Loop over requested models
    for model_name in selected_models:
        if model_name not in model_registry:
            logger.warning(f"Unsupported model: {model_name}. Skipping.")
            continue
        try:
            logger.info(f"📌 Tuning hyperparameters for {model_name}...")
            tuner_func = model_registry[model_name]

            best_params, best_score, best_estimator = tuner_func(
                X_train, y_train, scoring_metric=scoring_metric
            )
            logger.info(f"✅ {model_name} tuning done. Best Params: {best_params}, Best CV Score: {best_score}")

            # Evaluate on X_test
            metrics = evaluate_model(best_estimator, X_test, y_test, save_path=classification_save_path)
            metric_key = metric_key_mapping.get(selection_metric.lower(), selection_metric)
            metric_value = metrics.get(metric_key)

            # Debugging
            logger.debug(f"Selection Metric Key: {metric_key}")
            logger.debug(f"Available Metrics: {metrics.keys()}")

            if metric_value is not None:
                logger.debug(f"Metric value for {selection_metric}: {metric_value}")
                if best_metric_value is None:
                    best_metric_value = metric_value
                    best_model_name = model_name
                    best_model = best_estimator
                    logger.debug(f"Best model set to {best_model_name} with {selection_metric}={best_metric_value}")
                else:
                    # For log loss, lower is better
                    if selection_metric.lower() == "log loss" and metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_model_name = model_name
                        best_model = best_estimator
                        logger.debug(f"Best model updated to {best_model_name} with {selection_metric}={best_metric_value}")
                    # For other metrics (accuracy, f1, etc.), higher is better
                    elif selection_metric.lower() != "log loss" and metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model_name = model_name
                        best_model = best_estimator
                        logger.debug(f"Best model updated to {best_model_name} with {selection_metric}={best_metric_value}")
            else:
                logger.debug(f"Metric value for {selection_metric} is None. Best model not updated.")

            # Save partial results
            tuning_results[model_name] = {
                "Best Params": best_params,
                "Best CV Score": best_score,
                "Evaluation Metrics": metrics,
            }

            # Plot boundary (optional for tree-based with PCA)
            try:
                plot_decision_boundary(best_estimator, X_test, y_test, f"{model_name} Decision Boundary", use_pca=use_pca)
            except ValueError as e:
                logger.warning(f"Skipping decision boundary plot for {model_name}: {e}")

            # Add feature importance plots for XGBoost
            if model_name.lower() == "xgboost":
                logger.info("Generating feature importance plots for XGBoost...")
                try:
                    xgb.plot_importance(best_model, importance_type="weight")
                    plt.title("Feature Importance by Weight")
                    plt.show()

                    xgb.plot_importance(best_model, importance_type="cover")
                    plt.title("Feature Importance by Cover")
                    plt.show()

                    xgb.plot_importance(best_model, importance_type="gain")
                    plt.title("Feature Importance by Gain")
                    plt.show()
                except Exception as e:
                    logger.error(f"Error generating feature importance plots: {e}")
        except Exception as e:
            logger.error(f"❌ Error tuning {model_name}: {e}")
            continue

    # Save best model information
    if best_model_name:
        logger.info(f"✅ Best model is {best_model_name} with {selection_metric}={best_metric_value}")
        try:
            save_model(best_model, best_model_name, save_dir=model_save_dir)
            logger.info(f"✅ Model '{best_model_name}' saved successfully in '{model_save_dir}'.")
        except Exception as e:
            logger.error(f"❌ Failed to save best model {best_model_name}: {e}")
            raise  # Ensure the exception is propagated

        # Add Best Model info to tuning_results
        tuning_results["Best Model"] = {
            "model_name": best_model_name,
            "metric_value": best_metric_value,
            "path": str(Path(model_save_dir) / best_model_name.replace(" ", "_") / 'trained_model.pkl')
        }
    else:
        logger.warning("⚠️ No best model was selected. Tuning might have failed for all models.")

    # Save tuning results
    try:
        with tuning_results_save.open("w") as f:
            json.dump(tuning_results, f, indent=4)
        logger.info(f"✅ Tuning results saved to {tuning_results_save}.")
    except Exception as e:
        logger.error(f"❌ Error saving tuning results: {e}")



def main():
    # ----------------------------
    # 1. Load Configuration
    # ----------------------------
    config = load_config(Path('../../data/model/preprocessor_config/preprocessor_config.yaml'))

    # Extract paths from configuration
    paths_config = config.get('paths', {})
    base_data_dir = Path(paths_config.get('data_dir', '../../dataset/test/data')).resolve()
    raw_data_file = base_data_dir / paths_config.get('raw_data', 'final_ml_dataset.csv')

    # Output directories
    log_dir = Path(paths_config.get('log_dir', '../preprocessor/logs')).resolve()
    model_save_base_dir = Path(paths_config.get('model_save_base_dir', '../preprocessor/models')).resolve()
    transformers_save_base_dir = Path(paths_config.get('transformers_save_base_dir', '../preprocessor/transformers')).resolve()
    plots_output_dir = Path(paths_config.get('plots_output_dir', '../preprocessor/plots')).resolve()
    training_output_dir = Path(paths_config.get('training_output_dir', '../preprocessor/training_output')).resolve()

    # Initialize Paths for saving
    MODEL_SAVE_DIR = model_save_base_dir
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    CLASSIFICATION_REPORT_PATH = MODEL_SAVE_DIR / "classification_report.txt"
    TUNING_RESULTS_SAVE_PATH = MODEL_SAVE_DIR / "tuning_results.json"


    LOG_FILE = 'training.log'

    # Extract model-related config
    selected_models = config.get('models', {}).get('selected_models', ["XGBoost", "Random Forest", "Decision Tree"])
    selection_metric = config.get('models', {}).get('selection_metric', "Log Loss")

    # Extract Tree Based Classifier options from config
    tree_classifier_options = config.get('models', {}).get('Tree Based Classifier', {})

    # ----------------------------
    # 2. Setup Logging
    # ----------------------------
    logger.info("✅ Starting the training module.")

    # ----------------------------
    # 3. Load Data
    # ----------------------------
    try:
        filtered_df = pd.read_csv(raw_data_file)
        logger.info(f"✅ Loaded dataset from {raw_data_file}. Shape: {filtered_df.shape}")
    except FileNotFoundError:
        logger.error(f"❌ Dataset not found at {raw_data_file}.")
        return
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return


    # Define paths (optional, will use defaults if not provided)
    paths = {
        'features': '../../data/model/pipeline/final_ml_df_selected_features_columns_test.pkl',
        'ordinal_categoricals': '../../data/model/pipeline/features_info/ordinal_categoricals.pkl',
        'nominal_categoricals': '../../data/model/pipeline/features_info/nominal_categoricals.pkl',
        'numericals': '../../data/model/pipeline/features_info/numericals.pkl',
        'y_variable': '../../data/model/pipeline/features_info/y_variable.pkl'
    }

    # Load features and metadata
    loaded = manage_features(
        mode='load',
        paths=paths
    )
    
    # Access loaded data
    if loaded:
        features = loaded.get('features')
        ordinals = loaded.get('ordinal_categoricals')
        nominals = loaded.get('nominal_categoricals')
        nums = loaded.get('numericals')
        y_var = loaded.get('y_variable')
        
        print("\n📥 Loaded Data:")
        print("Features:", features)
        print("Ordinal Categoricals:", ordinals)
        print("Nominal Categoricals:", nominals)
        print("Numericals:", nums)
        print("Y Variable:", y_var)
    # ----------------------------
    # 4. Initialize DataPreprocessor
    # ----------------------------
    # Assuming a supervised classification use case: "Tree Based Classifier"
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=y_var,
        ordinal_categoricals=ordinals,
        nominal_categoricals=nominals,
        numericals=nums, 
        mode='train',
        #options=tree_classifier_options,  # The options from config for "Tree Based Classifier"
        debug=config.get('logging', {}).get('debug', False),  # or config-based
        normalize_debug=config.get('execution', {}).get('train', {}).get('normalize_debug', False),
        normalize_graphs_output=config.get('execution', {}).get('train', {}).get('normalize_graphs_output', False),
        graphs_output_dir=plots_output_dir,
        transformers_dir=transformers_save_base_dir
    )

    # ----------------------------
    # 5. Execute Preprocessing
    # ----------------------------
    try:
        # Execute preprocessing by passing the entire filtered_df
        X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(filtered_df)
        print("types of all variables starting with X_train", type(X_train), "X_test type", type(X_test), "y_train type =", type(y_train), "y_test type =", type(y_test),"X_test_inverse type =", type(X_test_inverse))
        logger.info(f"✅ Preprocessing complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {e}")
        return
 
    # ----------------------------
    # 6. Train & Tune the Model
    # ----------------------------
    try:
        bayes_best_model_train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selection_metric=selection_metric,
            model_save_dir=MODEL_SAVE_DIR,
            classification_save_path=CLASSIFICATION_REPORT_PATH,
            tuning_results_save=TUNING_RESULTS_SAVE_PATH,
            selected_models=selected_models,
            use_pca=True  
        )
    except Exception as e:
        logger.error(f"❌ Model training/tuning failed: {e}")
        return

    # ----------------------------
    # 7. Completion Message
    # ----------------------------
    logger.info("✅ Training workflow completed successfully.")

if __name__ == "__main__":
    main()

