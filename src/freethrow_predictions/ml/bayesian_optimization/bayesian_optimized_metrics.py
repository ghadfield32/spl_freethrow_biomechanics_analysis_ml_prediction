# previously: %%writefile ../../src/freethrow_predictions/ml/bayes_optim_angles_xgboostpreds.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import pickle

# Import the early stopping callback from skopt
from skopt.callbacks import DeltaYStopper

# Configuration and model utilities
from ml.config.config_loader import load_config
from ml.config.config_models import AppConfig
from ml.train_utils.train_utils import load_model

# Updated DataPreprocessor (which now returns X_inversed as part of final_preprocessing)
from datapreprocessor import DataPreprocessor
from ml.feature_selection.feature_importance_calculator import manage_features

# Simple helper for debug prints (logger removed)
def log_debug(message, debug):
    if debug:
        print(message)

def get_preprocessing_results(preprocessor, df, debug):
    """
    Run preprocessing and return X_preprocessed, X_inversed.
    We assume final_preprocessing returns: X_preprocessed, recommendations, X_inversed.
    """
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(df)
        log_debug(f"[Debug] Preprocessing complete. X_preprocessed shape: {X_preprocessed.shape}", debug)
        return X_preprocessed, X_inversed
    except Exception as e:
        log_debug(f"[Error] Preprocessing failed: {e}", debug)
        raise

def compute_real_ranges(X_inversed, optimization_columns, debug):
    """
    Compute the min and max for each optimization column from X_inversed (real domain).
    """
    real_ranges = {}
    for col in optimization_columns:
        # Here we assume that the column names in X_inversed are the original names.
        real_min = float(X_inversed[col].min())
        real_max = float(X_inversed[col].max())
        real_ranges[col] = (real_min, real_max)
    log_debug(f"[Debug] Computed optimization ranges (real domain): {real_ranges}", debug)
    return real_ranges

def map_transformed_to_real(candidate_val, trans_range, real_range):
    """
    Given a candidate value in the transformed domain, linearly map it into
    the real domain using the provided ranges.
    
    Parameters:
      - candidate_val: a value in the transformed domain
      - trans_range: (min, max) tuple from X_preprocessed
      - real_range: (min, max) tuple from X_inversed
      
    Returns:
      The candidate value mapped into the real domain.
    """
    trans_min, trans_max = trans_range
    real_min, real_max = real_range
    # Avoid division by zero by checking if trans_max == trans_min.
    if trans_max == trans_min:
        return real_min
    # Linear mapping formula:
    real_val = real_min + ((candidate_val - trans_min) / (trans_max - trans_min)) * (real_max - real_min)
    return real_val


def compute_optimization_ranges(X_transformed, optimization_columns, debug):
    """
    Compute the min and max for each optimization column from X_transformed.
    If a transformed feature has a prefix (e.g., 'num__'), use that column.
    """
    ranges = {}
    for col in optimization_columns:
        # Determine the key in the transformed data
        transformed_col = f"num__{col}" if f"num__{col}" in X_transformed.columns else col
        ranges[col] = (float(X_transformed[transformed_col].min()), float(X_transformed[transformed_col].max()))
    log_debug(f"[Debug] Computed optimization ranges (transformed domain): {ranges}", debug)
    return ranges

def define_search_space(opt_ranges, optimization_columns, debug):
    """
    Define the search space for Bayesian optimization using ranges from X_transformed.
    """
    missing = [col for col in optimization_columns if col not in opt_ranges]
    if missing:
        raise KeyError(f"Missing columns in optimization ranges: {missing}")
    space = [Real(opt_ranges[col][0], opt_ranges[col][1], name=col) for col in optimization_columns]
    log_debug(f"[Debug] Defined search space: {space}", debug)
    return space



def get_model_feature_order(model):
    if hasattr(model, "get_booster"):
        # XGBoost case
        return model.get_booster().feature_names
    elif hasattr(model, "feature_names_"):
        # CatBoost case
        return model.feature_names_
    else:
        raise AttributeError("The model does not have a known attribute for feature names.")

def objective(params, optimization_columns, X_preprocessed, model, debug):
    # Create the baseline feature vector by taking the mean of X_preprocessed
    feature_vector = X_preprocessed.mean(axis=0).copy()
    
    if debug:
        print("[Debug] Columns in X_preprocessed.mean(axis=0):", feature_vector.index.tolist())
    
    # Update features with candidate parameter values
    for col, value in zip(optimization_columns, params):
        transformed_col = f"num__{col}"
        if transformed_col in feature_vector.index:
            print(f"[Debug] Updating {transformed_col} with value {value}")
            feature_vector[transformed_col] = value
        else:
            print(f"[Debug] Warning: {transformed_col} not found in feature_vector!")
    
    if debug:
        print("[Debug] Columns in feature_vector after assignment:", feature_vector.index.tolist())
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([feature_vector])
    
    # Get the expected feature order using our helper function
    expected_feature_order = get_model_feature_order(model)
    if debug:
        print("[Debug] Expected feature order:", expected_feature_order)
        print("[Debug] Feature DataFrame columns before reindex:", feature_df.columns.tolist())
    
    feature_df = feature_df.reindex(columns=expected_feature_order)
    
    if debug:
        print("[Debug] Feature DataFrame columns after reindex:", feature_df.columns.tolist())
    
    # Get success probability
    success_prob = model.predict_proba(feature_df)[0, 1]
    if debug:
        print(f"[Debug] Objective with params {params}: success_prob = {success_prob:.4f}")
    return -success_prob



def perform_optimization(wrapper_objective, search_space, n_calls, debug,
                         delta_threshold=0.005, n_best=3):  # adjusted values
    """
    Run gp_minimize with the wrapped objective and search space.
    Adds early stopping via a callback if improvement is below delta_threshold
    for n_best iterations.
    """
    stopper = DeltaYStopper(delta=delta_threshold, n_best=n_best)
    callbacks = [stopper]

    result = gp_minimize(func=wrapper_objective, 
                         dimensions=search_space, 
                         n_calls=n_calls, 
                         random_state=42,
                         callback=callbacks)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(result.func_vals) + 1), [-val for val in result.func_vals], marker='o')
        plt.title("Bayesian Optimization Progress (Transformed Domain)")
        plt.xlabel("Iteration")
        plt.ylabel("Success Probability")
        plt.grid(True)
        plt.show()
    return result



def bayesian_optimization_main(config: AppConfig, delta_threshold: float, n_best: float, n_calls: float, df: pd.DataFrame, debug=False):
    # Your implementation here
    if debug:
        print(f"Delta Threshold: {delta_threshold}, Type: {type(delta_threshold)}")
        print(f"n_best: {n_best}, Type: {type(n_best)}")
        print(f"n_calls: {n_calls}, Type: {type(n_calls)}")
    # ----------------------------
    # Step 1: Extract Configuration Values
    # ----------------------------
    data_dir = Path(config.paths.data_dir).resolve()
    features_file = data_dir / config.paths.features_metadata_file
    model_save_dir = Path(config.paths.model_save_base_dir).resolve()
    transformers_dir = Path(config.paths.transformers_save_base_dir).resolve()
    
    # ----------------------------
    # Step 2: Load Optimization Columns via Feature Assets
    # ----------------------------
    try:
        with open(features_file, 'rb') as f:
            selected_features = pickle.load(f)
        # Allow for either list or DataFrame format.
        if isinstance(selected_features, list):
            optimization_columns = selected_features
        else:
            optimization_columns = selected_features.columns.tolist()
        log_debug(f"[Debug] Loaded optimization columns: {optimization_columns}", debug)
    except Exception as e:
        print(f"[Error] Failed to load selected features: {e}")
        raise

    # Remove y_variable(s) from optimization columns if present.
    y_variables = config.features.y_variable  # List of target variables
    log_debug(f"[Debug] Target variables to remove: {y_variables}", debug)
    for y_var in y_variables:
        if y_var in optimization_columns:
            optimization_columns.remove(y_var)
            log_debug(f"[Debug] Removed '{y_var}' from optimization columns.", debug)
        else:
            log_debug(f"[Debug] '{y_var}' not found in optimization columns.", debug)
    
    log_debug(f"[Debug] Final optimization columns: {optimization_columns}", debug)
    
    # ----------------------------
    # Step 3: Initialize DataPreprocessor Using Column Assets
    # ----------------------------
    paths = config.paths
    feature_paths = {
        'features': Path('../../data/preprocessor/features_info/final_ml_df_selected_features_columns.pkl'),
        'ordinal_categoricals': Path('../../data/preprocessor/features_info/ordinal_categoricals.pkl'),
        'nominal_categoricals': Path('../../data/preprocessor/features_info/nominal_categoricals.pkl'),
        'numericals': Path('../../data/preprocessor/features_info/numericals.pkl'),
        'y_variable': Path('../../data/preprocessor/features_info/y_variable.pkl')
    }

    try:
        feature_lists = manage_features(mode='load', paths=feature_paths)
        y_variable_list = feature_lists.get('y_variable', [])
        ordinal_categoricals = feature_lists.get('ordinal_categoricals', [])
        nominal_categoricals = feature_lists.get('nominal_categoricals', [])
        numericals = feature_lists.get('numericals', [])
    except Exception as e:
        print(f"[Error] Failed to load feature lists: {e}")
        raise

    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=y_variable_list,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode='predict',  # The same mode as in predict pipelines
        options={},
        debug=False,
        normalize_debug=False,
        normalize_graphs_output=False,
        graphs_output_dir=Path(config.paths.plots_output_dir).resolve(),
        transformers_dir=transformers_dir
    )

    # ----------------------------
    # Step 4: Preprocess the Data (Transformed Domain)
    # ----------------------------
    X_preprocessed, X_inversed = get_preprocessing_results(preprocessor, df, debug)
    # Compute optimization ranges using X_preprocessed (transformed space)
    opt_ranges = compute_optimization_ranges(X_preprocessed, optimization_columns, debug)
    
    # ----------------------------
    # Step 5: Load the Trained Model
    # ----------------------------
    # For example, load the best model based on tuning info; here we assume you know the best model name.
    trained_model = load_model('CatBoost', model_save_dir)  # Replace with your logic if available

    # ----------------------------
    # Step 6: Define the Search Space and Objective Function
    # ----------------------------
    search_space = define_search_space(opt_ranges, optimization_columns, debug)
    wrapper_objective = lambda params: objective(params, optimization_columns, X_preprocessed, trained_model, debug)

    # ----------------------------
    # Step 7: Perform Bayesian Optimization with Early Stopping
    # ----------------------------
    res = perform_optimization(wrapper_objective, search_space, n_calls=n_calls, debug=debug,
                               delta_threshold=delta_threshold, n_best=n_best)
    params_df = pd.DataFrame(res.x_iters, columns=optimization_columns)
    params_df['success_prob'] = [-val for val in res.func_vals]

    # ----------------------------
    # Step 8: Compare Baseline vs. Optimized (in Transformed Domain)
    # ----------------------------
    # Use the preprocessed (transformed) mean for the baseline.
    baseline_feature_vector = X_preprocessed.mean(axis=0)
    print("[Debug] Baseline feature vector (transformed):", baseline_feature_vector)
    baseline_df = pd.DataFrame([baseline_feature_vector])
    # Use this:
    expected_feature_order = get_model_feature_order(trained_model)
    baseline_df = baseline_df.reindex(columns=expected_feature_order)
    print("[Debug] Baseline DataFrame columns:", baseline_df.columns.tolist())
    baseline_success = trained_model.predict_proba(baseline_df)[0, 1]

    print(f"[Debug] Baseline success probability: {baseline_success:.4f}")

    # ----- Step 8: Comparison using Real Numbers -----

    # (A) Compute the baseline (real) values using X_inversed.
    baseline_real = {}
    for col in optimization_columns:
        baseline_real[col] = X_inversed[col].mean()
    baseline_real_series = pd.Series(baseline_real)

    # (B) Compute candidate (optimized) parameters in the real domain.
    # We already have candidate values (res.x) in the transformed domain.
    real_ranges = compute_real_ranges(X_inversed, optimization_columns, debug)
    candidate_real = {}
    for col, cand_val in zip(optimization_columns, res.x):
        trans_range = opt_ranges[col]  # from transformed X_preprocessed
        candidate_real[col] = map_transformed_to_real(cand_val, trans_range, real_ranges[col])
    candidate_real_series = pd.Series(candidate_real)

    # (C) Build the parameter comparison table
    min_values_real = [real_ranges[col][0] for col in optimization_columns]
    max_values_real = [real_ranges[col][1] for col in optimization_columns]

    comparison_real = pd.DataFrame({
        "Parameter": optimization_columns,
        "Baseline (Real)": baseline_real_series.values,
        "Optimized (Candidate, Real)": candidate_real_series.values,
        "Difference": candidate_real_series.values - baseline_real_series.values,
        "Min (Real)": min_values_real,
        "Max (Real)": max_values_real
    })

    # (D) Compute success rates from the model.
    baseline_success = trained_model.predict_proba(pd.DataFrame([X_preprocessed.mean(axis=0)]))[0, 1]
    candidate_success = -res.fun  # recall objective returns negative success probability

    # Instead of appending a separate row, add new columns for the success rates.
    comparison_real["Success Rate (Baseline)"] = baseline_success
    comparison_real["Success Rate (Candidate)"] = candidate_success
    comparison_real["Success Rate Diff"] = candidate_success - baseline_success

    print("Comparison of Baseline vs. Optimized Parameters (Real Domain):")
    print(comparison_real)
    
    # save the results to data\predictions\bayesian_optimization_results
    output_dir = Path(config.paths.predictions_output_dir).resolve() / 'bayesian_optimization_results'
    comparison_real.to_csv(output_dir / 'bayesian_optimization_results.csv', index=False)

    return comparison_real

if __name__ == "__main__":
    config_path = Path('../../data/model/preprocessor_config/preprocessor_config.yaml')
    try:
        config: AppConfig = load_config(config_path)
        print(f"Config loaded from {config_path}")
    except Exception as e:
        print("Failed to load config:", e)
        exit(1)
    
    data_dir = Path(config.paths.data_dir).resolve()
    df_path = data_dir / config.paths.raw_data
    df = pd.read_csv(df_path)
    
    results = bayesian_optimization_main(config, 
                                         delta_threshold=0.001, # the minimum improvement (change in the objective value) that must be observed for the optimizer to consider a new candidate as “better.”
                                         n_best=5, #the number of successive iterations that are compared to decide if the improvement is below the delta_threshold.
                                         n_calls=50, #maximum number of function evaluations (iterations) that the optimization algorithm will perform.
                                         df=df, 
                                         debug=True)
