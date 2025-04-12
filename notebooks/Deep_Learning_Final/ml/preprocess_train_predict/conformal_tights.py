
import numpy as np
import pandas as pd
import json
import sys
import logging
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from ml.load_and_prepare_data.load_data_and_analyze import (
    load_data, prepare_joint_features, feature_engineering, summarize_data, check_and_drop_nulls,
    prepare_base_datasets)

from ml.feature_selection.feature_selection import (
    load_top_features, perform_feature_importance_analysis, save_top_features,
    analyze_joint_injury_features, check_for_invalid_values,
    perform_feature_importance_analysis, analyze_and_display_top_features,
    run_feature_import_and_load_top_features)

from ml.preprocess_train_predict.base_training import (
    temporal_train_test_split, scale_features, create_sequences, train_exhaustion_model, 
    train_injury_model,  train_joint_models, forecast_and_plot_exhaustion, forecast_and_plot_injury,
    forecast_and_plot_joint, summarize_regression_model, summarize_classification_model, 
    summarize_joint_models, summarize_all_models, final_model_summary, 
    summarize_joint_exhaustion_models
    )



# ==================== CONFORMAL UNCERTAINTY FUNCTIONS ====================
def train_conformal_model(train_data, test_data, features, target_col="by_trial_exhaustion_score", 
                         estimator=None):
    """
    Trains a scikit-learn model with Conformal Tights for uncertainty quantification.
    
    Parameters:
      - train_data (DataFrame): Training data.
      - test_data (DataFrame): Testing data.
      - features (list): List of feature column names.
      - target_col (str): Target variable name (default "by_trial_exhaustion_score").
      - estimator: A scikit-learn estimator. If None, uses XGBRegressor by default.
      
    Returns:
      - conformal_predictor: Fitted ConformalCoherentQuantileRegressor.
      - X_test (numpy array): Test features.
      - y_test (numpy array): Test target values.
    """
    from conformal_tights import ConformalCoherentQuantileRegressor
    from xgboost import XGBRegressor
    
    # Use XGBRegressor as the default estimator if none is provided.
    if estimator is None:
        estimator = XGBRegressor(objective="reg:squarederror")
    
    # Extract features and target from train and test data.
    X_train = train_data[features].values
    y_train = train_data[target_col].values
    X_test = test_data[features].values
    y_test = test_data[target_col].values
    
    # Create and fit the conformal predictor.
    conformal_predictor = ConformalCoherentQuantileRegressor(estimator=estimator)
    conformal_predictor.fit(X_train, y_train)
    
    logging.info(f"Trained conformal model for target '{target_col}' using {len(features)} features")
    
    return conformal_predictor, X_test, y_test

def predict_with_uncertainty(conformal_predictor, X_test, y_test, 
                           quantiles=(0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975)):
    """
    Makes predictions with uncertainty estimates.
    
    Parameters:
      - conformal_predictor: Fitted conformal predictor.
      - X_test (numpy array): Test features.
      - y_test (numpy array): True target values.
      - quantiles (tuple): Quantiles to compute.
      
    Returns:
      - y_pred: Point predictions.
      - y_quantiles: DataFrame of quantile predictions.
      - y_interval: Prediction interval for 95% coverage.
    """
    # Make point predictions.
    y_pred = conformal_predictor.predict(X_test)
    
    # Get quantile predictions (as a DataFrame).
    y_quantiles = conformal_predictor.predict_quantiles(X_test, quantiles=quantiles)
    
    # Get interval predictions (95% coverage).
    y_interval = conformal_predictor.predict_interval(X_test, coverage=0.95)
    
    # Calculate empirical coverage.
    lower_bound = y_interval[:, 0]
    upper_bound = y_interval[:, 1]
    empirical_coverage = np.mean((lower_bound <= y_test) & (y_test <= upper_bound))
    print(f"Empirical coverage: {empirical_coverage:.4f}")
    
    return y_pred, y_quantiles, y_interval

def plot_conformal_results(X_test, y_test, y_pred, y_quantiles, y_interval, sample_size=50):
    """
    Plots predictions with uncertainty for a subset of test points.
    
    Parameters:
      - X_test: Test features (not used directly in plot).
      - y_test: True target values.
      - y_pred: Point predictions.
      - y_quantiles: DataFrame of quantile predictions.
      - y_interval: Array of prediction intervals.
      - sample_size: Number of sample points to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Select sample indices.
    if len(y_test) > sample_size:
        idx = np.sort(np.random.choice(len(y_test), sample_size, replace=False))
    else:
        idx = np.arange(len(y_test))
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual values.
    plt.plot(range(len(idx)), y_test[idx], 'ro', markersize=5, label='Actual')
    
    # Plot predicted values.
    plt.plot(range(len(idx)), y_pred[idx], 'bo', markersize=3, label='Predicted')
    
    # Plot a 90% prediction interval (using 0.05 and 0.95 quantiles).
    try:
        lower_q = np.where(y_quantiles.columns == 0.05)[0][0]
        upper_q = np.where(y_quantiles.columns == 0.95)[0][0]
        plt.fill_between(
            range(len(idx)),
            y_quantiles.iloc[idx, lower_q],
            y_quantiles.iloc[idx, upper_q],
            alpha=0.3,
            color='skyblue',
            label='90% Prediction Interval'
        )
    except Exception as e:
        print("Error plotting quantile intervals:", e)
    
    plt.title('Predictions with Conformal Uncertainty')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def add_conformal_to_exhaustion_model(train_data, test_data, target_col="by_trial_exhaustion_score"):
    """
    Example integration to add conformal prediction to the exhaustion model.

    Parameters
    ----------
    train_data : pandas.DataFrame
        The training dataset, must include all features listed below plus the target column.
    test_data : pandas.DataFrame
        The test dataset, must include all features listed below plus the target column.

    Returns
    -------
    conformal_model : object
        The fitted conformal predictor.
    X_test : pandas.DataFrame
        The test‐set features used for evaluation.
    y_test : pandas.Series
        The true target values for the test set.
    y_pred : numpy.ndarray
        The point predictions on the test set.
    y_quantiles : numpy.ndarray
        The predicted lower and upper quantiles for each test instance.
    y_interval : numpy.ndarray
        The full prediction intervals for each test instance.
    """
    # Define features for the exhaustion model (can be adjusted as needed).
    features_exhaustion = [
        'joint_energy', 'joint_power', 'energy_acceleration',
        'elbow_asymmetry', 'hip_asymmetry', 'wrist_asymmetry', 'knee_asymmetry',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_SHOULDER_ROM', 'R_SHOULDER_ROM',
        'exhaustion_lag1', 'power_avg_5', 'rolling_hr_mean'
    ]

    # Split into X/y for train and test
    X_train = train_data[features_exhaustion]
    y_train = train_data[target_col]

    # Train the conformal model using the provided train/test data.
    conformal_model, X_test, y_test = train_conformal_model(
        train_data, test_data,
        features=features_exhaustion,
        target_col=target_col
    )

    # Get predictions with uncertainty.
    y_pred, y_quantiles, y_interval = predict_with_uncertainty(
        conformal_model, X_test, y_test
    )

    # Visualize the results.
    plot_conformal_results(X_test, y_test, y_pred, y_quantiles, y_interval)

    return conformal_model


def add_time_series_forecasting(
    data: pd.DataFrame,
    time_col: str = 'timestamp',
    value_cols: list = ['by_trial_exhaustion_score']
):
    """
    Parameters
    ----------
    data : pandas.DataFrame
        The full DataFrame with your time series data.
    time_col : str
        Column name for timestamps.
    value_cols : list of str
        Column(s) for target series.
    """
    import pandas as pd
    from darts import TimeSeries
    from conformal_tights import ConformalCoherentQuantileRegressor, DartsForecaster
    from xgboost import XGBRegressor
    
    features_exhaustion = [
        'joint_energy', 'joint_power', 'energy_acceleration',
        'elbow_asymmetry', 'hip_asymmetry', 'wrist_asymmetry', 'knee_asymmetry',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_SHOULDER_ROM', 'R_SHOULDER_ROM',
        'exhaustion_lag1', 'power_avg_5', 'rolling_hr_mean'
    ]
    # Ensure timestamp column exists
    if time_col not in data.columns:
        raise ValueError(f"Timestamp column '{time_col}' not found in DataFrame.")

    # Instantiate predictor
    xgb_regressor = XGBRegressor(objective="reg:squarederror")
    conformal_predictor = ConformalCoherentQuantileRegressor(estimator=xgb_regressor)

    # Build TimeSeries objects
    target_series = TimeSeries.from_dataframe(data, time_col=time_col, value_cols=value_cols)
    covariates    = TimeSeries.from_dataframe(data, time_col=time_col, value_cols=features_exhaustion)

    # Split 80/20
    train_cutoff   = int(0.8 * len(data))
    train_target   = target_series[:train_cutoff]
    test_target    = target_series[train_cutoff:]
    train_covariates = covariates[:train_cutoff]
    test_covariates  = covariates[train_cutoff:]

    # Fit forecaster
    forecaster = DartsForecaster(model=conformal_predictor, lags=24, lags_future_covariates=[0])
    forecaster.fit(train_target, future_covariates=train_covariates)

    # Predict
    forecast = forecaster.predict(
        n=len(test_target),
        future_covariates=test_covariates,
        num_samples=500,
        quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    )

    return forecaster, forecast, test_target


# ==================== MODEL SUMMARY FUNCTIONS ====================

# --- Main Script: Running Three Separate Analyses ---
if __name__ == "__main__":
    import os
    from pathlib import Path
    from ml.load_and_prepare_data.load_data_and_analyze import (
        load_data, prepare_joint_features, feature_engineering, summarize_data, check_and_drop_nulls
    )
    debug = True
    importance_threshold = 0.01
    csv_path = "../../data/processed/final_granular_dataset.csv"
    json_path = "../../data/basketball/freethrow/participant_information.json"
    output_dir = "../../data/Deep_Learning_Final"
    
    base_feature_dir = os.path.join(output_dir, "feature_lists/base")
    trial_feature_dir = os.path.join(output_dir, "feature_lists/trial_summary")
    shot_feature_dir = os.path.join(output_dir, "feature_lists/shot_phase_summary")
    
    data, trial_df, shot_df = prepare_base_datasets(csv_path, json_path, debug=debug)
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    summary_targets = ['exhaustion_rate', 'injury_risk']
    trial_summary_features = [col for col in trial_df.columns if col not in summary_targets]
    trial_summary_features = [col for col in trial_summary_features if col in numeric_features]
    shot_summary_features = [col for col in shot_df.columns if col not in summary_targets]
    shot_summary_features = [col for col in shot_summary_features if col in numeric_features]
    # ========================================
    # 1) Overall Base Dataset (including Joint-Specific Targets)
    # ========================================
    features = [
        'joint_energy', 'joint_power', 'energy_acceleration',
        'elbow_asymmetry', 'hip_asymmetry', 'ankle_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 
        '1stfinger_asymmetry', '5thfinger_asymmetry',
        'elbow_power_ratio', 'hip_power_ratio', 'ankle_power_ratio', 'wrist_power_ratio', 
        'knee_power_ratio', '1stfinger_power_ratio', '5thfinger_power_ratio',
        'L_KNEE_ROM', 'L_KNEE_ROM_deviation', 'L_KNEE_ROM_extreme',
        'R_KNEE_ROM', 'R_KNEE_ROM_deviation', 'R_KNEE_ROM_extreme',
        'L_SHOULDER_ROM', 'L_SHOULDER_ROM_deviation', 'L_SHOULDER_ROM_extreme',
        'R_SHOULDER_ROM', 'R_SHOULDER_ROM_deviation', 'R_SHOULDER_ROM_extreme',
        'L_HIP_ROM', 'L_HIP_ROM_deviation', 'L_HIP_ROM_extreme',
        'R_HIP_ROM', 'R_HIP_ROM_deviation', 'R_HIP_ROM_extreme',
        'L_ANKLE_ROM', 'L_ANKLE_ROM_deviation', 'L_ANKLE_ROM_extreme',
        'R_ANKLE_ROM', 'R_ANKLE_ROM_deviation', 'R_ANKLE_ROM_extreme',
        'exhaustion_lag1', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'time_since_start', 'ema_exhaustion', 'rolling_exhaustion', 'rolling_energy_std',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ]
    base_targets = ['exhaustion_rate', 'injury_risk']
    joints = ['ANKLE', 'WRIST', 'ELBOW', 'KNEE', 'HIP']
    joint_injury_targets = [f"{side}_{joint}_injury_risk" for joint in joints for side in ['L', 'R']]
    joint_exhaustion_targets = [f"{side}_{joint}_exhaustion_rate" for joint in joints for side in ['L', 'R']]
    joint_targets = joint_injury_targets + joint_exhaustion_targets
    all_targets = base_targets + joint_targets

    logging.info("=== Base Dataset Analysis (Overall + Joint-Specific) ===")
    base_loaded_features = run_feature_import_and_load_top_features(
        dataset=data,
        features=features,
        targets=all_targets,
        base_output_dir=output_dir,
        output_subdir="feature_lists/base",
        debug=debug,
        dataset_label="Base Data",
        importance_threshold=importance_threshold,
        n_top=10,
        run_analysis=False
    )
    print(f"Base Loaded Features: {base_loaded_features}")
    
    joint_feature_dict = {}
    for target in joint_targets:
        try:
            feat_loaded = base_loaded_features.get(target, [])
            logging.info(f"Test Load: Features for {target}: {feat_loaded}")
            joint_feature_dict[target] = feat_loaded
        except Exception as e:
            logging.error(f"Error loading features for {target}: {e}")
    
    # ========================================
    # 2) Trial Summary Dataset Analysis
    # ========================================

    
    logging.info("=== Trial Summary Dataset Analysis ===")
    trial_loaded_features = run_feature_import_and_load_top_features(
        dataset=trial_df,
        features=trial_summary_features,
        targets=summary_targets,
        base_output_dir=output_dir,
        output_subdir="feature_lists/trial_summary",
        debug=debug,
        dataset_label="Trial Summary Data",
        importance_threshold=importance_threshold,
        n_top=10,
        run_analysis=False
    )
    features_exhaustion_trial = trial_loaded_features.get('exhaustion_rate', [])
    features_injury_trial = trial_loaded_features.get('injury_risk', [])
    trial_summary_data = trial_df.copy()
    
    # ========================================
    # 3) Shot Phase Summary Dataset Analysis
    # ========================================

    
    logging.info("=== Shot Phase Summary Dataset Analysis ===")
    shot_loaded_features = run_feature_import_and_load_top_features(
        dataset=shot_df,
        features=shot_summary_features,
        targets=summary_targets,
        base_output_dir=output_dir,
        output_subdir="feature_lists/shot_phase_summary",
        debug=debug,
        dataset_label="Shot Phase Summary Data",
        importance_threshold=importance_threshold,
        n_top=10,
        run_analysis=False
    )
    features_exhaustion_shot = shot_loaded_features.get('exhaustion_rate', [])
    features_injury_shot = shot_loaded_features.get('injury_risk', [])
    shot_phase_summary_data = shot_df.copy()
    
    # ------------------------------
    # 5. Split Base Data for Training Models
    # ------------------------------
    train_data, test_data = temporal_train_test_split(data, test_size=0.2)
    timesteps = 5

    # Hyperparameters and architecture definitions.
    hyperparams = {
        "epochs": 10,
        "batch_size": 32,
        "early_stop_patience": 5
    }
    arch_exhaustion = {
        "num_lstm_layers": 1,
        "lstm_units": 64,
        "dropout_rate": 0.2,
        "dense_units": 1,
        "dense_activation": None
    }
    arch_injury = {
        "num_lstm_layers": 1,
        "lstm_units": 64,
        "dropout_rate": 0.2,
        "dense_units": 1,
        "dense_activation": "sigmoid"
    }
    
    # For demonstration, define features/targets (you can adjust these as needed)
    features_exhaustion = [
        'joint_power', 
        'joint_energy', 
        'elbow_asymmetry',  
        'L_WRIST_angle', 'R_WRIST_angle',  # Updated: removed "wrist_angle"
        'exhaustion_lag1', 
        'power_avg_5',
        'simulated_HR',
        'player_height_in_meters',
        'player_weight__in_kg'
    ]
    target_exhaustion = 'by_trial_exhaustion_score'

    features_injury = [
        'joint_power', 
        'joint_energy', 
        'elbow_asymmetry',  
        'knee_asymmetry', 
        'L_WRIST_angle', 'R_WRIST_angle',  # Updated: removed "wrist_angle"
        'exhaustion_lag1', 
        'power_avg_5',
        'simulated_HR',
        'player_height_in_meters',
        'player_weight__in_kg'
    ]
    target_injury = 'injury_risk'
    # ------------------------------
    # 6a. Train Models on Base Data for Overall Exhaustion and Injury Risk
    # ------------------------------
    model_exhaustion, scaler_exhaustion, target_scaler, X_val_exh, y_val_exh = train_exhaustion_model(
        train_data, test_data, features_exhaustion, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        early_stop_patience=hyperparams["early_stop_patience"],
        num_lstm_layers=arch_exhaustion["num_lstm_layers"],
        lstm_units=arch_exhaustion["lstm_units"],
        dropout_rate=arch_exhaustion["dropout_rate"],
        dense_units=arch_exhaustion["dense_units"],
        dense_activation=arch_exhaustion["dense_activation"]
    )
    model_injury, scaler_injury, X_val_injury, y_val_injury = train_injury_model(
        train_data, test_data, features_injury, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        num_lstm_layers=arch_injury["num_lstm_layers"],
        lstm_units=arch_injury["lstm_units"],
        dropout_rate=arch_injury["dropout_rate"],
        dense_units=arch_injury["dense_units"],
        dense_activation=arch_injury["dense_activation"]
    )
    # For joint models, train using the base data and the corresponding feature lists.
    joint_models = {}
    for joint_target, features_list in joint_feature_dict.items():
        try:
            logging.info(f"Training joint-specific injury model for {joint_target} using features: {features_list}")
            model, scaler, X_val_joint, y_val_joint = train_injury_model(
                train_data, test_data,
                features=features_list,
                timesteps=timesteps, 
                epochs=hyperparams["epochs"],
                batch_size=hyperparams["batch_size"],
                early_stop_patience=hyperparams["early_stop_patience"],
                num_lstm_layers=arch_exhaustion["num_lstm_layers"],
                lstm_units=arch_exhaustion["lstm_units"],
                dropout_rate=arch_exhaustion["dropout_rate"],
                dense_units=arch_exhaustion["dense_units"],
                dense_activation=arch_exhaustion["dense_activation"],
                target_col=joint_target
            )
            joint_models[joint_target] = {
                'model': model,
                'features': features_list,
                'scaler': scaler
            }
            logging.info(f"Successfully trained joint model for {joint_target}.")
        except Exception as e:
            logging.error(f"Error training joint model for {joint_target}: {e}")

    # 6c. Train joint exhaustion models (by_trial_exhaustion) for each joint.
    joint_exhaustion_models = {}
    for joint in joints:
        for side in ['L', 'R']:
            target_joint_exh = f"{side}_{joint}_energy_by_trial_exhaustion_score"
            try:
                if target_joint_exh in joint_feature_dict:
                    features_list = joint_feature_dict[target_joint_exh]
                    logging.info(f"Using preloaded features for {target_joint_exh}: {features_list}")
                else:
                    features_list = load_top_features(target_joint_exh, feature_dir=base_feature_dir, df=data, n_top=10)
                    logging.info(f"Loaded features for {target_joint_exh}: {features_list}")
                
                model_exh, scaler_exh, target_scaler_exh, X_val_joint_exh, y_val_joint_exh = train_exhaustion_model(
                    train_data, test_data,
                    features=features_list,
                    timesteps=timesteps,
                    epochs=hyperparams["epochs"],
                    batch_size=hyperparams["batch_size"],
                    early_stop_patience=hyperparams["early_stop_patience"],
                    num_lstm_layers=arch_exhaustion["num_lstm_layers"],
                    lstm_units=arch_exhaustion["lstm_units"],
                    dropout_rate=arch_exhaustion["dropout_rate"],
                    dense_units=arch_exhaustion["dense_units"],
                    dense_activation=arch_exhaustion["dense_activation"],
                    target_col=target_joint_exh
                )
                joint_exhaustion_models[target_joint_exh] = {
                    'model': model_exh,
                    'features': features_list,
                    'scaler': scaler_exh
                }
                logging.info(f"Successfully trained joint exhaustion model for {target_joint_exh}.")
            except Exception as e:
                logging.error(f"Error training joint exhaustion model for {target_joint_exh}: {e}")

    # ------------------------------
    # 7a. Forecasting for Base Models
    # ------------------------------
    forecast_and_plot_exhaustion(
        model=model_exhaustion,
        test_data=test_data,
        forecast_features=features_exhaustion,
        scaler_exhaustion=scaler_exhaustion,
        target_scaler=target_scaler,
        timesteps=timesteps,
        future_steps=50,
        title="Overall Exhaustion Model Forecast"
    )
    forecast_and_plot_injury(
        model=model_injury,
        test_data=test_data,
        forecast_features=features_injury,
        scaler_injury=scaler_injury,
        timesteps=timesteps,
        future_steps=50,
        title="Overall Injury Risk Forecast"
    )
    forecast_and_plot_joint(
        joint_models=joint_models,
        test_data=test_data,
        timesteps=timesteps,
        future_steps=50
    )

    # ------------------------------
    # 8. Summarize Base Model Testing Results
    # ------------------------------
    summary_df = summarize_all_models(
        model_exhaustion, X_val_exh, y_val_exh, target_scaler,
        model_injury, X_val_injury, y_val_injury,
        joint_models, test_data, timesteps, output_dir
    )
    print("=== Model Summaries (Base Data) ===")
    print(summary_df)


    # ------------------------------
    # Additional: Conformal Uncertainty Integration
    # ------------------------------
    logging.info("Running conformal uncertainty integration for exhaustion model...")
    conformal_model = add_conformal_to_exhaustion_model(train_data, test_data, target_col='by_trial_exhaustion_score',)
    
    try:
        # Time series forecasting with uncertainty using Darts.
        logging.info("Running time series forecasting with conformal uncertainty...")
        forecaster, forecast, test_target = add_time_series_forecasting(data
                                                                        , time_col='timestamp'
                                                                        , value_cols=['by_trial_exhaustion_score'])
        # Plot forecast vs. actual (using Darts built-in plotting).
        forecast.plot(label="Forecast")
        test_target.plot(label="Actual")
        plt.title("Time Series Forecast with Conformal Uncertainty")
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error("An error occurred in time series forecasting: " + str(e))
        sys.exit(1)

    # ------------------------------
    # 9. Train, Forecast, and Summarize Aggregated Models
    # ------------------------------
    # Instead of using a hard-coded summary_features list, we now load the top features
    # specific to each aggregated dataset (which were saved using the threshold filter).
    
    # --- 9a. Process Trial Summary Data ---
    trial_train_data, trial_test_data = temporal_train_test_split(trial_summary_data, test_size=0.2)
    

    model_exhaustion_trial, scaler_exhaustion_trial, target_scaler_trial, X_val_exh_trial, y_val_exh_trial = train_exhaustion_model(
        trial_train_data, trial_test_data, features_exhaustion_trial, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        early_stop_patience=hyperparams["early_stop_patience"],
        num_lstm_layers=arch_exhaustion["num_lstm_layers"],
        lstm_units=arch_exhaustion["lstm_units"],
        dropout_rate=arch_exhaustion["dropout_rate"],
        dense_units=arch_exhaustion["dense_units"],
        dense_activation=arch_exhaustion["dense_activation"]
    )
    model_injury_trial, scaler_injury_trial, X_val_injury_trial, y_val_injury_trial = train_injury_model(
        trial_train_data, trial_test_data, features_injury_trial, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        num_lstm_layers=arch_injury["num_lstm_layers"],
        lstm_units=arch_injury["lstm_units"],
        dropout_rate=arch_injury["dropout_rate"],
        dense_units=arch_injury["dense_units"],
        dense_activation=arch_injury["dense_activation"]
    )
    
    forecast_and_plot_exhaustion(
        model=model_exhaustion_trial,
        test_data=trial_test_data,
        forecast_features=features_exhaustion_trial,
        scaler_exhaustion=scaler_exhaustion_trial,
        target_scaler=target_scaler_trial,
        timesteps=timesteps,
        future_steps=50,
        title="Trial Summary Aggregated Exhaustion Forecast"
    )
    forecast_and_plot_injury(
        model=model_injury_trial,
        test_data=trial_test_data,
        forecast_features=features_injury_trial,
        scaler_injury=scaler_injury_trial,
        timesteps=timesteps,
        future_steps=50,
        title="Trial Summary Aggregated Injury Forecast"
    )
    
    trial_summary_df = summarize_all_models(
        model_exhaustion_trial, X_val_exh_trial, y_val_exh_trial, target_scaler_trial,
        model_injury_trial, X_val_injury_trial, y_val_injury_trial,
        joint_models, trial_test_data, timesteps, output_dir,
        include_joint_models=False, debug=debug
    )

    print("=== Model Summaries (Trial Summary Aggregated Data) ===")
    print(trial_summary_df)
    
    # --- 9b. Process Shot Phase Summary Data ---
    shot_train_data, shot_test_data = temporal_train_test_split(shot_phase_summary_data, test_size=0.2)


    model_exhaustion_shot, scaler_exhaustion_shot, target_scaler_shot, X_val_exh_shot, y_val_exh_shot = train_exhaustion_model(
        shot_train_data, shot_test_data, features_exhaustion_shot, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        early_stop_patience=hyperparams["early_stop_patience"],
        num_lstm_layers=arch_exhaustion["num_lstm_layers"],
        lstm_units=arch_exhaustion["lstm_units"],
        dropout_rate=arch_exhaustion["dropout_rate"],
        dense_units=arch_exhaustion["dense_units"],
        dense_activation=arch_exhaustion["dense_activation"]
    )
    model_injury_shot, scaler_injury_shot, X_val_injury_shot, y_val_injury_shot = train_injury_model(
        shot_train_data, shot_test_data, features_injury_shot, timesteps,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        num_lstm_layers=arch_injury["num_lstm_layers"],
        lstm_units=arch_injury["lstm_units"],
        dropout_rate=arch_injury["dropout_rate"],
        dense_units=arch_injury["dense_units"],
        dense_activation=arch_injury["dense_activation"]
    )
    
    forecast_and_plot_exhaustion(
        model=model_exhaustion_shot,
        test_data=shot_test_data,
        forecast_features=features_exhaustion_shot,
        scaler_exhaustion=scaler_exhaustion_shot,
        target_scaler=target_scaler_shot,
        timesteps=timesteps,
        future_steps=50,
        title="Shot Phase Summary Aggregated Exhaustion Forecast"
    )
    forecast_and_plot_injury(
        model=model_injury_shot,
        test_data=shot_test_data,
        forecast_features=features_injury_shot,
        scaler_injury=scaler_injury_shot,
        timesteps=timesteps,
        future_steps=50,
        title="Shot Phase Summary Aggregated Injury Forecast"
    )
    
    shot_summary_df = summarize_all_models(
        model_exhaustion_shot, X_val_exh_shot, y_val_exh_shot, target_scaler_shot,
        model_injury_shot, X_val_injury_shot, y_val_injury_shot,
        joint_models, shot_test_data, timesteps, output_dir,
        include_joint_models=False, debug=debug
    )

    print("=== Model Summaries (Shot Phase Summary Aggregated Data) ===")
    print(shot_summary_df)


    # ------------------------------
    # Final Step: Group and Compare Summaries Across Datasets
    # ------------------------------

    # Separate base summary into regression and classification parts.
    base_reg = summary_df[summary_df["Type"] == "Regression"]
    base_class = summary_df[summary_df["Type"] == "Classification"]

    trial_reg = trial_summary_df[trial_summary_df["Type"] == "Regression"]
    trial_class = trial_summary_df[trial_summary_df["Type"] == "Classification"]

    shot_reg = shot_summary_df[shot_summary_df["Type"] == "Regression"]
    shot_class = shot_summary_df[shot_summary_df["Type"] == "Classification"]

    # Generate joint injury summary from the base test data.
    joint_injury_dict = summarize_joint_models(joint_models, test_data, timesteps, debug=debug)
    joint_injury_df = pd.DataFrame.from_dict(joint_injury_dict, orient='index').reset_index().rename(columns={'index': 'Model'})
    joint_injury_df["Type"] = "Classification"

    # Generate joint exhaustion summary from the base test data.
    joint_exh_dict = summarize_joint_exhaustion_models(joint_exhaustion_models, test_data, timesteps, debug=debug)
    joint_exh_df = pd.DataFrame.from_dict(joint_exh_dict, orient='index').reset_index().rename(columns={'index': 'Model'})
    joint_exh_df["Type"] = "Regression"
    

    # Build lists for each group.
    regression_summaries = [base_reg, trial_reg, shot_reg]
    classification_summaries = [base_class, trial_class, shot_class]
    # Combine both joint summaries into one list.
    joint_summaries = [joint_injury_df, joint_exh_df]

    # Provide names for each dataset.
    dataset_names = ["Base", "Trial Aggregated", "Shot Aggregated"]
    # For joint models, you may label them as "Joint Injury" and "Joint Exhaustion".
    joint_names = ["Joint Injury Models", "Joint Exhaustion Models"]

    # Get the final combined summaries (including joint summaries).
    final_reg, final_class, final_joint, final_all = final_model_summary(
        regression_summaries, classification_summaries, 
        regression_names=dataset_names, classification_names=dataset_names,
        joint_summaries=joint_summaries, joint_names=joint_names
    )

    print("=== Final Regression Summary ===")
    print(final_reg)
    print("\n=== Final Classification Summary ===")
    print(final_class)
    print("\n=== Final Joint Summary ===")
    print(final_joint)
    print("\n=== Final Combined Summary ===")
    print(final_all) 
