

import torch
import logging
import pandas as pd
import sys
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(device_str)
from darts.models import NBEATSModel, ExponentialSmoothing
from darts import TimeSeries
from darts.dataprocessing import Pipeline
forecast_horizon = 36  # Set your forecast horizon as needed
model_nbeats = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=forecast_horizon,
    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
)
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
from ml.preprocess_train_predict.conformal_tights import (
    train_conformal_model, predict_with_uncertainty, plot_conformal_results, add_time_series_forecasting
    )


def preprocess_timeseries_darts(ts, transformers=None):
    """
    Preprocesses a Darts TimeSeries object using a pipeline of transformers from Darts.
    
    Parameters:
        ts (TimeSeries): Darts TimeSeries object to be transformed.
        transformers (list): Optional list of transformer objects. If None, a default pipeline
                             using MissingValuesFiller and Scaler is used.
                             
    Returns:
        TimeSeries: The transformed TimeSeries.
    """
    from darts.dataprocessing import Pipeline
    from darts.dataprocessing.transformers import MissingValuesFiller, Scaler


    # Use default transformers if none are provided.
    if transformers is None:
        transformers = [MissingValuesFiller(), Scaler()]

    # Create and fit the transformation pipeline.
    pipeline = Pipeline(transformers)
    ts_transformed = pipeline.fit_transform(ts)
    return ts_transformed



def detect_anomalies_with_darts(time_series, training_window=24, high_quantile=0.99, k=2, window=5):
    """
    Detects anomalies in a Darts TimeSeries using forecasting-based anomaly detection.
    
    This function utilizes the KMeansScorer and QuantileDetector from Darts AD module
    to compute anomaly scores and then convert them into binary anomaly flags.
    
    Parameters:
      - time_series (TimeSeries): Darts TimeSeries object containing the data.
      - training_window (int): Window size for training the anomaly scorer.
      - high_quantile (float): High quantile threshold for binary detection.
      - k (int): Number of clusters for KMeansScorer.
      - window (int): Rolling window size for anomaly scoring.
    
    Returns:
      - binary_anomalies (np.array): Array of binary anomaly flags (1 for anomaly, 0 for normal).
      - anomaly_scores (np.array): Anomaly scores computed for the time series.
    """
    from darts.ad import KMeansScorer, QuantileDetector

    # Use 80% of the series to train the anomaly scorer
    train_length = int(0.8 * len(time_series))
    training_data = time_series[:train_length]
    new_data = time_series[train_length:]
    
    # Train the anomaly scorer on the training data
    scorer = KMeansScorer(k=k, window=window)
    scorer.fit(training_data)
    
    # Score new data for anomalies.
    anomaly_scores = scorer.score(new_data)
    
    # Fit a detector on the training scores and detect anomalies on new data.
    detector = QuantileDetector(high_quantile=high_quantile)
    detector.fit(scorer.score(training_data))
    binary_anomalies = detector.detect(anomaly_scores)
    
    return binary_anomalies, anomaly_scores


from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.models import NBEATSModel, ExponentialSmoothing
from darts.metrics import mae, mape, rmse, smape
import pandas as pd

def enhanced_forecasting_with_darts_and_metrics(
    data,
    timestamp_col='timestamp',
    target_col='exhaustion_rate',
    train_frac=0.8,
    freq='33ms'
):
    """
    Splits the series, fits two Darts models on train, forecasts on test,
    aligns timestamps, and returns forecasts + test_series + metrics_df.
    
    Parameters:
        data (pd.DataFrame): Input data with timestamp and target columns.
        timestamp_col (str): Name of the timestamp column.
        target_col (str): Name of the target column to forecast.
        train_frac (float): Fraction of data to use for training.
        freq (str): Frequency string for timestamps.
    
    Returns:
        train_series (TimeSeries): Training portion of the time series.
        test_series (TimeSeries): Testing portion of the time series.
        forecast_nbeats (TimeSeries): Forecast from the NBEATS model.
        forecast_es (TimeSeries): Forecast from the Exponential Smoothing model.
        metrics_df (pd.DataFrame): DataFrame containing metrics for both models.
    """
    from darts import TimeSeries
    from darts.dataprocessing import Pipeline
    from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
    from darts.models import NBEATSModel, ExponentialSmoothing
    from darts.metrics import mae, mape, rmse, smape
    import pandas as pd
    import logging

    # Build and preprocess full series
    full_series = TimeSeries.from_dataframe(
        data, time_col=timestamp_col, value_cols=[target_col],
        fill_missing_dates=True, freq=freq
    )
    pipeline = Pipeline([MissingValuesFiller(), Scaler()])
    full_series = pipeline.fit_transform(full_series)

    # Split
    split_idx    = int(train_frac * len(full_series))
    train_series = full_series[:split_idx]
    test_series  = full_series[split_idx:]

    # Instantiate & fit
    model_nbeats = NBEATSModel(input_chunk_length=24, output_chunk_length=len(test_series))
    model_es     = ExponentialSmoothing()
    
    logging.info(f"Fitting NBEATS model for {target_col}...")
    model_nbeats.fit(train_series)
    
    logging.info(f"Fitting ExponentialSmoothing model for {target_col}...")
    model_es.fit(train_series)

    # Forecast exactly test length
    logging.info("Generating forecasts...")
    forecast_nbeats = model_nbeats.predict(n=len(test_series))
    forecast_es     = model_es.predict(n=len(test_series))

    # Compute metrics
    logging.info("Computing metrics...")
    metrics = {
        "NBEATS": {
            "MAE":   mae(test_series, forecast_nbeats),
            "RMSE":  rmse(test_series, forecast_nbeats),
            "SMAPE": smape(test_series, forecast_nbeats),
        },
        "ExpSmoothing": {
            "MAE":   mae(test_series, forecast_es),
            "RMSE":  rmse(test_series, forecast_es),
            "SMAPE": smape(test_series, forecast_es),
        }
    }
    metrics_df = pd.DataFrame(metrics).T
    
    logging.info(f"Darts metrics for {target_col}:\n{metrics_df}")
    
    return train_series, test_series, forecast_nbeats, model_nbeats, forecast_es, metrics_df







    

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
        "epochs": 100,
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
    # For joint models, we train using the base data and the corresponding features saved earlier.
    joints = ['ANKLE', 'WRIST', 'ELBOW', 'KNEE', 'HIP']
    # Now train joint models by passing the preloaded joint feature dictionary.
    # Assume joint_feature_dict is already built (or loaded) mapping each joint target to its feature list.
    # For example:
    # joint_feature_dict = {
    #    "L_ANKLE_injury_risk": [list of features],
    #    "R_ANKLE_injury_risk": [list of features],
    #    "L_WRIST_injury_risk": [list of features],
    #    ... etc.
    # }
    # 6b. Train Models on Base Data for individual joint Exhaustion and Injury Risk
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
                dense_activation=arch_exhaustion["dense_activation"],  # Use the same activation as for injury models, typically sigmoid for binary classification
                target_col=joint_target  # use the joint-specific target
            )
            joint_models[joint_target] = {
                'model': model,
                'features': features_list,
                'scaler': scaler
            }
            logging.info(f"Successfully trained joint model for {joint_target}.")
        except Exception as e:
            logging.error(f"Error training joint model for {joint_target}: {e}")



    # 6c. Train Models on Base Data for individual joint Exhaustion (by_trial_exhaustion) 
    #    (in addition to the injury models)
    joint_exhaustion_models = {}
    for joint in joints:
        for side in ['L', 'R']:
            target_joint_exh = f"{side}_{joint}_energy_by_trial_exhaustion_score"
            try:
                # Try to use the preloaded feature list if available; otherwise, load it.
                if target_joint_exh in joint_feature_dict:
                    features_list = joint_feature_dict[target_joint_exh]
                    logging.info(f"Using preloaded features for {target_joint_exh}: {features_list}")
                else:
                    features_list = load_top_features(target_joint_exh, feature_dir=base_feature_dir, df=data, n_top=10)
                    logging.info(f"Loaded features for {target_joint_exh}: {features_list}")
                
                # Train a regression model for the joint-specific exhaustion score.
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
    # 6. Base Model Forecasting
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
        joint_models=joint_models,  # Replace with your actual joint models dictionary
        test_data=test_data,
        timesteps=timesteps,
        future_steps=50
    )


    # ------------------------------
    # 7. New Darts-based Preprocessing and Forecasting Examples
    # ------------------------------
    # Ensure a valid 'timestamp' column exists.
    # if 'timestamp' not in data.columns:
    #     data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='h')


    # --- A. Preprocess TimeSeries with Darts ---
    from darts import TimeSeries
    # Create a TimeSeries from the exhaustion target column.
    ts_exhaustion = TimeSeries.from_dataframe(
    data,
    time_col='timestamp',
    value_cols=['exhaustion_rate'],
    fill_missing_dates=True,
    freq='33ms'
)

    # Preprocess the TimeSeries using the Darts pipeline.
    ts_preprocessed = preprocess_timeseries_darts(ts_exhaustion)
    print("Preprocessed TimeSeries head:")
    print(ts_preprocessed.to_dataframe().head())

    # --- B. Detect Anomalies with Darts ---
    anomaly_flags, anomaly_scores = detect_anomalies_with_darts(ts_preprocessed)
    print("Anomaly flags (first 10 values):", anomaly_flags[:10])
    print("Anomaly scores (first 10 values):", anomaly_scores[:10])

    # --- C. Enhanced Forecasting with Darts ---
    train_series, test_series, forecast_nbeats, model_nbeats, forecast_es, metrics_df = enhanced_forecasting_with_darts_and_metrics(
        data, timestamp_col='timestamp', 
        target_col='exhaustion_rate')

    # And update the plotting code:
    plt.figure(figsize=(12, 6))
    test_series.plot(label="Actual")
    forecast_nbeats.plot(label="NBEATS Forecast")
    forecast_es.plot(label="ExpSmoothing Forecast")
    plt.title("Enhanced Forecasting with Darts")
    plt.legend()
    plt.show()

    # Optionally, print the metrics:
    print("\nForecast Metrics:")
    print(metrics_df)


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


