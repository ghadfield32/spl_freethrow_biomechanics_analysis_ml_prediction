

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from datapreprocessor import DataPreprocessor
from darts import TimeSeries
from ml.load_and_prepare_data.load_data_and_analyze import (
    load_data, prepare_joint_features, feature_engineering, summarize_data)

from ml.feature_selection.feature_selection import (
    load_top_features, perform_feature_importance_analysis, save_top_features,
    analyze_joint_injury_features, check_for_invalid_values,
    perform_feature_importance_analysis, analyze_and_display_top_features)

from ml.preprocess_train_predict.base_training import (
    temporal_train_test_split, scale_features, create_sequences, train_exhaustion_model, 
    train_injury_model,  train_joint_models, forecast_and_plot_exhaustion, forecast_and_plot_injury,
    forecast_and_plot_joint, summarize_regression_model, summarize_classification_model, 
    summarize_joint_models, summarize_all_models, final_model_summary, 
    summarize_joint_exhaustion_models
    )


def evaluate_model_metrics(y_true, y_pred):
    """
    Evaluate model predictions using common metrics: MAE, RMSE, and R².
    
    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: A dictionary with keys 'mae', 'rmse', and 'r2' representing the evaluation metrics.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # Ensure dimensions are compatible using the existing helper function.
    y_true, y_pred = ensure_compatible_dimensions(y_true, y_pred)
    
    # Compute metrics.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def generate_final_report(report_data):
    """
    Generate and print a summary report for model evaluation metrics.

    Args:
        report_data (list of dict): A list where each dictionary contains:
            'test' (str): A descriptive test name,
            'mae' (float): The Mean Absolute Error,
            'rmse' (float): The Root Mean Squared Error,
            'r2' (float): The R² score.
    """
    print("\n\n==== Final Evaluation Report ====")
    for entry in report_data:
        print(f"Test: {entry['test']}")
        print(f"  MAE: {entry['mae']:.4f}")
        print(f"  RMSE: {entry['rmse']:.4f}")
        print(f"  R²: {entry['r2']:.4f}\n")
    print("==== End of Report ====\n")


#---------------------------------------------
# Custom functions for preprocessing
#---------------------------------------------
def debug_datasets(variables, max_sample_rows=5):
    """
    Debug multiple datasets with detailed information.
    
    Args:
        variables (dict): Dictionary of variable_name: variable_value pairs to debug
        max_sample_rows (int, optional): Maximum number of sample rows to display
    """
    print("\n==== DATASET DEBUG INFORMATION ====")
    
    for name, value in variables.items():
        print(f"\n[{name}]:")
        print(f"  Type: {type(value)}")
        
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
        elif isinstance(value, dict):
            print(f"  Length: {len(value)} items")
        elif hasattr(value, '__len__'):
            print(f"  Length: {len(value)}")
        
        # Handle different data types
        if isinstance(value, pd.DataFrame):
            print("\n  Data Sample:")
            print(value.head(max_sample_rows))
            print("\n  Columns:")
            print(value.columns.tolist())
            print("\n  Data Types:")
            print(value.dtypes)
            print(f"\n  Missing Values: {value.isna().sum().sum()} total")
        elif isinstance(value, np.ndarray):
            print("\n  Array Sample:")
            if value.ndim == 1:
                print(value[:min(max_sample_rows, value.shape[0])])
            elif value.ndim == 2:
                print(value[:min(max_sample_rows, value.shape[0]), :min(10, value.shape[1])])
            else:
                print(f"  {value.ndim}-dimensional array (sample not shown)")
            print(f"\n  Data Type: {value.dtype}")
            if np.isnan(value).any():
                print(f"  Warning: Contains {np.isnan(value).sum()} NaN values")
        elif isinstance(value, dict):
            print("\n  Dictionary Keys:")
            print(list(value.keys())[:min(20, len(value))])
            if len(value) > 20:
                print(f"  ... and {len(value) - 20} more keys")
        
        # Add more detailed information for model prediction results
        if name.startswith('result') and isinstance(value, tuple):
            print("\n  Tuple Contents:")
            for i, item in enumerate(value):
                print(f"  Element {i}:")
                print(f"    Type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    Shape: {item.shape}")
                if isinstance(item, np.ndarray) and item.size > 0:
                    print(f"    Sample: {item.flatten()[:min(5, item.size)]}")
    
    print("\n==== END DEBUG INFORMATION ====")

# Example usage:
def debug_preprocessing_result(result, expected_shape=None):
    print(f"Type of result: {type(result)}")
    
    # Create a dictionary to pass to our debug function
    debug_data = {
        'result': result,
        'summary': summary,
        'test_data': test_data,
        'train_data': train_data
    }
    
    if expected_shape:
        debug_data['expected_shape'] = expected_shape
        
    # If result is a tuple, add each component separately
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            debug_data[f'result_element_{i}'] = item
            
    # If we have sequence data, add those too
    if 'y_test_seq' in globals():
        debug_data['y_test_seq'] = y_test_seq
    if 'y_train_seq' in globals():
        debug_data['y_train_seq'] = y_train_seq
        
    debug_datasets(debug_data)
    
    return result

# Updated usage example:
# result = dtw_date_predict.final_preprocessing(new_data, model_input_shape=expected_shape)
# result = debug_preprocessing_result(result, expected_shape)

def select_complete_test_data(full_data, n_trials=2):
    """
    Select a subset of the data that contains complete sequences with all phases.
    
    Args:
        full_data (pd.DataFrame): The complete dataset
        n_trials (int): Number of complete trials to select
        
    Returns:
        pd.DataFrame: A subset containing complete sequences with all phases
    """
    # Get all unique phases in the dataset
    all_phases = full_data['pitch_phase_biomech'].unique()
    print(f"All phases in dataset: {all_phases}")
    
    # Find trials that contain all required phases
    complete_trials = []
    
    # Get unique trial/session combinations
    trial_combinations = full_data[['session_biomech', 'trial_biomech']].drop_duplicates().values
    
    for session, trial in trial_combinations:
        # Get data for this trial
        trial_data = full_data[(full_data['session_biomech'] == session) & 
                              (full_data['trial_biomech'] == trial)]
        
        # Check if this trial has all phases
        trial_phases = set(trial_data['pitch_phase_biomech'].unique())
        
        if len(trial_phases) >= len(all_phases) - 1:  # Allow for one missing phase
            complete_trials.append((session, trial, len(trial_data)))
    
    print(f"Found {len(complete_trials)} trials with complete phase data")
    
    # Sort by data size (descending) and select the top n_trials
    complete_trials.sort(key=lambda x: x[2], reverse=True)
    selected_trials = complete_trials[:n_trials]
    
    # Create a new DataFrame with the selected trials
    test_data = pd.DataFrame()
    for session, trial, _ in selected_trials:
        trial_data = full_data[(full_data['session_biomech'] == session) & 
                              (full_data['trial_biomech'] == trial)]
        print(f"Selected trial {session}/{trial} with {len(trial_data)} samples and phases: {trial_data['pitch_phase_biomech'].unique()}")
        test_data = pd.concat([test_data, trial_data])
    
    return test_data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import os
    import logging
    import yaml
    import shutil
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import os
    from pathlib import Path
    from ml.load_and_prepare_data.load_data_and_analyze import (
        load_data, prepare_joint_features, feature_engineering, summarize_data)
    
    from ml.feature_selection.feature_selection import (
        load_top_features, perform_feature_importance_analysis, save_top_features,
        analyze_joint_injury_features, check_for_invalid_values,
        perform_feature_importance_analysis, analyze_and_display_top_features, 
        run_feature_importance_analysis)
    from ml.preprocess_train_predict.darts_models_for_comparison import (
        preprocess_timeseries_darts, detect_anomalies_with_darts, enhanced_forecasting_with_darts, 
        add_time_series_forecasting, add_datetime_column )

    debug = True
    csv_path = "../../data/processed/final_granular_dataset.csv"
    json_path = "../../data/basketball/freethrow/participant_information.json"
    graphs_output_dir="../../data/Deep_Learning_Final/graphs"
    transformers_dir="../../data/Deep_Learning_Final/transformers"
    
    # Load and process data using imported modules
    data = load_data(csv_path, json_path, debug=debug)
    data = prepare_joint_features(data, debug=debug)
    data = feature_engineering(data, debug=debug)
    ## Add datetime column using our new function
    data = add_datetime_column(data, base_datetime=pd.Timestamp('2025-01-01 00:00:00'), 
                               break_seconds=10, trial_id_col='trial_id', freq='33ms')
    print("Data columns for Darts processing:", data.columns.tolist())
    
    # Nominal/Categorical variables: For example, identifiers or labels (none of these apply here)
    nominal_categorical = ['player_height_in_meters', 'player_weight__in_kg']

    # Ordinal/Categorical variables: Categorical variables with a natural order (none of these apply here)
    ordinal_categorical = []

    # Numerical variables: All of your features are continuous numerical measurements.
    numerical = [
        'joint_energy',
        'joint_power',
        'energy_acceleration',
        'hip_asymmetry',
        'wrist_asymmetry',
        'rolling_power_std',
        'rolling_hr_mean',
        'rolling_energy_std',
        'simulated_HR',
        'player_height_in_meters',
        'player_weight__in_kg'
    ]
    # Set up logging for debugging purposes.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    features = nominal_categorical + ordinal_categorical + numerical
    base_targets = ['by_trial_exhaustion_score', 'injury_risk']
    
    
    # Load your training data
    logger.info(f"Training data loaded from {csv_path}. Shape: {data.shape}")
    
    y_variable = base_targets[0]
    ordinal_categoricals=ordinal_categorical
    nominal_categoricals=nominal_categorical
    numericals=numerical

    
    # Define model building function
    def build_lstm_model(input_shape, horizon=1):
        """
        Build an LSTM model with an output layer that matches the specified horizon.
        
        Args:
            input_shape: Tuple defining the input shape (timesteps, features)
            horizon: Number of future timesteps to predict (output dimension)
            
        Returns:
            A compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(horizon)  # Output dimension now dynamically set by horizon
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def ensure_compatible_dimensions(targets, predictions):
        """
        Ensure that targets and predictions have compatible dimensions for error metric calculation.
        
        This function converts inputs to NumPy arrays, squeezes the last dimension if it is 1
        (to convert a (samples, time_steps, 1) array to (samples, time_steps)), truncates both arrays
        to the minimum number of samples if they differ, and reshapes 1D arrays to 2D if needed.
        
        Args:
            targets (array-like): Ground truth target values.
            predictions (array-like): Predicted values.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The adjusted target and prediction arrays.
        """
        import numpy as np

        # Convert inputs to NumPy arrays
        targets = np.array(targets)
        predictions = np.array(predictions)

        # If targets or predictions have an extra dimension of size 1, squeeze that axis.
        if targets.ndim == 3 and targets.shape[2] == 1:
            targets = targets.squeeze(axis=2)
        if predictions.ndim == 3 and predictions.shape[2] == 1:
            predictions = predictions.squeeze(axis=2)

        # If number of samples (first axis) differ, truncate both arrays to the minimum count.
        if targets.shape[0] != predictions.shape[0]:
            n_samples = min(targets.shape[0], predictions.shape[0])
            targets = targets[:n_samples]
            predictions = predictions[:n_samples]

        # If one array is 1D and the other 2D, reshape the 1D array to 2D.
        if targets.ndim == 1 and predictions.ndim == 2:
            targets = targets.reshape(-1, 1)
        elif predictions.ndim == 1 and targets.ndim == 2:
            predictions = predictions.reshape(-1, 1)

        # Debug print the adjusted shapes
        print(f"Adjusted shapes - targets: {targets.shape}, predictions: {predictions.shape}")

        return targets, predictions



    def get_horizon_from_preprocessor(preprocessor):
        """
        Extract the horizon parameter from the preprocessor.
        
        For DTW or pad modes, if the horizon has not been computed yet,
        it returns the product of horizon_sequence_number and sequence_length.
        Otherwise, it returns the computed horizon.
        """
        if hasattr(preprocessor, 'time_series_sequence_mode') and preprocessor.time_series_sequence_mode in ["dtw", "pad"]:
            if preprocessor.horizon is not None:
                return preprocessor.horizon
            else:
                return preprocessor.horizon_sequence_number * preprocessor.sequence_length
        elif hasattr(preprocessor, 'options') and isinstance(preprocessor.options, dict):
            return preprocessor.options.get('horizon', 1)
        elif hasattr(preprocessor, 'horizon'):
            return preprocessor.horizon
        else:
            return 1  # Default horizon if not specified


    
    # ---------- Test 1: Percentage-based Split ----------
    print("\n\n=== Test 1: Percentage-based Split (80/20) ===")

    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)
    # Calculate the index to split the dataset into thirds
    split_index = int(len(data) * (2 / 3))

    # Set new_data as the last third of the dataset
    new_data = data.iloc[split_index:].copy()
    
    # list columns
    print("New data columns:", new_data.columns.tolist())

    # Debugging information
    print(f"Total dataset size: {len(data)}")
    print(f"Split index (start of last third): {split_index}")
    print(f"New data (last third) shape: {new_data.shape}")

    # Configure the preprocessor for training without explicit window_size, step_size, or horizon parameters.
    preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,  # Horizon value is provided here
            "step_size": 1,  # Step size provided here
            "sequence_modes": {
                "set_window": {
                    "window_size": 10,  # Window size provided here
                    "max_sequence_length": 10
                }
            },
            "ts_sequence_mode": "set_window",
            "split_dataset": {
                "test_size": 0.2,
                "random_state": 42
            },
            "time_series_split": {
                "method": "standard"
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="set_window",
        debug=True
    )

    # Preprocess the training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)


    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    # Train a model
    print("Training LSTM model with percentage-based split...")
    # Extract horizon from preprocessor
    horizon = get_horizon_from_preprocessor(preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    # Build the LSTM model using the extracted horizon
    model1 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)

    model1.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model1.save('./transformers/model_percentage_split.h5')
    
    # Predict using the test set
    predictions = model1.predict(X_test_seq)
    print(f"Predictions shape: {predictions.shape}, Target shape: {y_test_seq.shape}")
    if predictions.shape[-1] != y_test_seq.shape[-1]:
        print(f"WARNING: Shape mismatch detected: predictions {predictions.shape} vs targets {y_test_seq.shape}")
        
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    mae = mean_absolute_error(y_test_seq, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_seq, predictions))
    print(f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    
    # Predict using predict mode
    print("\nTesting prediction mode with new data...")
    
    # Take the last segment of data as "new" data for prediction
    # new_data = data.iloc[-48:].copy()  
    
    # Configure the preprocessor for prediction
    predict_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,              # Number of time steps ahead to predict
            "step_size": 1,             # Step size for moving the window
            "sequence_modes": {         # Window configuration for sequence mode
                "set_window": {
                    "window_size": 10,         # Size of each window
                    "max_sequence_length": 10  # Maximum sequence length
                }
            },
            "ts_sequence_mode": "set_window"
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        transformers_dir="./transformers"
    )

    

    # Make predictions
    model1 = load_model('./transformers/model_percentage_split.h5')
    expected_shape = model1.input_shape
    print(f"Expected model input shape: {expected_shape}")

    # Preprocess new data for prediction
    # results = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = results[0]
    X_new_preprocessed, recommendations, X_inversed = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    
    print(f"Prediction data shape: {X_new_preprocessed.shape}")
    
    predictions = model1.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    print(f"Predictions: {predictions[:5].flatten()}")
    # Compute and print evaluation metrics using the new function.
    model1_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation metrics - MAE: {model1_metrics['mae']:.4f}, RMSE: {model1_metrics['rmse']:.4f}, R²: {model1_metrics['r2']:.4f}")

    
    # ---------- Test 2: Date-based Split ----------
    print("\n\n=== Test 2: Date-based Split (2025-02-14 11:00) ===")
    
    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    new_data = new_data.copy()
    # Configure the preprocessor for training with date-based split
    preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,
            "step_size": 1,
            "sequence_modes": {
                "set_window": {
                    "window_size": 10,  # 1 day window
                    "max_sequence_length": 10
                }
            },
            "ts_sequence_mode": "set_window",
            "split_dataset": {
                "test_size": 0.2,  # Not used for date-based split
                "random_state": 42,
                "time_split_column": "datetime",
                "time_split_value": pd.Timestamp("2025-02-14 11:50:00")
            },
            "time_series_split": {
                "method": "standard"
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="set_window",
        debug=True
    )
    
    # Preprocess the training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)
    
    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    # Train a model
    print("Training LSTM model with date-based split...")
    # Extract horizon from preprocessor
    horizon = get_horizon_from_preprocessor(preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    # Build the LSTM model using the extracted horizon
    model2 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)

    model2.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model2.save('./transformers/model_date_split.h5')
    
    # Test prediction mode
    print("\nTesting prediction mode with new data...")
    
    # Take the last segment of data as "new" data for prediction
    # new_data = data.iloc[-48:].copy()  
    
    # Configure the preprocessor for prediction
    predict_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,              # Number of time steps ahead to predict
            "step_size": 1,             # Step size for moving the window
            "sequence_modes": {         # Window configuration for sequence mode
                "set_window": {
                    "window_size": 10,         # Size of each window
                    "max_sequence_length": 10  # Maximum sequence length
                }
            },
            "ts_sequence_mode": "set_window"
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        transformers_dir="./transformers"
    )

    
    # Preprocess new data for prediction
    # results = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = results[0]
    X_new_preprocessed, recommendations, X_inversed = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    # Make predictions
    model2 = load_model('./transformers/model_date_split.h5')
    expected_shape = model2.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model2.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model2_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation metrics - MAE: {model2_metrics['mae']:.4f}, RMSE: {model2_metrics['rmse']:.4f}, R²: {model2_metrics['r2']:.4f}")

    # ---------- Test 3: PSI-based Split with Feature-Engine ----------
    print("\n\n=== Test 3: PSI-based Split with Feature-Engine ===")
    
    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    new_data = new_data.copy()
    # Configure the preprocessor for training with PSI-based split
    preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,
            "step_size": 1,
            "sequence_modes": {
                "set_window": {
                    "window_size": 10,  # 1 day window
                    "max_sequence_length": 10
                }
            },
            "ts_sequence_mode": "set_window",
            "psi_feature_selection": {
                "enabled": True,
                "threshold": 0.25,
                "split_frac": 0.75,
                "split_distinct": False,
                "apply_before_split": True
            },
            "feature_engine_split": {
                "enabled": True,
                "split_frac": 0.75,
                "split_distinct": False
            },
            "time_series_split": {
                "method": "feature_engine"
            }
        },
        # sequence_categorical=["session_biomech", "trial_biomech"],
        # sub_sequence_categorical=["pitch_phase_biomech"],
        time_series_sequence_mode="set_window",
        debug=True,
        graphs_output_dir="./plots"
    )
    
    # Preprocess the training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)
    
    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    # Visualize PSI results if the method was run
    preprocessor.visualize_psi_results(data, top_n=5)
    
    # Train a model
    print("Training LSTM model with PSI-based split...")
    # Extract horizon from preprocessor
    horizon = get_horizon_from_preprocessor(preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    # Build the LSTM model using the extracted horizon
    model3 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)

    model3.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model3.save('./transformers/model_psi_split.h5')
    
    # Test prediction mode
    print("\nTesting prediction mode with new data...")
    
    # Take the last segment of data as "new" data for prediction
    # new_data = data.iloc[-48:].copy()  
    
    # Configure the preprocessor for prediction
    predict_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "horizon": 10,              # Number of time steps ahead to predict
            "step_size": 1,             # Step size for moving the window
            "sequence_modes": {         # Window configuration for sequence mode
                "set_window": {
                    "window_size": 10,         # Size of each window
                    "max_sequence_length": 10  # Maximum sequence length
                }
            },
            "ts_sequence_mode": "set_window"
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        transformers_dir="./transformers"
    )

    
    # Preprocess new data for prediction
    # results = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = results[0]
    X_new_preprocessed, recommendations, X_inversed = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    # Make predictions
    model3 = load_model('./transformers/model_psi_split.h5')
    expected_shape = model3.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model3.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")

    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model3_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation metrics - MAE: {model3_metrics['mae']:.4f}, RMSE: {model3_metrics['rmse']:.4f}, R²: {model3_metrics['r2']:.4f}")

    # ---------- Test 4: DTW/Pad Mode with PSI-based Split ----------
    print("\n\n=== Test 4: DTW/Pad Mode with PSI-based Split ===")
    
    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    new_data = new_data.copy()
    # Configure the preprocessor for training with DTW mode
    preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "pad": {
                    "pad_threshold": 0.3,  # Allows up to 90% padding
                    "padding_side": "post"
                }
            },
            "ts_sequence_mode": "pad",
            "psi_feature_selection": {
                "enabled": True,
                "threshold": 0.25,
                "split_frac": 0.75,
                "split_distinct": False,
                "apply_before_split": True
            },
            "feature_engine_split": {
                "enabled": True,
                "split_frac": 0.75,
                "split_distinct": False
            },
            "time_series_split": {
                "method": "feature_engine"
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        debug=True,
        graphs_output_dir="./plots"
    )
    
    # Preprocess the training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)
    
    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    # Train a model
    print("Training LSTM model with DTW/Pad mode...")
    # Extract horizon from preprocessor
    horizon = get_horizon_from_preprocessor(preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    # Build the LSTM model using the extracted horizon
    model4 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)

    model4.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model4.save('./transformers/model_dtw_pad.h5')
    
    # Test prediction mode
    print("\nTesting prediction mode with new data...")
    
    # Take the last segment of data as "new" data for prediction
    # new_data = data.iloc[-48:].copy()  
    
    # Configure the preprocessor for prediction
    predict_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "ts_sequence_mode": "pad"
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        transformers_dir="./transformers"
    )
    
    # Preprocess new data for prediction
    # results = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = results[0]
    X_new_preprocessed, recommendations, X_inversed = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    # Make predictions
    model4 = load_model('./transformers/model_dtw_pad.h5')
    expected_shape = model4.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model4.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    
    print("\n\nAll tests completed successfully!")

    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model4_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation metrics - MAE: {model4_metrics['mae']:.4f}, RMSE: {model4_metrics['rmse']:.4f}, R²: {model4_metrics['r2']:.4f}")


    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    new_data = new_data.copy()
    # Configure the preprocessor for training with DTW mode
    preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "dtw": {
                    "reference_sequence": "max",  # Use mean sequence as reference
                    "dtw_threshold": 0.3          # DTW threshold for sequences
                }
            },
            "ts_sequence_mode": "dtw",
            "psi_feature_selection": {
                "enabled": True,
                "threshold": 0.25,
                "split_frac": 0.75,
                "split_distinct": False,
                "apply_before_split": True
            },
            "feature_engine_split": {
                "enabled": True,
                "split_frac": 0.75,
                "split_distinct": False
            },
            "time_series_split": {
                "method": "feature_engine"
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="dtw",
        debug=True,
        graphs_output_dir="./plots"
    )
    
    # Preprocess the training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)
    
    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    # Train a model
    print("Training LSTM model with DTW/Pad mode...")
    # Extract horizon from preprocessor
    horizon = get_horizon_from_preprocessor(preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    # Build the LSTM model using the extracted horizon
    model5 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)

    model5.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model5.save('./transformers/model_dtw.h5')
    
    # Test prediction mode
    print("\nTesting prediction mode with new data...")
    
    # Take the last segment of data as "new" data for prediction
    # new_data = data.iloc[-48:].copy()  
    
    # Configure the preprocessor for prediction
    predict_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "ts_sequence_mode": "dtw"
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="dtw",
        transformers_dir="./transformers"
    )
    
    # Preprocess new data for prediction
    # results = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = results[0]
    X_new_preprocessed, recommendations, X_inversed = predict_preprocessor.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    # Make predictions
    model5 = load_model('./transformers/model_dtw.h5')
    expected_shape = model5.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model5.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    
    print("\n\nAll tests completed successfully!")

    # Test 5: Pad Mode with Percentage-Based Sequence-Aware Split
    print("\n\n=== Test 5: Pad Mode with Percentage-Based Sequence-Aware Split ===")

    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)
    new_data = new_data.copy()
    # Configure preprocessor for training with pad mode and percentage-based split
    pad_pct_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "pad": {
                    "pad_threshold": 0.3,  # Allows up to 90% padding
                    "padding_side": "post"
                }
            },
            "time_series_split": {
                "method": "sequence_aware",  # Use sequence-aware splitting
                # "test_size": 0.2,            # Use 20% of sequences for testing
                'target_train_fraction': 0.8,  # Aim for 80% training, 20% testing
                "debug_phases": True         # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        debug=True,
        graphs_output_dir="./plots"
    )

    # Preprocess training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = pad_pct_preprocessor.final_ts_preprocessing(data)

    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")

    # Train model
    print("Training LSTM model with pad mode and percentage-based split...")
    horizon = get_horizon_from_preprocessor(pad_pct_preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    model5 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)
    model5.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model5.save('./transformers/model_pad_pct.h5')

    # Test prediction
    print("\nTesting prediction mode with new data...")
    # new_data = data.iloc[-48:].copy()

    pad_pct_predict = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "time_series_split": {
                "method": "sequence_aware",  # Use sequence-aware splitting
                # "test_size": 0.2,            # Use 20% of sequences for testing
                'target_train_fraction': 0.8,  # Aim for 80% training, 20% testing
                "debug_phases": True         # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        transformers_dir="./transformers"
    )

    # result = pad_pct_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = result[0]
    X_new_preprocessed, recommendations, X_inversed = pad_pct_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    model5 = load_model('./transformers/model_pad_pct.h5')
    expected_shape = model5.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model5.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model5_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation model5_metrics - MAE: {model5_metrics['mae']:.4f}, RMSE: {model5_metrics['rmse']:.4f}, R²: {model5_metrics['r2']:.4f}")

    # Test 6: Pad Mode with Date-Based Sequence-Aware Split
    print("\n\n=== Test 6: Pad Mode with Date-Based Sequence-Aware Split ===")

    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    new_data = new_data.copy()
    # Calculate median date for splitting
    median_date = data['datetime'].median()
    print(f"Using median date as split point: {median_date}")

    # Configure preprocessor for training with pad mode and date-based split
    pad_date_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "pad": {
                    "pad_threshold": 0.3,  # Allows up to 90% padding
                    "padding_side": "post"
                }
            },
            "time_series_split": {
                "method": "sequence_aware",   # Use sequence-aware splitting
                "split_date": str(median_date), # Split at the median date
                "debug_phases": True          # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        debug=True,
        graphs_output_dir="./plots"
    )

    # Analyze potential split points first
    print("Analyzing potential split points...")
    split_options = pad_date_preprocessor.analyze_split_options(data)
    for i, option in enumerate(split_options[:3]):  # Show top 3
        print(f"Option {i+1}: Split at {option['split_time']} - Train fraction: {option['train_fraction']:.2f}")

    # Preprocess training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = pad_date_preprocessor.final_ts_preprocessing(data)

    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")

    # Train model
    print("Training LSTM model with pad mode and date-based split...")
    horizon = get_horizon_from_preprocessor(pad_date_preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    model6 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)
    model6.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model6.save('./transformers/model_pad_date.h5')

    # Test prediction
    print("\nTesting prediction mode with new data...")
    # new_data = data.iloc[-48:].copy()

    pad_date_predict = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "time_series_split": {
                "method": "sequence_aware",   # Use sequence-aware splitting
                "split_date": str(median_date), # Split at the median date
                "debug_phases": True          # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="pad",
        transformers_dir="./transformers"
    )

    # result = pad_date_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = result[0]
    X_new_preprocessed, recommendations, X_inversed = pad_date_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    
    model6 = load_model('./transformers/model_pad_date.h5')
    expected_shape = model6.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model6.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model6_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation model6_metrics - MAE: {model6_metrics['mae']:.4f}, RMSE: {model6_metrics['rmse']:.4f}, R²: {model6_metrics['r2']:.4f}")

    # Test 7: DTW Mode with Percentage-Based Sequence-Aware Split
    print("\n\n=== Test 7: DTW Mode with Percentage-Based Sequence-Aware Split ===")

    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)
    
    new_data = new_data.copy()
    # Configure preprocessor for training with DTW mode and percentage-based split
    dtw_pct_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "dtw": {
                    "reference_sequence": "max",  # Use max length sequence as reference
                    "dtw_threshold": 0.3          # DTW threshold for sequences
                }
            },
            "time_series_split": {
                "method": "sequence_aware",  # Use sequence-aware splitting
                # "test_size": 0.2,            # Use 20% of sequences for testing
                'target_train_fraction': 0.75,  # Aim for 80% training, 20% testing
                "debug_phases": True         # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="dtw",
        debug=True,
        graphs_output_dir="./plots"
    )

    # Preprocess training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = dtw_pct_preprocessor.final_ts_preprocessing(data)

    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")

    # Train model
    print("Training LSTM model with DTW mode and percentage-based split...")
    horizon = get_horizon_from_preprocessor(dtw_pct_preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    model7 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)
    model7.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model7.save('./transformers/model_dtw_pct.h5')

    # Test prediction
    print("\nTesting prediction mode with new data...")
    # new_data = data.iloc[-48:].copy()

    dtw_pct_predict = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "time_series_split": {
                "method": "sequence_aware",  # Use sequence-aware splitting
                # "test_size": 0.2,            # Use 20% of sequences for testing
                'target_train_fraction': 0.75,  # Aim for 80% training, 20% testing
                "debug_phases": True         # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="dtw",
        transformers_dir="./transformers"
    )
    
    # result = dtw_pct_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # X_new_preprocessed = result[0]
    X_new_preprocessed, recommendations, X_inversed = dtw_pct_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    model7 = load_model('./transformers/model_dtw_pct.h5')
    expected_shape = model7.input_shape
    print(f"Expected model input shape: {expected_shape}")
    predictions = model7.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model7_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation model7_metrics - MAE: {model7_metrics['mae']:.4f}, RMSE: {model7_metrics['rmse']:.4f}, R²: {model7_metrics['r2']:.4f}")

    # Test 8: DTW Mode with Date-Based Sequence-Aware Split
    print("\n\n=== Test 8: DTW Mode with Date-Based Sequence-Aware Split ===")

    # Clean transformers directory
    shutil.rmtree('./transformers', ignore_errors=True)
    os.makedirs('./transformers', exist_ok=True)

    # Configure preprocessor for training with DTW mode and date-based split
    dtw_date_preprocessor = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="train",
        options={
            "enabled": True,
            "time_column": "datetime",
            # "horizon": 379,
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {
                "dtw": {
                    "reference_sequence": "max",  # Use max length sequence as reference
                    "dtw_threshold": 0.3          # DTW threshold for sequences
                }
            },
            "time_series_split": {
                "method": "sequence_aware",      # Use sequence-aware splitting
                "split_date": str(split_date),   # Split at the calculated date
                "debug_phases": True             # Enable detailed phase debugging
            }
        },
        sequence_categorical=["session_biomech", "trial_biomech"],
        sub_sequence_categorical=["pitch_phase_biomech"],
        time_series_sequence_mode="dtw",
        debug=True,
        graphs_output_dir="./plots"
    )

    # Analyze potential split points first
    print("Analyzing potential split points...")
    split_options = dtw_date_preprocessor.analyze_split_options(data)
    for i, option in enumerate(split_options[:3]):  # Show top 3
        print(f"Option {i+1}: Split at {option['split_time']} - Train fraction: {option['train_fraction']:.2f}")

    # Preprocess training data
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = dtw_date_preprocessor.final_ts_preprocessing(data)

    print(f"Train shapes - X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"Test shapes - X: {X_test_seq.shape}, y: {y_test_seq.shape}")

    # Train model
    print("Training LSTM model with DTW mode and date-based split...")
    horizon = get_horizon_from_preprocessor(dtw_date_preprocessor)
    print(f"Using horizon of {horizon} for model output dimension")

    model8 = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), horizon=horizon)
    model8.fit(
        X_train_seq, y_train_seq, 
        validation_data=(X_test_seq, y_test_seq),
        epochs=10, batch_size=32, verbose=1
    )
    model8.save('./transformers/model_dtw_date.h5')

    # Test prediction
    print("\nTesting prediction mode with new data...")
    # new_data = select_complete_test_data(data, n_trials=2)
    # print(f"Selected test data shape: {new_data.shape}")
    new_data = test_data


    dtw_date_predict = DataPreprocessor(
        model_type="LSTM",
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode="predict",
        options={
            "enabled": True,
            "time_column": "datetime",
            "time_series_split": {
                "method": "sequence_aware",      # Use sequence-aware splitting
                "split_date": str(split_date),   # Split at the calculated date
                "debug_phases": True             # Enable detailed phase debugging
            }
        },
        sequence_categorical=["trial_id"],
        sub_sequence_categorical=["shooting_phases"],
        time_series_sequence_mode="dtw",
        transformers_dir="./transformers"
    )


    expected_shape = model8.input_shape
    X_new_preprocessed, recommendations, X_inversed = dtw_date_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    print(f"Expected model input shape: {expected_shape}")
    # result = dtw_date_predict.final_ts_preprocessing(new_data, model_input_shape=expected_shape)
    # result = debug_preprocessing_result(result, expected_shape)
                

    # X_new_preprocessed = result[0]
    # print(f"Type of result: {type(result)}")
    # if isinstance(result, tuple):
    #     print(f"Result contains {len(result)} elements")
    #     for i, item in enumerate(result):
    #         print(f"Item {i} is of type {type(item)}")
    #         if hasattr(item, 'shape'):
    #             print(f"  Shape: {item.shape}")
    model8 = load_model('./transformers/model_dtw_date.h5')
    predictions = model8.predict(X_new_preprocessed)
    print(f"Prediction results shape: {predictions.shape}")
    # Apply dimension compatibility function
    y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)

    # Compute and print evaluation metrics using the new function.
    model8_metrics = evaluate_model_metrics(y_test_seq, predictions)
    print(f"Model evaluation metrics - MAE: {model8_metrics['mae']:.4f}, RMSE: {model8_metrics['rmse']:.4f}, R²: {model8_metrics['r2']:.4f}")


    print("\n\nAll tests completed successfully!")


    # At the end of all tests, collect each model's metrics into a report list.
    evaluation_report = []
    evaluation_report.append({
        'test': 'Model 1: Percentage-based Split',
        'mae': model1_metrics['mae'],
        'rmse': model1_metrics['rmse'],
        'r2': model1_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 2: Date-based Split',
        'mae': model2_metrics['mae'],
        'rmse': model2_metrics['rmse'],
        'r2': model2_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 3: PSI-based Split with Feature-Engine',
        'mae': model3_metrics['mae'],
        'rmse': model3_metrics['rmse'],
        'r2': model3_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 4: DTW/Pad Mode with PSI-based Split',
        'mae': model4_metrics['mae'],
        'rmse': model4_metrics['rmse'],
        'r2': model4_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 5: DTW Mode with Feature-Engine Split',
        'mae': model5_metrics['mae'],
        'rmse': model5_metrics['rmse'],
        'r2': model5_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 6: Pad Mode with Date-based Sequence-Aware Split',
        'mae': model6_metrics['mae'],
        'rmse': model6_metrics['rmse'],
        'r2': model6_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 7: DTW Mode with Percentage-Based Sequence-Aware Split',
        'mae': model7_metrics['mae'],
        'rmse': model7_metrics['rmmse'] if 'rmmse' in model7_metrics else model7_metrics['rmse'],  # ensuring consistency
        'r2': model7_metrics['r2']
    })
    evaluation_report.append({
        'test': 'Model 8: DTW Mode with Date-Based Sequence-Aware Split',
        'mae': model8_metrics['mae'],
        'rmse': model8_metrics['rmse'],
        'r2': model8_metrics['r2']
    })

    # Generate the final summary report.
    generate_final_report(evaluation_report)

def get_horizon_from_preprocessor(preprocessor):
    """
    Extract the horizon parameter from the preprocessor.
    
    For DTW or pad modes, if the horizon has not been computed yet,
    it returns the product of horizon_sequence_number and sequence_length.
    Otherwise, it returns the computed horizon.
    """
    if hasattr(preprocessor, 'time_series_sequence_mode') and preprocessor.time_series_sequence_mode in ["dtw", "pad"]:
        if preprocessor.horizon is not None:
            return preprocessor.horizon
        else:
            # Compute dynamic horizon as horizon_sequence_number * sequence_length
            return preprocessor.horizon_sequence_number * preprocessor.sequence_length
    elif hasattr(preprocessor, 'options') and isinstance(preprocessor.options, dict):
        return preprocessor.options.get('horizon', 1)
    elif hasattr(preprocessor, 'horizon'):
        return preprocessor.horizon
    else:
        return 1  # Default horizon if not specified

        
# -----------------------------------------------------------------------------
# Helper function to ensure targets and predictions have compatible dimensions.
def ensure_compatible_dimensions(targets, predictions):
    import numpy as np

    targets = np.array(targets)
    predictions = np.array(predictions)

    if targets.ndim == 3 and targets.shape[2] == 1:
        targets = targets.squeeze(axis=2)
    if predictions.ndim == 3 and predictions.shape[2] == 1:
        predictions = predictions.squeeze(axis=2)

    if targets.shape[0] != predictions.shape[0]:
        n_samples = min(targets.shape[0], predictions.shape[0])
        targets = targets[:n_samples]
        predictions = predictions[:n_samples]

    if targets.ndim == 1 and predictions.ndim == 2:
        targets = targets.reshape(-1, 1)
    elif predictions.ndim == 1 and targets.ndim == 2:
        predictions = predictions.reshape(-1, 1)

    print(f"Adjusted shapes - targets: {targets.shape}, predictions: {predictions.shape}")
    return targets, predictions

# -----------------------------------------------------------------------------
# Function to train an LSTM (or TCN-LSTM) model.
def train_lstm_model(X_train, y_train, ts_params, config, use_tcn=False, bidirectional=False):
    """
    Train an LSTM or TCN-LSTM model based on provided configuration.

    Args:
        X_train (np.ndarray): Training data of shape (samples, timesteps, features).
        y_train (np.ndarray): Target training data.
        ts_params (dict): Time series parameters dictionary, used to extract horizon.
        config (dict): Configuration dictionary.
        use_tcn (bool): If True, use TCN layer before LSTM.
        bidirectional (bool): If True, use Bidirectional wrapper for LSTM layers.

    Returns:
        model: Trained Keras model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    if bidirectional:
        from tensorflow.keras.layers import Bidirectional

    horizon = ts_params.get("horizon", 1)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = Sequential()
    
    if use_tcn:
        try:
            from tcn import TCN
        except ImportError:
            raise ImportError("TCN layer not found. Please install the tcn package.")
        
        # Add a TCN layer; wrap with Bidirectional if needed.
        if bidirectional:
            model.add(Bidirectional(TCN(nb_filters=64, return_sequences=True), input_shape=input_shape))
        else:
            model.add(TCN(nb_filters=64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Follow with an LSTM layer.
        if bidirectional:
            model.add(Bidirectional(LSTM(32)))
        else:
            model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(horizon))
    else:
        # Standard LSTM architecture.
        if bidirectional:
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
        else:
            model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        if bidirectional:
            model.add(Bidirectional(LSTM(32)))
        else:
            model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(horizon))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train the model using default training parameters.
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    return model

# -----------------------------------------------------------------------------
# Experiment function for running models in different sequence modes.
def run_sequence_mode_experiment(data, sequence_mode, model_architectures):
    """
    Run an experiment for a given sequence mode by preprocessing the data,
    training a model for each architecture, and evaluating predictions.
    
    Args:
        data (pd.DataFrame): The complete dataset.
        sequence_mode (str): The sequence mode (e.g., "set_window", "dtw", "pad").
        model_architectures (list): List of architecture configurations to test.
        
    Returns:
        dict: A dictionary of results with metrics and shapes for each architecture.
    """
    results = {}

    # Set up sequence-specific parameters.
    if sequence_mode in ["set_window"]:
        ts_params = {
            "enabled": True,
            "time_column": "datetime",
            "horizon": 5,  # initial dummy horizon; will be updated later
            "step_size": 1,
            "window_size": 10,
            "sequence_modes": {},
            "ts_sequence_mode": sequence_mode,
            "split_dataset": {"test_size": 0.2, "random_state": 42},
            "time_series_split": {"method": "standard"}
        }
        ts_params["sequence_modes"]["set_window"] = {"window_size": 10, "max_sequence_length": 10}
    elif sequence_mode == "pad":
        ts_params = {
            "enabled": True,
            "time_column": "datetime",
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {},
            "ts_sequence_mode": sequence_mode,
            "split_dataset": {"test_size": 0.2, "random_state": 42},
            "time_series_split": {"method": "standard"}
        }
        ts_params["sequence_modes"]["pad"] = {"pad_threshold": 0.3, "padding_side": "post"}
    elif sequence_mode == "dtw":
        ts_params = {
            "enabled": True,
            "time_column": "datetime",
            "use_horizon_sequence": True,
            "horizon_sequence_number": 1,
            "step_size": 1,
            "sequence_modes": {},
            "ts_sequence_mode": sequence_mode,
            "split_dataset": {"test_size": 0.2, "random_state": 42},
            "time_series_split": {"method": "standard"}
        }
        ts_params["sequence_modes"]["dtw"] = {"reference_sequence": "max", "dtw_threshold": 0.3}

    try:
        # Create the DataPreprocessor for training.
        preprocessor = DataPreprocessor(
            model_type="LSTM",
            y_variable=y_variable,
            ordinal_categoricals=ordinal_categoricals,
            nominal_categoricals=nominal_categoricals,
            numericals=numericals,
            mode="train",
            options=ts_params,
            sequence_categorical=["trial_id"],
            sub_sequence_categorical=["shooting_phases"],
            time_series_sequence_mode=sequence_mode,
            debug=True,
            graphs_output_dir=graphs_output_dir,
            transformers_dir=transformers_dir
        )

        # Call final_ts_preprocessing; this call updates preprocessor.horizon dynamically
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, recommendations, _ = preprocessor.final_ts_preprocessing(data)

        # --- NEW STEP: Update ts_params with the computed horizon ---
        ts_params["horizon"] = preprocessor.horizon
        preprocessor.logger.info(f"Updated ts_params horizon to: {ts_params['horizon']}")

        # Loop over each model architecture to train and evaluate.
        for arch in model_architectures:
            arch_name = f"{'TCN-' if arch['use_tcn'] else ''}{'Bi' if arch['bidirectional'] else ''}LSTM"
            # Use the updated horizon from ts_params when building the model.
            model = build_lstm_model(
                (X_train_seq.shape[1], X_train_seq.shape[2]),
                horizon=ts_params.get("horizon", 1)
            )
            model.fit(
                X_train_seq, y_train_seq, 
                validation_data=(X_test_seq, y_test_seq),
                epochs=10, batch_size=32, verbose=1
            )
            predictions = model.predict(X_test_seq)
            # Ensure targets and predictions have compatible dimensions.
            y_test_seq, predictions = ensure_compatible_dimensions(y_test_seq, predictions)
            metrics = {
                'mae': mean_absolute_error(y_test_seq, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test_seq, predictions)),
                'r2': r2_score(y_test_seq, predictions)
            }
            results[arch_name] = {
                'metrics': metrics,
                'train_shape': X_train_seq.shape,
                'test_shape': X_test_seq.shape,
                'architecture': arch
            }
            del model
            tf.keras.backend.clear_session()

    except Exception as e:
        results['preprocessing_error'] = str(e)

    return results



# -----------------------------------------------------------------------------
def save_experiment_results(results, config):
    """Save experiment results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    filepath = os.path.join(config["paths"]["training_output_dir"], filename)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = json.loads(json.dumps(results, default=convert_to_serializable))
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")

# -----------------------------------------------------------------------------
def print_experiment_summary(all_results):
    """Print a summary of all experiment results."""
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    
    for sequence_mode, results in all_results.items():
        print(f"\nSequence Mode: {sequence_mode}")
        print("-" * 80)
        
        if 'preprocessing_error' in results:
            print(f"ERROR: {results['preprocessing_error']}")
            continue
            
        for arch_name, arch_results in results.items():
            if 'error' in arch_results:
                print(f"{arch_name}: ERROR - {arch_results['error']}")
                continue
                
            metrics = arch_results['metrics']
            print(f"\n{arch_name}:")
            print(f"  Sequence Shape: {arch_results['train_shape']}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
            # Assuming MAPE is part of metrics if computed elsewhere.
            if 'mape' in metrics:
                print(f"  MAPE: {metrics['mape']:.4f}")

# -----------------------------------------------------------------------------
# Define model architectures to test
model_architectures = [
    {'use_tcn': False, 'bidirectional': False},  # LSTM
    {'use_tcn': False, 'bidirectional': True},   # BiLSTM
    {'use_tcn': True, 'bidirectional': False},   # TCN-LSTM
    {'use_tcn': True, 'bidirectional': True}       # TCN-BiLSTM
]

# Sequence modes to test
sequence_modes = ["set_window", "dtw", "pad"]

# Run experiments for each sequence mode and collect results.
all_results = {}
for mode in sequence_modes:
    all_results[mode] = run_sequence_mode_experiment(data, mode, model_architectures)

# Print summary of all experiments.
print_experiment_summary(all_results)
