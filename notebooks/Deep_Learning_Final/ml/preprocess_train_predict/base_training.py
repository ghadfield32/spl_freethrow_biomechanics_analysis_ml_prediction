
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
from sklearn.metrics import (mean_squared_error,
    mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tensorflow.keras import Input
from ml.load_and_prepare_data.load_data_and_analyze import (
    load_data, prepare_joint_features, feature_engineering, summarize_data, check_and_drop_nulls)

from ml.feature_selection.feature_selection import (
    load_top_features, perform_feature_importance_analysis, save_top_features,
    analyze_joint_injury_features, check_for_invalid_values,
    perform_feature_importance_analysis, analyze_and_display_top_features, validate_features)



# ==================== UTILS ====================
       
def temporal_train_test_split(data, test_size=0.2):
    """Time-based split maintaining temporal order"""
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    logging.info(f"Performed temporal train-test split with test size = {test_size}")
    logging.info(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
    return train_data, test_data

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Features have been scaled using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler

def create_sequences(X, y, timesteps):
    """
    Creates sequences of data for LSTM input.
    """
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    logging.info(f"Created LSTM sequences: {X_seq.shape}, {y_seq.shape}")
    return X_seq, y_seq
# ==================== TRAINING FUNCTIONS ====================

def train_exhaustion_model(train_data, test_data, features, timesteps, 
                           epochs=50, batch_size=32, early_stop_patience=5,
                           num_lstm_layers=1, lstm_units=64, dropout_rate=0.2,
                           dense_units=1, dense_activation=None, target_col="by_trial_exhaustion_score"):
    """
    Trains an exhaustion model (regression) with a separate target scaler.
    
    An optional parameter 'target_col' is added so that this function can be used
    for joint-specific exhaustion targets if needed. The default is "by_trial_exhaustion_score".
    
    Parameters:
      - train_data (DataFrame): Training set.
      - test_data (DataFrame): Testing set.
      - features (list): List of feature column names for exhaustion.
      - timesteps (int): Number of past observations to include in each sequence.
      - epochs (int): Number of training epochs.
      - batch_size (int): Batch size for training.
      - early_stop_patience (int): Patience for EarlyStopping callback.
      - num_lstm_layers (int): Number of LSTM layers in the model.
      - lstm_units (int): Number of units in each LSTM layer.
      - dropout_rate (float): Dropout rate applied after each LSTM layer.
      - dense_units (int): Number of units in the final Dense layer.
      - dense_activation (str or None): Activation function for the Dense layer.
      - target_col (str): The name of the target column to use. Default is "by_trial_exhaustion_score".
      
    Returns:
      - model_exhaustion: Trained Keras model.
      - scaler_exhaustion: Fitted scaler for the features.
      - target_scaler: Fitted scaler for the target values.
      - X_lstm_exhaustion_val, y_lstm_exhaustion_val: Validation sequences.
    """
    # --- Debug: Log the feature list and available columns ---
    logging.info(f"Features provided for training exhaustion model: {features}")
    logging.info(f"Available train_data columns: {train_data.columns.tolist()}")
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        logging.error(f"Missing features in train_data for target {target_col}: {missing_features}")
        raise KeyError(f"{missing_features} not in train_data.columns")
    
    # Extract features and target from training and testing data using target_col
    X_train = train_data[features].values
    y_train = train_data[target_col].values
    X_test = test_data[features].values
    y_test = test_data[target_col].values

    # Scale features
    X_train_scaled, X_test_scaled, scaler_exhaustion = scale_features(X_train, X_test)
    
    # Scale target values separately
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
    
    # Create sequences for LSTM input
    X_lstm, y_lstm = create_sequences(X_train_scaled, y_train_scaled, timesteps)
    X_lstm_val, y_lstm_val = create_sequences(X_test_scaled, y_test_scaled, timesteps)


    model_exhaustion = Sequential()
    model_exhaustion.add(Input(shape=(X_lstm.shape[1], X_lstm.shape[2])))
    for i in range(num_lstm_layers):
        return_seq = True if i < num_lstm_layers - 1 else False
        model_exhaustion.add(LSTM(lstm_units, return_sequences=return_seq))
        model_exhaustion.add(Dropout(dropout_rate))
    model_exhaustion.add(Dense(dense_units, activation=dense_activation))
    
    model_exhaustion.compile(optimizer='adam', loss='mse')
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
    
    logging.info("Training exhaustion model...")
    model_exhaustion.fit(
        X_lstm, y_lstm,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_lstm_val, y_lstm_val),
        callbacks=[early_stop]
    )
    
    return model_exhaustion, scaler_exhaustion, target_scaler, X_lstm_val, y_lstm_val


def train_injury_model(train_data, test_data, features, timesteps,
                       epochs=50, batch_size=32, early_stop_patience=5,
                       num_lstm_layers=1, lstm_units=64, dropout_rate=0.2,
                       dense_units=1, dense_activation='sigmoid', target_col="injury_risk"):
    """
    Trains an injury risk model (classification). An optional parameter 'target_col'
    is added so that this function can be used for joint-specific injury targets.
    The default is "injury_risk".
    
    Parameters:
      - train_data (DataFrame): Training set.
      - test_data (DataFrame): Testing set.
      - features (list): List of feature column names for injury risk.
      - timesteps (int): Number of past observations to include in each sequence.
      - epochs (int): Number of training epochs.
      - batch_size (int): Batch size for training.
      - num_lstm_layers (int): Number of LSTM layers in the model.
      - lstm_units (int): Number of units in each LSTM layer.
      - dropout_rate (float): Dropout rate applied after each LSTM layer.
      - dense_units (int): Number of units in the final Dense layer.
      - dense_activation (str): Activation function for the Dense layer.
      - target_col (str): The target column to use. Default is "injury_risk".
      
    Returns:
      - model_injury: Trained Keras model.
      - scaler_injury: Fitted scaler for the features.
      - X_lstm_injury_val, y_lstm_injury_val: Validation sequences.
    """
    # --- Debug: Log the feature list and available columns ---
    logging.info(f"Features provided for training injury model: {features}")
    logging.info(f"Available train_data columns: {train_data.columns.tolist()}")
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        logging.error(f"Missing features in train_data for target {target_col}: {missing_features}")
        raise KeyError(f"{missing_features} not in train_data.columns")
    
    # Extract features and target using target_col
    X_train = train_data[features].values
    y_train = train_data[target_col].values
    X_test = test_data[features].values
    y_test = test_data[target_col].values

    # Scale features
    X_train_scaled, X_test_scaled, scaler_injury = scale_features(X_train, X_test)
    # Create LSTM sequences
    X_lstm, y_lstm = create_sequences(X_train_scaled, y_train, timesteps)
    X_lstm_val, y_lstm_val = create_sequences(X_test_scaled, y_test, timesteps)


    model_injury = Sequential()
    model_injury.add(Input(shape=(X_lstm.shape[1], X_lstm.shape[2])))
    for i in range(num_lstm_layers):
        return_seq = True if i < num_lstm_layers - 1 else False
        model_injury.add(LSTM(lstm_units, return_sequences=return_seq))
        model_injury.add(Dropout(dropout_rate))
    model_injury.add(Dense(dense_units, activation=dense_activation))
    
    model_injury.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Training injury risk model...")
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
    # Include early_stop in model.fit():
    model_injury.fit(
        X_lstm, y_lstm,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_lstm_val, y_lstm_val),
        callbacks=[early_stop]
    )
    
    return model_injury, scaler_injury, X_lstm_val, y_lstm_val



# ==================== JOINT-SPECIFIC TRAINING FUNCTION ====================



def train_joint_models(train_data, test_data, joints, timesteps, feature_dir,
                       epochs=50, batch_size=32,
                       num_lstm_layers=1, lstm_units=64, dropout_rate=0.2,
                       dense_units=1, dense_activation='sigmoid',
                       joint_feature_dict=None):
    """
    Trains injury risk models for multiple joints.
    
    Parameters:
      - train_data (DataFrame): Training set.
      - test_data (DataFrame): Testing set.
      - joints (list): List of joint names.
      - timesteps (int): Number of past observations to include in each sequence.
      - feature_dir (str): Directory containing feature lists.
      - epochs (int): Number of training epochs.
      - batch_size (int): Batch size for training.
      - num_lstm_layers (int): Number of LSTM layers in the model.
      - lstm_units (int): Number of units in each LSTM layer.
      - dropout_rate (float): Dropout rate applied after each LSTM layer.
      - dense_units (int): Number of units in the final Dense layer.
      - dense_activation (str): Activation function for the Dense layer.
      - joint_feature_dict (dict or None): (Optional) Dictionary mapping each joint target (e.g., 
            "L_ANKLE_injury_risk") to a preloaded feature list. If None, the function calls
            load_top_features for each target.
      
    Returns:
      - joint_models (dict): Dictionary with joint model information.
    """
    joint_models = {}

    for joint in joints:
        for side in ['L', 'R']:
            target_joint = f"{side}_{joint}_injury_risk"
            logging.info(f"Training model for {target_joint}...")

            # If a joint_feature_dict is provided and has the target, use it;
            # otherwise, call load_top_features from the specified directory.
            if joint_feature_dict is not None and target_joint in joint_feature_dict:
                joint_features = joint_feature_dict[target_joint]
                logging.info(f"Using preloaded feature list for {target_joint}: {joint_features}")
            else:
                joint_features = load_top_features(target_joint, feature_dir=feature_dir)
            
            # Extract joint-specific features and target values.
            X_train_joint = train_data[joint_features].values
            y_train_joint = train_data[target_joint].values
            X_test_joint = test_data[joint_features].values
            y_test_joint = test_data[target_joint].values

            # Scale features for the joint-specific model.
            X_train_scaled, X_test_scaled, scaler_joint = scale_features(X_train_joint, X_test_joint)
            # Create sequences for LSTM input.
            X_lstm, y_lstm = create_sequences(X_train_scaled, y_train_joint, timesteps)
            X_lstm_val, y_lstm_val = create_sequences(X_test_scaled, y_test_joint, timesteps)

            # Build the joint model.
            model_joint = Sequential()
            model_joint.add(Input(shape=(X_lstm.shape[1], X_lstm.shape[2])))
            for i in range(num_lstm_layers):
                return_seq = True if i < num_lstm_layers - 1 else False
                model_joint.add(LSTM(lstm_units, return_sequences=return_seq))
                model_joint.add(Dropout(dropout_rate))
            model_joint.add(Dense(dense_units, activation=dense_activation))

            model_joint.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model_joint.fit(
                X_lstm, y_lstm,
                epochs=epochs, 
                batch_size=batch_size,
                validation_data=(X_lstm_val, y_lstm_val)
            )

            joint_models[target_joint] = {
                'model': model_joint,
                'features': joint_features,
                'scaler': scaler_joint
            }

    # Save the loaded feature lists for all joint models.
    with open("loaded_features.json", "w") as f:
        json.dump({target: info['features'] for target, info in joint_models.items()}, f, indent=4)
    logging.info("Saved loaded features list for each joint model to 'loaded_features.json'.")

    return joint_models




# ==================== FORECASTING FUNCTION ====================

def forecast_and_plot_exhaustion(model, test_data, forecast_features, scaler_exhaustion, target_scaler, timesteps, future_steps=0, title="Exhaustion Forecast"):
    """
    Generates predictions for the exhaustion target using multi-feature input.
    
    This function extracts the same features used during training (e.g. a 10-dimensional input),
    scales them with the features scaler (scaler_exhaustion), builds forecasting sequences, makes predictions,
    and finally inverse-transforms the predictions using the target scaler.
    
    Parameters:
      - model: Trained exhaustion Keras model.
      - test_data (DataFrame): The test DataFrame containing all features.
      - forecast_features (list): List of feature names used for forecasting (e.g. features_exhaustion).
      - scaler_exhaustion: Fitted StandardScaler used to scale the features.
      - target_scaler: Fitted StandardScaler used to scale the target values.
      - timesteps (int): Number of past observations to include in each sequence.
      - future_steps (int): Number of future time steps to forecast.
                          (Note: Future forecasting is approximate since it assumes constant features.)
      - title (str): Plot title.
    """

    # Extract multi-dimensional input from test data
    X_forecast = test_data[forecast_features].values  # shape (n, num_features)
    
    # Scale the features using the features scaler
    X_forecast_scaled = scaler_exhaustion.transform(X_forecast)
    
    # Create sequences for forecasting using a dummy y array (since only X is needed)
    X_seq, _ = create_sequences(X_forecast_scaled, np.zeros(len(X_forecast_scaled)), timesteps)
    
    # Make predictions on the scaled sequences
    predictions_scaled = model.predict(X_seq)
    # Inverse-transform predictions using the target scaler
    predictions = target_scaler.inverse_transform(predictions_scaled)
    
    forecast_predictions_inv = None
    if future_steps > 0:
        # For additional future steps, we assume the features remain constant.
        # WARNING: This is an approximation.
        current_sequence = X_seq[-1].copy()  # shape: (timesteps, num_features)
        forecast_predictions = []
        for _ in range(future_steps):
            next_pred = model.predict(current_sequence.reshape(1, timesteps, current_sequence.shape[1]))
            forecast_predictions.append(next_pred[0, 0])
            # Update sequence: drop the first row and append the last row (assumed constant)
            new_row = current_sequence[-1, :].copy()
            current_sequence = np.vstack([current_sequence[1:], new_row])
        forecast_predictions = np.array(forecast_predictions).reshape(-1, 1)
        forecast_predictions_inv = target_scaler.inverse_transform(forecast_predictions)
    
    # Plot actual exhaustion scores versus predictions
    plt.figure(figsize=(10, 6))
    actual = test_data['by_trial_exhaustion_score'].values
    plt.plot(range(timesteps, len(actual)), actual[timesteps:], color='red', label='Actual')
    plt.plot(range(timesteps, len(actual)), predictions, color='blue', label='Predicted')
    if forecast_predictions_inv is not None:
        future_x = list(range(len(actual), len(actual) + future_steps))
        plt.plot(future_x, forecast_predictions_inv, color='green', linestyle='--', label='Forecast')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Exhaustion Score')
    plt.legend()
    plt.show()


def forecast_and_plot_injury(model, test_data, forecast_features, scaler_injury, timesteps, future_steps=0, title="Injury Risk Forecast"):
    """
    Generates predictions for the injury risk model using multi-feature input.
    
    This function extracts the injury features from the test data, scales them using scaler_injury,
    builds forecasting sequences, and makes predictions. Since this is a classification model,
    it outputs probability predictions. These probabilities (or rounded binary classes) are compared
    to the actual injury risk (assumed to be 0 or 1).
    
    Parameters:
      - model: Trained injury risk Keras model.
      - test_data (DataFrame): The test DataFrame containing all features.
      - forecast_features (list): List of feature names used for forecasting (e.g. features_injury).
      - scaler_injury: Fitted StandardScaler used to scale the injury features.
      - timesteps (int): Number of past observations to include in each sequence.
      - future_steps (int): Number of future time steps to forecast.
                          (For classification, future forecasting is less common.)
      - title (str): Plot title.
    """

    # Extract multi-dimensional input for injury risk from test data
    X_forecast = test_data[forecast_features].values  # shape (n, num_features)
    print(f"X_forecast shape: {X_forecast.shape}, min: {X_forecast.min()}, max: {X_forecast.max()}")
        
    # Scale the features using the injury features scaler
    X_forecast_scaled = scaler_injury.transform(X_forecast)
    print(f"X_forecast_scaled shape: {X_forecast_scaled.shape}, min: {X_forecast_scaled.min()}, max: {X_forecast_scaled.max()}")
    
    # Create sequences for forecasting (dummy y used, since only X is needed)
    X_seq, _ = create_sequences(X_forecast_scaled, np.zeros(len(X_forecast_scaled)), timesteps)
    
    # Predict probabilities on the sequences
    predictions_prob = model.predict(X_seq)
    print(f"predictions_prob shape: {predictions_prob.shape}, min: {predictions_prob.min()}, max: {predictions_prob.max()}")
    
    # Add clipping to ensure valid probability range
    predictions_prob_clipped = np.clip(predictions_prob, 0, 1)
    print(f"After clipping - min: {predictions_prob_clipped.min()}, max: {predictions_prob_clipped.max()}")
    
    # Convert probabilities to binary predictions (threshold=0.5)
    predictions_class = (predictions_prob >= 0.5).astype(int)
    
    forecast_predictions = None
    if future_steps > 0:
        # For future steps, we assume features remain constant (approximation)
        current_sequence = X_seq[-1].copy()  # shape: (timesteps, num_features)
        forecast_predictions = []
        for _ in range(future_steps):
            next_pred = model.predict(current_sequence.reshape(1, timesteps, current_sequence.shape[1]))
            forecast_predictions.append((next_pred[0, 0] >= 0.5).astype(int))
            new_row = current_sequence[-1, :].copy()
            current_sequence = np.vstack([current_sequence[1:], new_row])
        forecast_predictions = np.array(forecast_predictions)
    
    # Plot the actual injury risk versus predicted probability (or binary prediction)
    plt.figure(figsize=(10, 6))
    actual = test_data['injury_risk'].values
    # For plotting, we align the sequences starting at index 'timesteps'
    plt.plot(range(timesteps, len(actual)), actual[timesteps:], color='red', label='Actual')
    plt.plot(range(timesteps, len(actual)), predictions_prob, color='blue', label='Predicted Probability')
    if forecast_predictions is not None:
        future_x = list(range(len(actual), len(actual) + future_steps))
        plt.plot(future_x, forecast_predictions, color='green', linestyle='--', label='Forecasted Class')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Injury Risk')
    plt.legend()
    plt.show()


def forecast_and_plot_joint(joint_models, test_data, timesteps, future_steps=0):
    """
    Generates forecasts for each joint model using their corresponding features and scalers.
    
    For each joint model in the joint_models dictionary (returned by train_joint_models),
    this function extracts the joint-specific features from test_data, scales them using the model's scaler,
    builds sequences, obtains predictions (probabilities), converts them to binary predictions,
    and then plots the actual joint injury risk versus predicted values.
    
    Parameters:
      - joint_models (dict): Dictionary where each key is a joint target name and each value is a dict 
                             containing 'model', 'features', and 'scaler'.
      - test_data (DataFrame): The test DataFrame containing all features.
      - timesteps (int): Number of past observations to include in each sequence.
      - future_steps (int): Number of future time steps to forecast (optional).
    """
    for target_joint, info in joint_models.items():
        model_joint = info['model']
        joint_features = info['features']
        scaler_joint = info['scaler']
        
        # Extract joint-specific features from test data
        X_forecast = test_data[joint_features].values  # shape (n, num_features)
        # Scale the features
        X_forecast_scaled = scaler_joint.transform(X_forecast)
        # Create sequences for forecasting (dummy y used)
        X_seq, _ = create_sequences(X_forecast_scaled, np.zeros(len(X_forecast_scaled)), timesteps)
        
        # Predict probabilities and convert to binary predictions
        predictions_prob = model_joint.predict(X_seq)
        predictions_class = (predictions_prob >= 0.5).astype(int)
        
        forecast_predictions = None
        if future_steps > 0:
            # Approximate forecasting by assuming constant features
            current_sequence = X_seq[-1].copy()  # shape: (timesteps, num_features)
            forecast_predictions = []
            for _ in range(future_steps):
                next_pred = model_joint.predict(current_sequence.reshape(1, timesteps, current_sequence.shape[1]))
                forecast_predictions.append((next_pred[0, 0] >= 0.5).astype(int))
                new_row = current_sequence[-1, :].copy()
                current_sequence = np.vstack([current_sequence[1:], new_row])
            forecast_predictions = np.array(forecast_predictions)
        
        # Plot for this joint model
        plt.figure(figsize=(10, 6))
        actual = test_data[target_joint].values
        plt.plot(range(timesteps, len(actual)), actual[timesteps:], color='red', label='Actual')
        plt.plot(range(timesteps, len(actual)), predictions_prob, color='blue', label='Predicted Probability')
        if forecast_predictions is not None:
            future_x = list(range(len(actual), len(actual) + future_steps))
            plt.plot(future_x, forecast_predictions, color='green', linestyle='--', label='Forecasted Class')
        plt.title(f"Forecast for {target_joint}")
        plt.xlabel('Time')
        plt.ylabel('Injury Risk')
        plt.legend()
        plt.show()



def summarize_regression_model(model, X_val, y_val, target_scaler=None, debug=False):
    """
    Evaluates regression model performance.
    
    Parameters:
      - model: Trained regression Keras model.
      - X_val: Validation feature sequences.
      - y_val: Validation target values (scaled).
      - target_scaler: Optional StandardScaler used to inverse-transform targets.
      - debug (bool): If True, prints detailed debug information.
      
    Returns:
      - A dictionary containing Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score.
    """
    preds_scaled = model.predict(X_val)
    
    if target_scaler:
        preds = target_scaler.inverse_transform(preds_scaled)
        y_true = target_scaler.inverse_transform(y_val)
    else:
        preds = preds_scaled
        y_true = y_val

    if debug:
        print(f"[DEBUG] Predictions stats: min={preds.min()}, max={preds.max()}, nan_count={np.isnan(preds).sum()}")
        print(f"[DEBUG] True values stats: min={y_true.min()}, max={y_true.max()}, nan_count={np.isnan(y_true).sum()}")

    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2
    }


def summarize_classification_model(model, X_val, y_val, debug=False):
    """
    Evaluates classification model performance.
    
    Parameters:
      - model: Trained classification Keras model.
      - X_val: Validation feature sequences.
      - y_val: True binary labels.
      - debug (bool): If True, prints detailed debug information.
      
    Returns:
      - A dictionary containing Accuracy, Precision, Recall, and F1 Score.
    """
    preds_prob = model.predict(X_val)
    preds_class = (preds_prob >= 0.5).astype(int)
    
    if debug:
        print(f"[DEBUG] Predicted probabilities stats: min={preds_prob.min()}, max={preds_prob.max()}, nan_count={np.isnan(preds_prob).sum()}")

    accuracy = accuracy_score(y_val, preds_class)
    precision = precision_score(y_val, preds_class, zero_division=0)
    recall = recall_score(y_val, preds_class, zero_division=0)
    f1 = f1_score(y_val, preds_class, zero_division=0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def summarize_joint_models(joint_models, test_data, timesteps, debug=False):
    """
    Summarizes evaluation metrics for each joint model (classification or regression).
    
    For every joint model, the function:
      - Extracts joint-specific features from test_data.
      - Scales the features.
      - Creates sequences for LSTM input.
      - Computes classification or regression metrics using the appropriate function.
    
    Parameters:
      - joint_models (dict): Dictionary with joint model info ('model', 'features', 'scaler').
      - test_data (DataFrame): Test dataset containing joint-specific features.
      - timesteps (int): Number of past observations per sequence.
      - debug (bool): If True, enables debugging outputs.
      
    Returns:
      - Dictionary mapping each joint target to its evaluation metrics.
    """
    summaries = {}
    for target_joint, info in joint_models.items():
        model_joint = info['model']
        joint_features = info['features']
        scaler_joint = info['scaler']
        
        validate_features(joint_features, test_data, context=f"summarize_joint_models for {target_joint}")
        
        X_joint = test_data[joint_features].values
        y_joint = test_data[target_joint].values

        X_joint_scaled = scaler_joint.transform(X_joint)
        X_seq, y_seq = create_sequences(X_joint_scaled, y_joint, timesteps)
        
        # Determine if this is a classification or regression model
        if 'injury_risk' in target_joint:
            metrics = summarize_classification_model(model_joint, X_seq, y_seq, debug=debug)
        elif 'by_trial_exhaustion_score' in target_joint:
            # Assuming we have a target_scaler for regression models
            target_scaler = info.get('target_scaler')
            metrics = summarize_regression_model(model_joint, X_seq, y_seq, target_scaler, debug=debug)
        else:
            logging.warning(f"Unknown model type for {target_joint}. Skipping evaluation.")
            continue
        
        summaries[target_joint] = metrics
        
        if debug:
            logging.debug(f"[DEBUG] For {target_joint}: expected features: {joint_features}")
            logging.debug(f"[DEBUG] Test data columns: {test_data.columns.tolist()}")

    return summaries



def summarize_all_models(model_exhaustion, X_val_exh, y_val_exh, target_scaler,
                         model_injury, X_val_injury, y_val_injury,
                         joint_models, test_data, timesteps, output_dir,
                         include_joint_models=True, debug=False):
    """
    Summarizes exhaustion, injury, and optionally joint models into one DataFrame.

    Parameters:
      - include_joint_models (bool): Set False when summarizing aggregated datasets to prevent errors.
    """
    summary_data = []

    # Regression model summary
    exh_metrics = summarize_regression_model(model_exhaustion, X_val_exh, y_val_exh, target_scaler, debug=debug)
    summary_data.append({
        "Model": "Exhaustion Model",
        "Type": "Regression",
        "MSE": exh_metrics["MSE"],
        "MAE": exh_metrics["MAE"],
        "R2 Score": exh_metrics["R2 Score"],
        "Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None
    })

    # Classification model summary
    injury_metrics = summarize_classification_model(model_injury, X_val_injury, y_val_injury, debug=debug)
    summary_data.append({
        "Model": "Injury Model",
        "Type": "Classification",
        "MSE": None, "MAE": None, "R2 Score": None,
        "Accuracy": injury_metrics["Accuracy"],
        "Precision": injury_metrics["Precision"],
        "Recall": injury_metrics["Recall"],
        "F1 Score": injury_metrics["F1 Score"]
    })

    # Joint models summary (if included)
    if include_joint_models:
        joint_summaries = summarize_joint_models(joint_models, test_data, timesteps, debug=debug)
        for joint, metrics in joint_summaries.items():
            # Determine the model type based on the metrics available
            if "MSE" in metrics:
                # This is a regression model
                summary_data.append({
                    "Model": joint,
                    "Type": "Regression",
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                    "R2 Score": metrics["R2 Score"],
                    "Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None
                })
            elif "Accuracy" in metrics:
                # This is a classification model
                summary_data.append({
                    "Model": joint,
                    "Type": "Classification",
                    "MSE": None, "MAE": None, "R2 Score": None,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"]
                })
            else:
                logging.warning(f"Unknown metrics type for {joint}: {list(metrics.keys())}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Path(output_dir) / "model_summary_final.csv", index=False)
    logging.info(f"Saved summary to {output_dir}/model_summary_final.csv")
    
    return summary_df



def final_model_summary(
    regression_summaries, classification_summaries,
    regression_names=None, classification_names=None,
    joint_summaries=None, joint_names=None,
    darts_metrics=None
):
    """
    Combines multiple summary DataFrames for regression and classification models.
    
    Parameters:
      - regression_summaries (list of DataFrame): Each DataFrame contains regression model summary metrics.
      - classification_summaries (list of DataFrame): Each DataFrame contains classification model summary metrics.
      - regression_names (list of str, optional): Names for each regression summary (e.g., "Base", "Trial Aggregated", "Shot Aggregated").
      - classification_names (list of str, optional): Names for each classification summary.
      - joint_summaries (list of DataFrame, optional): Each DataFrame contains joint model summary metrics.
      - joint_names (list of str, optional): Names for each joint summary.
    
    Returns:
      - final_reg (DataFrame): Combined regression summaries.
      - final_class (DataFrame): Combined classification summaries.
      - final_joint (DataFrame): Combined joint model summaries.
      - final_all (DataFrame): Combined DataFrame of all summaries.
    """
    # If names are provided, add a "Dataset" column to each corresponding DataFrame.
    regression_dfs = []
    if regression_names is not None:
        for df, name in zip(regression_summaries, regression_names):
            df = df.copy()
            df["Dataset"] = name
            regression_dfs.append(df)
    else:
        regression_dfs = regression_summaries

    classification_dfs = []
    if classification_names is not None:
        for df, name in zip(classification_summaries, classification_names):
            df = df.copy()
            df["Dataset"] = name
            classification_dfs.append(df)
    else:
        classification_dfs = classification_summaries
        
    joint_dfs = []
    if joint_names is not None and joint_summaries is not None:
        for df, name in zip(joint_summaries, joint_names):
            df = df.copy()
            df["Dataset"] = name
            joint_dfs.append(df)
    elif joint_summaries is not None:
        joint_dfs = joint_summaries

    # Concatenate the lists into final DataFrames.
    final_reg = pd.concat(regression_dfs, ignore_index=True)
    final_class = pd.concat(classification_dfs, ignore_index=True)
    
    if joint_dfs:
        final_joint = pd.concat(joint_dfs, ignore_index=True)
        final_all = pd.concat([final_reg, final_class, final_joint], ignore_index=True)
    else:
        final_joint = pd.DataFrame()  # Empty DataFrame
        final_all = pd.concat([final_reg, final_class], ignore_index=True)
        
    # AFTER building final_all:
    if darts_metrics is not None and not darts_metrics.empty:
        try:
            # darts_metrics must have columns: Model, MAE, MAPE, RMSE, SMAPE, Target
            dm = darts_metrics.copy()
            dm['Type'] = 'Time Series Forecast'
            dm['Dataset'] = 'Darts Models'
            # reorder columns to match final_all if needed
            # e.g. ['Model','Type','Dataset','MAE','MAPE','RMSE','SMAPE','Target']
            final_all = pd.concat([final_all, dm], ignore_index=True, sort=False)
            logging.info("[Summary] appended Darts metrics")
        except Exception as e:
            logging.error(f"[Summary] failed to append Darts metrics: {e}")

    return final_reg, final_class, final_joint, final_all


def summarize_joint_exhaustion_models(joint_exh_models, test_data, timesteps, debug=False):
    """
    Computes evaluation metrics for each joint-specific exhaustion model (regression).
    
    For each model in joint_exh_models:
      - Extracts the joint-specific features from test_data.
      - Scales the features.
      - Creates sequences.
      - Computes regression metrics using summarize_regression_model.
    
    Parameters:
      - joint_exh_models (dict): Dictionary with keys as target names and values containing 'model', 'features', and 'scaler'.
      - test_data (DataFrame): Test data.
      - timesteps (int): Number of past observations per sequence.
      - debug (bool): Enables debug output if True.
      
    Returns:
      - summaries (dict): Dictionary mapping each joint exhaustion target to its evaluation metrics.
    """
    summaries = {}
    for target, info in joint_exh_models.items():
        model_exh = info['model']
        features = info['features']
        scaler = info['scaler']
        
        # Validate that the required features exist in test_data.
        validate_features(features, test_data, context=f"summarize_joint_exhaustion_models for {target}")
        
        # Extract features and target.
        X_joint = test_data[features].values
        y_joint = test_data[target].values
        
        # Scale features.
        X_joint_scaled = scaler.transform(X_joint)
        # Create sequences.
        X_seq, y_seq = create_sequences(X_joint_scaled, y_joint, timesteps)
        
        # Use the regression summary function (no target_scaler needed if y is not scaled further).
        metrics = summarize_regression_model(model_exh, X_seq, y_seq, target_scaler=None, debug=debug)
        summaries[target] = metrics
    return summaries


# --- Main Script (Updated) ---
if __name__ == "__main__":
    import os
    from pathlib import Path
    from ml.load_and_prepare_data.load_data_and_analyze import (
        load_data, prepare_joint_features, feature_engineering, summarize_data, check_and_drop_nulls)
    
    from ml.feature_selection.feature_selection import (
        load_top_features, perform_feature_importance_analysis, save_top_features,
        analyze_joint_injury_features, check_for_invalid_values,
        perform_feature_importance_analysis, analyze_and_display_top_features, validate_features)
    # from ml.preprocess_train_predict.base_training.py import (
    #     temporal_train_test_split, scale_features, create_sequences, train_exhaustion_model, 
    #     train_injury_model,  train_joint_models, forecast_and_plot_exhaustion, forecast_and_plot_injury,
    #     forecast_and_plot_joint, summarize_regression_model, summarize_classification_model, 
    #     summarize_joint_models, summarize_all_models, final_model_summary, 
    #     summarize_joint_exhaustion_models
    #     )
    debug = True
    importance_threshold = 0.05  # Set threshold as needed
    csv_path = "../../data/processed/final_granular_dataset.csv"
    json_path = "../../data/basketball/freethrow/participant_information.json"
    feature_dir = "../../data/Deep_Learning_Final"  # Directory where feature lists were saved
    output_dir = "../../data/Deep_Learning_Final"  # Base directory
    
    # Define directories for saving feature lists per dataset type
    base_feature_dir = os.path.join(output_dir, "feature_lists/base")
    trial_feature_dir = os.path.join(output_dir, "feature_lists/trial_summary")
    shot_feature_dir = os.path.join(output_dir, "feature_lists/shot_phase_summary")
    
    # Load and process data
    data = load_data(csv_path, json_path, debug=debug)
    data = prepare_joint_features(data, debug=debug)
    data = feature_engineering(data, debug=debug)
    print("Base data columns:", data.columns.tolist())
    
    # Create aggregated datasets for trial and shot-phase analyses
    default_agg_columns = [
        'joint_energy', 'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'L_KNEE_energy', 'R_KNEE_energy', 'L_HIP_energy', 'R_HIP_energy',
        'joint_power', 'L_ELBOW_ongoing_power', 'R_ELBOW_ongoing_power', 
        'L_WRIST_ongoing_power', 'R_WRIST_ongoing_power',
        'L_KNEE_ongoing_power', 'R_KNEE_ongoing_power', 'L_HIP_ongoing_power', 'R_HIP_ongoing_power',
        'elbow_asymmetry', 'wrist_asymmetry', 'knee_asymmetry', 'hip_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle', 
        'L_KNEE_angle', 'R_KNEE_angle',
        'L_SHOULDER_ROM', 'R_SHOULDER_ROM', 'L_WRIST_ROM', 'R_WRIST_ROM',
        'L_KNEE_ROM', 'R_KNEE_ROM', 'L_HIP_ROM', 'R_HIP_ROM',
        'exhaustion_rate', 'by_trial_exhaustion_score', 'injury_risk',
        'energy_acceleration', 'power_avg_5', 'rolling_power_std', 'rolling_hr_mean',
        'simulated_HR',
        'player_height_in_meters', 'player_weight__in_kg'
    ]
    
    default_lag_columns = [
        'joint_energy', 'joint_power',
        'L_ELBOW_energy', 'R_ELBOW_energy', 'L_WRIST_energy', 'R_WRIST_energy',
        'elbow_asymmetry', 'wrist_asymmetry',
        'L_ELBOW_angle', 'R_ELBOW_angle', 'L_WRIST_angle', 'R_WRIST_angle',
        'exhaustion_rate', 'by_trial_exhaustion_score',
        'simulated_HR'
    ]
    rolling_window = 3
    
    trial_data = prepare_joint_features(data, debug=True, group_trial=True)
    trial_data = feature_engineering(trial_data, debug=True, group_trial=True)
    
    trial_summary_data = summarize_data(data,
                                          groupby_cols=['trial_id'],
                                          lag_columns=default_lag_columns,
                                          rolling_window=rolling_window,
                                          agg_columns=default_agg_columns,
                                          global_lag=True,
                                          debug=True)
    
    shot_phase_data = prepare_joint_features(data, debug=True, group_trial=True, group_shot_phase=True)
    shot_phase_data = feature_engineering(shot_phase_data, debug=True, group_trial=True, group_shot_phase=True)
    shot_phase_summary_data = summarize_data(shot_phase_data,
                                               groupby_cols=['trial_id', 'shooting_phases'],
                                               lag_columns=default_lag_columns,
                                               rolling_window=rolling_window,
                                               agg_columns=default_agg_columns,
                                               phase_list=["arm_cock", "arm_release", "leg_cock", "wrist_release"],
                                               debug=True)
    print("Shot Phase Summary Sample:")
    print(shot_phase_summary_data.head())
    print("Trial Summary Sample:")
    print(trial_summary_data.head())
    data = check_and_drop_nulls(data, columns_to_drop=['energy_acceleration', 'exhaustion_rate'], df_name="Final Data")
    
    # Filter base data for modeling
    data.drop(columns=['event_idx_leg', 'event_idx_elbow', 'event_idx_release', 'event_idx_wrist'], inplace=True)
    data = data[data['shooting_motion'] == 1]
    invalid_count = check_for_invalid_values(data)
    if invalid_count > 0:
        logging.error("Invalid values detected in feature matrix")
        sys.exit(1)
    
    # Define the base features list and target variables.
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
    
    # Build target lists without duplication
    base_targets = ['by_trial_exhaustion_score', 'injury_risk']

    # Create joint-specific targets for each joint and side
    joints = ['ANKLE', 'WRIST', 'ELBOW', 'KNEE', 'HIP']
    joint_injury_targets = [f"{side}_{joint}_injury_risk" for joint in joints for side in ['L', 'R']]

    # Use the correct column naming pattern that exists in the dataset
    joint_exhaustion_targets = [f"{side}_{joint}_energy_by_trial_exhaustion_score" for joint in joints for side in ['L', 'R']]
    joint_targets = joint_injury_targets + joint_exhaustion_targets

    # Combine all targets
    all_targets = base_targets + joint_injury_targets + joint_exhaustion_targets

    # # Run feature importance analysis for all targets on the base dataset.
    # results = run_feature_importance_analysis(
    #     dataset=data,
    #     features=features,
    #     targets=all_targets,
    #     base_output_dir=output_dir,
    #     output_subdir="feature_lists/base",
    #     debug=debug,
    #     dataset_label="Base Data",
    #     importance_threshold=importance_threshold
    # )
    
    # Now, step by step, test that each joint-specific feature list is saved and can be loaded.
    # Build a dictionary of joint-specific feature lists for each joint target.
    joint_feature_dict = {}
    for target in joint_targets:
        try:
            features_loaded = load_top_features(target, feature_dir=base_feature_dir, df=data, n_top=10)
            logging.info(f"Test Load: Features for {target}: {features_loaded}")
            joint_feature_dict[target] = features_loaded
        except Exception as e:
            logging.error(f"Error loading features for {target}: {e}")
    # --- Aggregated Datasets: Run Feature Importance Analysis for Trial and Shot-Phase ---
    summary_features = default_agg_columns + [f"{col}_lag1" for col in default_lag_columns]
    logging.info(f"Summary features for aggregated datasets: {summary_features}")
    summary_targets = ['by_trial_exhaustion_score', 'injury_risk']
    
    # trial_results = run_feature_importance_analysis(
    #     dataset=trial_summary_data,
    #     features=summary_features,
    #     targets=summary_targets,
    #     base_output_dir=output_dir,
    #     output_subdir="feature_lists/trial_summary",
    #     debug=debug,
    #     dataset_label="Trial Summary Data",
    #     importance_threshold=importance_threshold
    # )
    
    # shot_results = run_feature_importance_analysis(
    #     dataset=shot_phase_summary_data,
    #     features=summary_features,
    #     targets=summary_targets,
    #     base_output_dir=output_dir,
    #     output_subdir="feature_lists/shot_phase_summary",
    #     debug=debug,
    #     dataset_label="Shot Phase Summary Data",
    #     importance_threshold=importance_threshold
    # )
    
    # Test loading a feature list for trial_summary (e.g., for "by_trial_exhaustion_score")
    loaded_features_trial = load_top_features("by_trial_exhaustion_score", feature_dir=trial_feature_dir, df=trial_summary_data, n_top=10)
    logging.info(f"Test Load: Trial Summary features for by_trial_exhaustion_score: {loaded_features_trial}")
    
    # Test loading a feature list for shot_phase_summary (e.g., for "by_trial_exhaustion_score")
    loaded_features_shot = load_top_features("by_trial_exhaustion_score", feature_dir=shot_feature_dir, df=shot_phase_summary_data, n_top=10)
    logging.info(f"Test Load: Shot Phase Summary features for by_trial_exhaustion_score: {loaded_features_shot}")
    
        
    # ------------------------------
    # 5. Split Base Data for Training Models
    # ------------------------------
    train_data, test_data = temporal_train_test_split(data, test_size=0.2)
    timesteps = 5

    # Hyperparameters and architecture definitions.
    hyperparams = {
        "epochs": 200,
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
    target_exhaustion = 'by_trial_exhaustion_score' # exhaustion_rate

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
    # 9. Train, Forecast, and Summarize Aggregated Models
    # ------------------------------
    # Instead of using a hard-coded summary_features list, we now load the top features
    # specific to each aggregated dataset (which were saved using the threshold filter).
    
    # --- 9a. Process Trial Summary Data ---
    trial_train_data, trial_test_data = temporal_train_test_split(trial_summary_data, test_size=0.2)
    
    # Load the dataset-specific features from the "trial_summary" folder.
    features_exhaustion_trial = load_top_features('by_trial_exhaustion_score',
                                                feature_dir=os.path.join(feature_dir, "trial_summary"),
                                                df=trial_summary_data,
                                                n_top=10)
    features_injury_trial = load_top_features('injury_risk',
                                            feature_dir=os.path.join(feature_dir, "trial_summary"),
                                            df=trial_summary_data,
                                            n_top=10)

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
    
    # Load the dataset-specific features from the "shot_phase_summary" folder.
    features_exhaustion_shot = load_top_features('by_trial_exhaustion_score',
                                                feature_dir=os.path.join(feature_dir, "shot_phase_summary"),
                                                df=shot_phase_summary_data,
                                                n_top=10)
    features_injury_shot = load_top_features('injury_risk',
                                            feature_dir=os.path.join(feature_dir, "shot_phase_summary"),
                                            df=shot_phase_summary_data,
                                            n_top=10)

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
    
