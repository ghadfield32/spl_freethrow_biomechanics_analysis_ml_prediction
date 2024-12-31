
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import pandas as pd
import pickle
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

def calculate_feature_importance(df, target_variable, n_estimators=100, random_state=42, debug=False):
    """
    Calculates feature importance using a Random Forest model.
    Args:
        df (DataFrame): Input DataFrame.
        target_variable (str): Target column name.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed.
        debug (bool): If True, prints debugging information.
    Returns:
        DataFrame: Feature importances.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Encode target variable if necessary
    if y.dtype == 'object' or str(y.dtype) == 'category':
        if debug:
            print(f"Target variable '{target_variable}' is categorical. Encoding labels.")
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Separate categorical and numerical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    if debug:
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numeric_cols}")

    # Encode categorical features if present
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = ohe.fit_transform(X[categorical_cols])
        X_encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=X.index)
        X = pd.concat([X[numeric_cols], X_encoded_df], axis=1)

    # Select model
    model = (
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if y.dtype in ['int64', 'float64'] else
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    )

    if debug:
        print(f"Training Random Forest model with {n_estimators} estimators...")

    model.fit(X, y)

    # Calculate feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    if debug:
        print("Feature Importances:")
        print(feature_importances)

    return feature_importances


def manage_features(
    mode: str,
    features_df: Optional[pd.DataFrame] = None,
    ordinal_categoricals: Optional[List[str]] = None,
    nominal_categoricals: Optional[List[str]] = None,
    numericals: Optional[List[str]] = None,
    y_variable: Optional[str] = None,
    paths: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Save or load features and metadata.

    Parameters:
        mode (str): Operation mode - 'save' or 'load'.
        features_df (pd.DataFrame, optional): DataFrame containing features to save (required for 'save').
        ordinal_categoricals (list, optional): List of ordinal categorical features.
        nominal_categoricals (list, optional): List of nominal categorical features.
        numericals (list, optional): List of numerical features.
        y_variable (str, optional): Target variable.
        paths (dict, optional): Dictionary mapping each item to its file path.

    Returns:
        If mode is 'load', returns a dictionary with loaded items.
        If mode is 'save', returns None.
    """

    # Define default paths if not provided
    default_paths = {
        'features': 'final_ml_df_selected_features_columns_test.pkl',
        'ordinal_categoricals': 'ordinal_categoricals.pkl',
        'nominal_categoricals': 'nominal_categoricals.pkl',
        'numericals': 'numericals.pkl',
        'y_variable': 'y_variable.pkl'
    }

    # Update default paths with any provided paths
    if paths:
        default_paths.update(paths)

    try:
        if mode == 'save':
            if features_df is None:
                raise ValueError("features_df must be provided in 'save' mode.")

            # Prepare data to save
            data_to_save = {
                'features': features_df.columns.tolist(),
                'ordinal_categoricals': ordinal_categoricals,
                'nominal_categoricals': nominal_categoricals,
                'numericals': numericals,
                'y_variable': y_variable
            }

            # Iterate and save each item
            for key, path in default_paths.items():
                with open(path, 'wb') as f:
                    pickle.dump(data_to_save[key], f)
                print(f"✅ {key.replace('_', ' ').capitalize()} saved to {path}")

        elif mode == 'load':
            loaded_data = {}

            # Iterate and load each item
            for key, path in default_paths.items():
                with open(path, 'rb') as f:
                    loaded_data[key] = pickle.load(f)
                print(f"✅ {key.replace('_', ' ').capitalize()} loaded from {path}")

            return loaded_data

        else:
            raise ValueError("Mode should be either 'save' or 'load'.")

    except Exception as e:
        print(f"❌ Error during '{mode}' operation: {e}")
        if mode == 'load':
            return {key: None for key in default_paths.keys()}
        

if __name__ == "__main__":
    # from feature_selection.multicollinearity_checker import check_multicollinearity

    final_ml_features_path = '../../data/model/pipeline/final_ml_df_selected_features_columns_test.pkl'
    final_ml_ordinal_categoricals_path = '../../data/model/pipeline/features_info/ordinal_categoricals.pkl'
    final_ml_nominal_categoricals_path = '../../data/model/pipeline/features_info/nominal_categoricals.pkl'
    final_ml_numericals_features_path = '../../data/model/pipeline/features_info/numericals.pkl'
    final_ml_y_variable_pkl_path = '../../data/model/pipeline/features_info/y_variable.pkl'
    final_ml_dataset_path = ''
    
    # Load the category bin configuration
    with open('../../data/model/pipeline/category_bin_config.pkl', 'rb') as f:
        loaded_category_bin_config = pickle.load(f)

    file_path = "../../data/processed/final_ml_dataset.csv"
    #import ml dataset from spl_dataset_prep
    final_ml_df = pd.read_csv(file_path)
    
    # Feature selection based on multi collinearity and random forest importance selection
    target_variable = 'result'
    correlation_threshold = 0.8
    debug = True

    
    # Step 1: Check for multicollinearity
    print("\nChecking for Multicollinearity...")
    multicollinearity_df = check_multicollinearity(final_ml_df, threshold=correlation_threshold, debug=debug)

    # Step 2: Handle multicollinearity
    if not multicollinearity_df.empty:
        for index, row in multicollinearity_df.iterrows():
            feature1, feature2, correlation = row['Feature1'], row['Feature2'], row['Correlation']
            print(f"High correlation ({correlation}) between '{feature1}' and '{feature2}'.")
    else:
        print("No multicollinearity issues detected.")


    # Remove columns to address collinearity
    drop_features = [
        'trial_id', 'player_participant_id', 'landing_y', 'landing_x', 'entry_angle', 'shot_id',
        'L_KNEE_avg_power', 'L_WRIST_energy_std', 'R_WRIST_energy_max', 
        'R_ANKLE_energy_mean', 'R_5THFINGER_energy_std', 'R_KNEE_avg_power', 'L_1STFINGER_max_power', 
        'L_5THFINGER_energy_max', 'L_WRIST_max_power', 'R_HIP_energy_std', 'L_1STFINGER_energy_max', 
        'R_ANKLE_energy_max', 'R_ELBOW_energy_max', 'R_ANKLE_energy_std', 'L_WRIST_energy_max', 
        'player_estimated_hand_length_cm', 'player_estimated_standing_reach_cm', 
        'player_estimated_wingspan_cm', 'player_weight__in_kg', 'L_KNEE_energy_std', 'L_HIP_energy_max', 
        'L_ANKLE_energy_max', 'L_WRIST_std_power', 'L_ELBOW_std_power', 'R_KNEE_max_power', 
        'L_ELBOW_avg_power', 'R_ELBOW_min_power', 'L_WRIST_min_power', 'R_HIP_energy_mean', 
        'L_ELBOW_energy_max', 'L_ELBOW_min_power', 'R_1STFINGER_min_power', 'L_ANKLE_min_power', 
        'L_1STFINGER_avg_power', 'R_ANKLE_std_power', 'R_5THFINGER_avg_power', 'L_1STFINGER_energy_mean', 
        'R_HIP_max_power', 'R_WRIST_avg_power', 'R_ELBOW_energy_mean', 'L_WRIST_avg_power', 
        'L_1STFINGER_std_power', 'L_KNEE_energy_max', 'L_WRIST_energy_mean', 'R_KNEE_energy_std', 
        'L_HIP_energy_std', 'L_KNEE_energy_mean', 'R_WRIST_energy_mean', 'L_ELBOW_max_power', 
        'R_WRIST_energy_std', 'L_ANKLE_std_power', 'L_HIP_energy_mean', 'L_ELBOW_energy_mean', 
        'R_HIP_avg_power', 'L_HIP_std_power', 'R_KNEE_std_power', 'L_ANKLE_energy_std', 
        'release_frame_time', 'R_ANKLE_avg_power', 'L_ANKLE_max_power', 'L_5THFINGER_energy_std', 
        'R_WRIST_min_power', 'R_1STFINGER_energy_mean', 'R_ELBOW_energy_std', 'R_HIP_std_power', 
        'R_KNEE_energy_max', 'R_WRIST_std_power', 'L_1STFINGER_energy_std', 'L_HIP_avg_power', 
        'R_5THFINGER_energy_mean', 'R_ANKLE_max_power', 'L_ANKLE_avg_power', 'R_5THFINGER_max_power', 
        'R_5THFINGER_energy_max', 'L_5THFINGER_min_power', 'L_ELBOW_energy_std', 
        'R_1STFINGER_energy_max', 'R_KNEE_min_power', 'R_1STFINGER_energy_std', 
        'R_5THFINGER_std_power', 'L_1STFINGER_min_power', 'R_ELBOW_max_power', 'L_HIP_min_power', 
        'L_5THFINGER_std_power', 'R_1STFINGER_max_power', 'R_KNEE_energy_mean', 'L_5THFINGER_avg_power', 
        'L_5THFINGER_max_power', 'R_HIP_min_power', 'L_KNEE_max_power', 'R_5THFINGER_min_power', 
        'R_1STFINGER_std_power', 'R_ELBOW_avg_power', 'L_ANKLE_energy_mean', 'R_ELBOW_std_power', 
        'L_5THFINGER_energy_mean', 'R_1STFINGER_avg_power', 'R_HIP_energy_max', 'L_KNEE_std_power',
        'R_ANKLE_min_power', 'L_KNEE_min_power', 'L_HIP_max_power'
    ]
    
    # Step 2: Handle multicollinearity
    if not final_ml_df.empty:

            # Drop or combine features based on criteria
            # Example decision logic here...
            # drop_features = ['trial_id', 'player_participant_id']
            # # Drop the identified features from the dataset
            # Drop the identified features from the dataset
            final_ml_df = final_ml_df.drop(columns=drop_features, errors='ignore')

            print(f"Dropped {len(drop_features)} features: {', '.join(drop_features)}")
    else:
        print("No multicollinearity issues detected.")

    # Step 3: Calculate feature importance
    print("\nCalculating Feature Importance...")
    feature_importances = calculate_feature_importance(
        final_ml_df, target_variable=target_variable, n_estimators=100, random_state=42, debug=debug
    )

    print("\nFinal Feature Importances:")
    print(feature_importances.to_string(index=False))
    
    
    #Final Decisions: 
    # Features recommended for dropping
    features_to_drop = [
        'peak_height_relative'
    ]
    print(f"Dropped features (for redundancy or multicollinearity): {', '.join(features_to_drop)}")
    

    # Define categories and column names
    ordinal_categoricals = []
    nominal_categoricals = ['player_estimated_hand_length_cm_category']
    numericals = [        'release_ball_direction_x' ,'release_ball_direction_z', 'release_ball_direction_y',
        'elbow_release_angle', 'elbow_max_angle',
        'wrist_release_angle', 'wrist_max_angle',
        'knee_release_angle', 'knee_max_angle',
        'result', 'release_ball_speed',
        'release_ball_velocity_x', 'release_ball_velocity_y','release_ball_velocity_z']
    y_variable = ['result']
    final_keep_list = ordinal_categoricals + nominal_categoricals + numericals + y_variable
    
    # Apply the filter to keep only these columns
    final_ml_df_selected_features = final_ml_df[final_keep_list]
    print(f"Retained {len(final_keep_list)} features: {', '.join(final_keep_list)}")

    # Save feature names to a file
    with open(final_ml_features_path, 'wb') as f:
        pickle.dump(final_ml_df_selected_features.columns.tolist(), f)

    print(f"Retained {len(final_keep_list)} features: {', '.join(final_keep_list)}")

    # Define paths (optional, will use defaults if not provided)
    paths = {
        'features': '../../data/model/pipeline/final_ml_df_selected_features_columns_test.pkl',
        'ordinal_categoricals': '../../data/model/pipeline/features_info/ordinal_categoricals.pkl',
        'nominal_categoricals': '../../data/model/pipeline/features_info/nominal_categoricals.pkl',
        'numericals': '../../data/model/pipeline/features_info/numericals.pkl',
        'y_variable': '../../data/model/pipeline/features_info/y_variable.pkl'
    }
    # Save features and metadata
    manage_features(
        mode='save',
        features_df=final_ml_df,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        y_variable=y_variable,
        paths=paths
    )
    
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
