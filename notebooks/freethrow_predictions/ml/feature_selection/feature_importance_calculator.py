import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import pickle
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

def calculate_feature_importance(
    df: pd.DataFrame,
    target_variable: Union[str, List[str]],
    n_estimators: int = 100,
    random_state: int = 42,
    debug: bool = False
) -> pd.DataFrame:
    """
    Calculates feature importance using a Random Forest model.
    
    Args:
        df (DataFrame): Input DataFrame.
        target_variable (str or list of str): Target column name or a list with a single target column.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed.
        debug (bool): If True, prints debugging information.
    
    Returns:
        DataFrame: Feature importances.
    
    Note:
        If target_variable is passed as a list, it must contain only one element.
    """
    # Normalize the target variable: if provided as a list, extract the single string.
    if isinstance(target_variable, list):
        if len(target_variable) == 1:
            target = target_variable[0]
        else:
            raise ValueError("calculate_feature_importance supports only a single target variable.")
    else:
        target = target_variable

    # Drop the target column from predictors and extract the target variable column.
    X = df.drop(columns=[target])
    y = df[target]
    
    # If y is a DataFrame (e.g. if someone mistakenly passed a list with more than one column),
    # you might need to extract a Series. Here, we assume y is a Series.
    if isinstance(y, pd.DataFrame):
        # If a single-column DataFrame, convert to Series.
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            raise ValueError("The target variable DataFrame should contain only one column.")

    # Encode target variable if necessary
    if y.dtype == 'object' or str(y.dtype) == 'category':
        if debug:
            print(f"Target variable '{target}' is categorical. Encoding labels.")
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
        X_encoded_df = pd.DataFrame(
            X_encoded,
            columns=ohe.get_feature_names_out(categorical_cols),
            index=X.index
        )
        X = pd.concat([X[numeric_cols], X_encoded_df], axis=1)

    # Select the model based on the type of y
    model = (
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if y.dtype in ['int64', 'float64'] else
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    )

    if debug:
        print(f"Training Random Forest model with {n_estimators} estimators...")

    model.fit(X, y)

    # Calculate feature importances and return them as a DataFrame.
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
    y_variable: Optional[Union[str, List[str]]] = None,
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
        y_variable (str or list of str, optional): Target variable.
        paths (dict, optional): Dictionary mapping each item to its file path.
    
    Returns:
        If mode is 'load', returns a dictionary with loaded items.
        If mode is 'save', returns None.
    """
    # Define default paths if not provided.
    default_paths = {
        'features': 'final_ml_df_selected_features_columns_test.pkl',
        'ordinal_categoricals': 'ordinal_categoricals.pkl',
        'nominal_categoricals': 'nominal_categoricals.pkl',
        'numericals': 'numericals.pkl',
        'y_variable': 'y_variable.pkl'
    }

    # Update default paths with any provided paths.
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

            # Iterate and save each item.
            for key, path in default_paths.items():
                with open(path, 'wb') as f:
                    pickle.dump(data_to_save[key], f)
                print(f"✅ {key.replace('_', ' ').capitalize()} saved to {path}")

        elif mode == 'load':
            loaded_data = {}

            # Iterate and load each item.
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
    # For testing purposes: load dataset and run feature importance.

    final_ml_features_path = '../../data/model/pipeline/final_ml_df_selected_features_columns_test.pkl'
    # (Other paths omitted for brevity.)

    file_path = "../../data/processed/final_ml_dataset.csv"
    final_ml_df = pd.read_csv(file_path)

    # For demonstration, set the target variable as a list.
    target_variable = ['result']
    correlation_threshold = 0.8
    debug = True

    # (Assume check_multicollinearity is defined elsewhere.)
    from ml.feature_selection.multicollinearity_checker import check_multicollinearity

    print("\nChecking for Multicollinearity...")
    multicollinearity_df = check_multicollinearity(final_ml_df, threshold=correlation_threshold, debug=debug)

    if multicollinearity_df.empty:
        print("No multicollinearity issues detected.")
    else:
        for _, row in multicollinearity_df.iterrows():
            print(f"High correlation ({row['Correlation']}) between '{row['Feature1']}' and '{row['Feature2']}'.")

    # Calculate feature importance.
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
    nominal_categoricals = [] #'player_estimated_hand_length_cm_category'
    numericals = [        'release_ball_direction_x' ,'release_ball_direction_z', 'release_ball_direction_y',
        'elbow_release_angle', 'elbow_max_angle',
        'wrist_release_angle', 'wrist_max_angle',
        'knee_release_angle', 'knee_max_angle',
        'release_ball_speed',
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
