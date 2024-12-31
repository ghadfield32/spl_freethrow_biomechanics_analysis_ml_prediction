

import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Ensure DEBUG level is enabled
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Create a module-specific logger (if not already created)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Explicitly set module-specific logger to DEBUG


def check_dataset_for_smote(
    X_train, y_train, debug=False,
    imbalance_threshold=0.2, noise_threshold=0.5,
    overlap_threshold=0.3, boundary_threshold=0.4,
    extreme_imbalance_threshold=0.05
):
    """
    Analyzes a dataset to recommend the best SMOTE variant.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target labels.
        debug (bool): Whether to log debug information.
        imbalance_threshold (float): Threshold for severe imbalance.
        noise_threshold (float): Threshold for noise detection.
        overlap_threshold (float): Threshold for class overlap detection.
        boundary_threshold (float): Threshold for boundary concentration detection.
        extreme_imbalance_threshold (float): Threshold for extreme imbalance.

    Returns:
        dict: Recommendations for SMOTE variants and analysis details.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series.")

    # Step 1: Class Distribution
    class_distribution = y_train.value_counts(normalize=True)
    majority_class = class_distribution.idxmax()
    minority_class = class_distribution.idxmin()

    severe_imbalance = class_distribution[minority_class] < imbalance_threshold
    extreme_imbalance = class_distribution[minority_class] < extreme_imbalance_threshold

    if debug:
        logger.debug(f"X_train Shape: {X_train.shape}")
        logger.debug(f"Class Distribution: {class_distribution.to_dict()}")
        if extreme_imbalance:
            logging.warning(f"Extreme imbalance detected: {class_distribution[minority_class]:.2%}")

    # Step 2: Noise Analysis
    minority_samples = X_train[y_train == minority_class]
    majority_samples = X_train[y_train == majority_class]

    try:
        knn = NearestNeighbors(n_neighbors=5).fit(majority_samples)
        distances, _ = knn.kneighbors(minority_samples)
        median_distance = np.median(distances)
        noise_ratio = np.mean(distances < median_distance)
        noisy_data = noise_ratio > noise_threshold

        if debug:
            logger.debug(f"Median Distance to Nearest Neighbors: {median_distance}")
            logger.debug(f"Noise Ratio: {noise_ratio:.2%}")
    except ValueError as e:
        logging.error(f"Noise analysis error: {e}")
        noisy_data = False

    # Step 3: Overlap Analysis
    try:
        pdistances = pairwise_distances(minority_samples, majority_samples)
        overlap_metric = np.mean(pdistances < 1.0)
        overlapping_classes = overlap_metric > overlap_threshold

        if debug:
            logger.debug(f"Overlap Metric: {overlap_metric:.2%}")
    except ValueError as e:
        logging.error(f"Overlap analysis error: {e}")
        overlapping_classes = False

    # Step 4: Boundary Concentration
    try:
        boundary_ratio = np.mean(np.min(distances, axis=1) < np.percentile(distances, 25))
        boundary_concentration = boundary_ratio > boundary_threshold

        if debug:
            logger.debug(f"Boundary Concentration Ratio: {boundary_ratio:.2%}")
    except Exception as e:
        logging.error(f"Boundary concentration error: {e}")
        boundary_concentration = False

    # Step 5: Recommendations
    recommendations = []
    if severe_imbalance:
        recommendations.append("ADASYN" if not noisy_data else "SMOTEENN")
    if noisy_data:
        recommendations.append("SMOTEENN")
    if overlapping_classes:
        recommendations.append("SMOTETomek")
    if boundary_concentration:
        recommendations.append("BorderlineSMOTE")
    if not recommendations:
        recommendations.append("SMOTE")

    if debug:
        logger.debug("SMOTE Analysis Complete.")
        logger.debug(f"Recommendations: {recommendations}")

    return {
        "recommendations": recommendations,
        "details": {
            "severe_imbalance": severe_imbalance,
            "noisy_data": noisy_data,
            "overlapping_classes": overlapping_classes,
            "boundary_concentration": boundary_concentration
        }
    }

def apply_smote(X_train, y_train, recommendations, debug=False, smote_params=None):
    """
    Applies the recommended SMOTE variant to the dataset.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target labels.
        recommendations (list or str): Recommended SMOTE variants or a single SMOTE variant.
        debug (bool): Whether to log debug information.
        smote_params (dict): Parameters for SMOTE variants.

    Returns:
        pd.DataFrame, pd.Series: Resampled features and target labels.
        str: The applied SMOTE technique.
    """
    if smote_params is None:
        smote_params = {"random_state": 42}

    # Define supported SMOTE variants
    smote_variants = {
        "SMOTE": SMOTE(**smote_params),
        "ADASYN": ADASYN(**smote_params),
        "BorderlineSMOTE": BorderlineSMOTE(**smote_params),
        "SMOTEENN": SMOTEENN(**smote_params),
        "SMOTETomek": SMOTETomek(**smote_params)
    }

    # Determine SMOTE technique
    if isinstance(recommendations, list):
        if len(recommendations) == 0:
            logging.warning("Empty SMOTE recommendations. Skipping SMOTE.")
            return X_train, y_train, None
        elif len(recommendations) == 1:
            smote_technique = recommendations[0]
        else:
            logging.info("Multiple SMOTE variants recommended. Choosing the first.")
            smote_technique = recommendations[0]
    elif isinstance(recommendations, str):
        smote_technique = recommendations
    else:
        logging.error("Recommendations must be a list or string.")
        raise ValueError("Recommendations must be a list or string.")

    logger.debug(f"SMOTE Technique Requested: {smote_technique}")
    logger.debug(f"Available SMOTE Variants: {list(smote_variants.keys())}")

    # Ensure the technique exists
    if smote_technique not in smote_variants:
        logging.error(f"SMOTE variant '{smote_technique}' is not recognized. Available variants: {list(smote_variants.keys())}")
        raise KeyError(f"SMOTE variant '{smote_technique}' is not recognized.")

    smote_instance = smote_variants[smote_technique]
    X_resampled, y_resampled = smote_instance.fit_resample(X_train, y_train)

    if debug:
        logger.debug(f"Applied SMOTE Technique: {smote_technique}")
        logger.debug(f"Original X_train Shape: {X_train.shape}")
        logger.debug(f"Resampled X_train Shape: {X_resampled.shape}")
        logger.debug(f"Original Class Distribution: {Counter(y_train)}")
        logger.debug(f"Resampled Class Distribution: {Counter(y_resampled)}")

    return X_resampled, y_resampled, smote_technique


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    y_variable = "result"
    debug = False
    from ml.feature_selection.data_loader_post_select_features import load_selected_features_data
    # Example usage:
    final_ml_df_selected_features = load_selected_features_data(
        features_path='../../data/model/pipeline/final_ml_df_selected_features_columns.pkl',
        dataset_path='../../data/processed/final_ml_dataset.csv',
        y_variable='result',
        debug=False
    )

    # Assuming numerical_info_df, categorical_info_df, and final_ml_df_selected_features are already defined
    y_variable = 'result'

    print("\n[Initial Dataset Info]")
    print(f"Columns to work with: {final_ml_df_selected_features.columns.tolist()}")

    # Step 1: Split dataset into features (X) and target (y)
    X = final_ml_df_selected_features.drop(columns=[y_variable])
    y = final_ml_df_selected_features[y_variable]

    # Step 2: Train-test split
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply scaling based on suggestions
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    # Features requiring StandardScaler
    standard_features = [
        'release_ball_velocity_z', 'knee_release_angle', 'wrist_release_angle',
        'knee_max_angle', 'release_ball_direction_z', 'wrist_max_angle'
    ]

    # Features requiring MinMaxScaler
    minmax_features = [
        'elbow_max_angle', 'elbow_release_angle', 'release_ball_direction_y',
        'release_ball_speed', 'release_ball_direction_x', 'release_ball_velocity_x',
        'release_ball_velocity_y', 'calculated_release_angle'
    ]

    # Apply StandardScaler
    X_train_standard = scaler_standard.fit_transform(X_train[standard_features])
    X_test_standard = scaler_standard.transform(X_test[standard_features])

    # Apply MinMaxScaler
    X_train_minmax = scaler_minmax.fit_transform(X_train[minmax_features])
    X_test_minmax = scaler_minmax.transform(X_test[minmax_features])

    # Combine scaled features
    import pandas as pd
    X_train_scaled = pd.DataFrame(
        data=np.hstack((X_train_standard, X_train_minmax)),
        columns=standard_features + minmax_features
    )
    X_test_scaled = pd.DataFrame(
        data=np.hstack((X_test_standard, X_test_minmax)),
        columns=standard_features + minmax_features
    )

    # add in SMOTE TO TRAINING DATASETS ONLY 

    # from smote_automation import  check_dataset_for_smote, apply_smote

    # Analyze dataset for SMOTE
    smote_analysis = check_dataset_for_smote(X_train_scaled, y_train, debug=True)
    print("SMOTE Analysis Recommendations:", smote_analysis["recommendations"])

    # Apply SMOTE
    X_resampled, y_resampled, smote_used = apply_smote(X_train, 
                                                       y_train, 
                                                       "SMOTEENN", # Can also select individual: ADASYN, SMOTEENN, SMOTETomek, BorderlineSMOTE, and SMOTE
                                                       debug=True)
    print("Applied SMOTE Variant:", smote_used)
    print("Resampled Class Distribution:", Counter(y_resampled))

    logging.info(f"SMOTE Technique Used: {smote_used}")
