
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from pathlib import Path
from joblib import Parallel, delayed
from ml.load_and_prepare_data.load_data_and_analyze import (
    load_data, prepare_joint_features, feature_engineering, summarize_data, check_and_drop_nulls
)
    
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def prepare_base_datasets(csv_path, json_path, debug=False):
    data = load_data(csv_path, json_path, debug=debug)
    data = prepare_joint_features(data, debug=debug)
    data = feature_engineering(data, debug=debug)
    data = check_and_drop_nulls(data,
                                columns_to_drop=['energy_acceleration', 'exhaustion_rate'],
                                df_name="Final Data")
    trial = prepare_joint_features(data, debug=debug, group_trial=True)
    trial = feature_engineering(trial, debug=debug, group_trial=True)
    trial_summary = summarize_data(
        data,
        groupby_cols=['trial_id'],
        lag_columns=[
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
        ],
        rolling_window=3,
        agg_columns=[
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
        ],
        global_lag=True,
        debug=debug
    )
    shot = prepare_joint_features(data, debug=debug, group_trial=True, group_shot_phase=True)
    shot = feature_engineering(shot, debug=debug, group_trial=True, group_shot_phase=True)
    shot_summary = summarize_data(
        shot,
        groupby_cols=['trial_id', 'shooting_phases'],
        lag_columns=[
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
        ],
        rolling_window=3,
        agg_columns=[
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
        ],
        phase_list=["arm_cock", "arm_release", "leg_cock", "wrist_release"],
        debug=debug
    )
    data = check_and_drop_nulls(data, columns_to_drop=['energy_acceleration', 'exhaustion_rate'], df_name="Final Data")
    return data, trial_summary, shot_summary





def validate_features(features, df, context):
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logging.error(f"{context} - Missing features: {missing_features}")
        raise ValueError(f"{context} - Missing features: {missing_features}")


def select_top_n_features_from_df(features_df, n_top=5, sort_by="Consensus_Rank", ascending=True):
    sorted_df = features_df.sort_values(by=sort_by, ascending=ascending)
    top_features = sorted_df.head(n_top)['Feature'].tolist()
    return top_features


def save_top_features(results, output_dir="feature_lists/base", importance_threshold=0.5, n_top=10):
    """
    Saves the top features that exceed the importance threshold to pickle files for each target.
    If the filtering yields fewer than n_top features, falls back to selecting the top n_top based on Consensus_Rank.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for target, (combined, _) in results.items():
        combined['Consensus_Rank'] = (
            combined['Perm_Importance'].rank(ascending=False) +
            combined['RFE_Rank'] +
            combined['SHAP_Importance'].rank(ascending=False)
        )
        logging.debug(f"[{target}] Combined shape before filtering: {combined.shape}")
        filtered_features = combined[
            (combined['Perm_Importance'] > importance_threshold) &
            (combined['SHAP_Importance'] > importance_threshold)
        ].sort_values("Consensus_Rank")
        logging.debug(f"[{target}] Shape after threshold filtering: {filtered_features.shape}")
        if len(filtered_features) < n_top:
            logging.warning(f"[{target}] Only {len(filtered_features)} features exceeded the thresholds; falling back to top {n_top} by Consensus_Rank.")
            filtered_features = combined.sort_values("Consensus_Rank", ascending=True).head(n_top)
        else:
            filtered_features = filtered_features.head(n_top)
        top_features = filtered_features['Feature'].tolist()
        logging.debug(f"[{target}] Top features to be saved: {top_features}")
        filename = Path(output_dir) / f"{target}_model_feature_list.pkl"
        pd.to_pickle(top_features, filename)
        logging.info(f"✅ {target}: Saved features at {filename}. Top features: {top_features}")



def load_top_features(target, feature_dir, df, n_top=10):
    filename = Path(feature_dir) / f"{target}_model_feature_list.pkl"
    logging.debug(f"Attempting to load feature list for target '{target}' from file: {filename}")
    if not filename.exists():
        logging.error(f"Feature list for '{target}' not found at {filename}")
        dir_contents = list(Path(feature_dir).iterdir())
        logging.error(f"Contents of {feature_dir}: {dir_contents}")
        sys.exit(1)
    features = pd.read_pickle(filename)
    logging.info(f"Loaded feature list for '{target}': {features}")
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logging.error(f"Features missing in DataFrame for {target}: {missing_features}")
        raise KeyError(f"{missing_features} not in index")
    if n_top is not None:
        features = features[:n_top]
    return features


def perform_feature_importance_analysis(data, features, target, n_features_to_select=5, debug=False):
    filtered_features = [f for f in features if f in data.columns]
    missing = [f for f in features if f not in data.columns]
    if missing:
        logging.warning(f"The following features are missing and will be ignored: {missing}")
    features = filtered_features
    X = data[features].fillna(method='ffill').fillna(method='bfill')
    y = data[target].fillna(method='ffill').fillna(method='bfill')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        'Feature': features,
        'Perm_Importance': perm_result.importances_mean
    })
    from sklearn.feature_selection import RFE
    rfe_selector = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
    rfe_selector.fit(X_train, y_train)
    rfe_df = pd.DataFrame({
        'Feature': features,
        'RFE_Rank': rfe_selector.ranking_,
        'RFE_Support': rfe_selector.support_
    })
    combined = perm_df.merge(rfe_df, on='Feature')
    explainer = shap.TreeExplainer(rf)
    sample_size = min(100, X_test.shape[0])
    X_test_sampled = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_test_sampled)
    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': features,
        'SHAP_Importance': shap_abs
    })
    combined = combined.merge(shap_df, on='Feature')
    if debug:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Perm_Importance', y='Feature', data=combined.nlargest(20, 'Perm_Importance'))
        plt.title('Top Permutation Importances')
        plt.xlabel('Mean Permutation Importance')
        plt.ylabel('Feature')
        plt.show()
        shap.summary_plot(shap_values, X_test_sampled, plot_type="bar", max_display=20)
    return combined, rf


def analyze_and_display_top_features(results, n_top=5):
    for target, (combined, _) in results.items():
        print(f"\n=== Feature Analysis for Target: {target.upper()} ===")
        perm_top = combined.nlargest(n_top, 'Perm_Importance')['Feature'].tolist()
        rfe_top = combined[combined['RFE_Support']]['Feature'].tolist()
        shap_top = combined.nlargest(n_top, 'SHAP_Importance')['Feature'].tolist()
        combined['Consensus_Rank'] = (
            combined['Perm_Importance'].rank(ascending=False) +
            combined['RFE_Rank'] +
            combined['SHAP_Importance'].rank(ascending=False)
        )
        consensus_top = combined.nsmallest(n_top, 'Consensus_Rank')['Feature'].tolist()
        print(f"Permutation Top {n_top}: {perm_top}")
        print(f"RFE Selected Features: {rfe_top}")
        print(f"SHAP Top {n_top}: {shap_top}")
        print(f"Consensus Top {n_top}: {consensus_top}")


def check_for_invalid_values(df):
    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = numeric_df.isin([np.inf, -np.inf])
    if inf_mask.any().any():
        logging.error(f"Found infinite values in columns: {numeric_df.columns[inf_mask.any()].tolist()}")
    na_mask = numeric_df.isna()
    if na_mask.any().any():
        logging.error(f"Found NA values in columns: {numeric_df.columns[na_mask.any()].tolist()}")
    extreme_mask = (numeric_df.abs() > 1e30).any(axis=1)
    if extreme_mask.any():
        logging.error(f"Found extreme values (>1e30) in rows: {numeric_df.index[extreme_mask].tolist()}")
    return inf_mask.sum().sum() + na_mask.sum().sum() + extreme_mask.sum()


def analyze_joint_injury_features(results, joint, output_dir, n_top=10, importance_threshold=0.0):
    joint_keys = [key for key in results if f"_{joint}_injury_risk" in key]
    if not joint_keys:
        logging.warning(f"No injury models found for joint: {joint}")
        return [], None
    df_list = []
    for key in joint_keys:
        combined_df, _ = results[key]
        df_list.append(combined_df.copy())
    concat_df = pd.concat(df_list, axis=0)
    agg_df = concat_df.groupby("Feature", as_index=False).agg({
        'Perm_Importance': 'mean',
        'SHAP_Importance': 'mean',
        'RFE_Rank': 'mean',
        'RFE_Support': 'max'
    })
    agg_df['Consensus_Rank'] = (
        agg_df['Perm_Importance'].rank(ascending=False) +
        agg_df['RFE_Rank'].rank(ascending=True) +
        agg_df['SHAP_Importance'].rank(ascending=False)
    )
    agg_df = agg_df.sort_values("Consensus_Rank")
    top_n = agg_df.nsmallest(n_top, "Consensus_Rank")
    filtered_top = top_n[
        (top_n['Perm_Importance'] > importance_threshold) &
        (top_n['SHAP_Importance'] > importance_threshold)
    ]
    top_features = filtered_top['Feature'].tolist()
    filename = Path(output_dir) / f"{joint}_aggregated_top_features.pkl"
    pd.to_pickle(top_features, filename)
    logging.info(f"Aggregated and saved top features for joint {joint} at {filename}: {top_features}")
    return top_features, agg_df


def run_feature_importance_analysis(dataset, features, targets, base_output_dir, output_subdir, 
                                    debug=False, dataset_label="Dataset", importance_threshold=0.0, n_top=10):
    features = [f for f in features if f in dataset.columns]
    missing_targets = [t for t in targets if t not in dataset.columns]
    if missing_targets:
        logging.error(f"{dataset_label} missing targets: {missing_targets}")
        sys.exit(1)
    results = dict(zip(
        targets,
        Parallel(n_jobs=-1)(delayed(perform_feature_importance_analysis)(dataset, features, target, debug=debug)
                             for target in targets)
    ))
    analyze_and_display_top_features(results, n_top=n_top)
    output_path = Path(base_output_dir) / output_subdir
    save_top_features(results, output_dir=str(output_path), importance_threshold=importance_threshold, n_top=n_top)


def run_feature_import_and_load_top_features(dataset, features, targets, base_output_dir, output_subdir,
                                             debug=False, dataset_label="Dataset", importance_threshold=0.0, n_top=10,
                                             run_analysis=True):
    output_path = Path(base_output_dir) / output_subdir
    if run_analysis:
        logging.info(f"Running feature importance analysis for {dataset_label}.")
        run_feature_importance_analysis(
            dataset=dataset,
            features=features,
            targets=targets,
            base_output_dir=base_output_dir,
            output_subdir=output_subdir,
            debug=debug,
            dataset_label=dataset_label,
            importance_threshold=importance_threshold,
            n_top=n_top
        )
    else:
        logging.info(f"Skipping analysis for {dataset_label}; using pre-saved feature lists from {output_path}")
    loaded_features = {}
    for t in targets:
        try:
            loaded = load_top_features(t, feature_dir=output_path, df=dataset, n_top=n_top)
            loaded_features[t] = loaded
            logging.info(f"[{dataset_label}] Loaded top features for target '{t}': {loaded}")
        except Exception as e:
            logging.error(f"Error loading features for target '{t}': {e}")
    return loaded_features





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
    exhaustion_rate_list = trial_loaded_features.get('exhaustion_rate', [])
    injury_risk_list = trial_loaded_features.get('injury_risk', [])

    
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
    exhaustion_rate_list = shot_loaded_features.get('exhaustion_rate', [])
    injury_risk_list = shot_loaded_features.get('injury_risk', [])
    
