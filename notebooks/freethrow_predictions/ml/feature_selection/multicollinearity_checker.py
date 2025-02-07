
import pandas as pd
import numpy as np

def check_multicollinearity(df, threshold=0.8, debug=False):
    """
    Identifies pairs of features with correlation above the specified threshold.
    Args:
        df (DataFrame): DataFrame containing numerical features.
        threshold (float): Correlation coefficient threshold.
        debug (bool): If True, prints debugging information.
    Returns:
        DataFrame: Pairs of features with high correlation.
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if debug:
        print(f"Computing correlation matrix for {len(numeric_df.columns)} numerical features...")

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify highly correlated features
    highly_correlated = [
        (column, idx, upper.loc[column, idx])
        for column in upper.columns
        for idx in upper.index
        if (upper.loc[column, idx] > threshold)
    ]

    multicollinearity_df = pd.DataFrame(highly_correlated, columns=['Feature1', 'Feature2', 'Correlation'])

    if debug:
        if not multicollinearity_df.empty:
            print(f"Found {len(multicollinearity_df)} pairs of highly correlated features:")
            print(multicollinearity_df)
        else:
            print("No highly correlated feature pairs found.")

    return multicollinearity_df




if __name__ == "__main__":
    import pickle
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

    # Remove columns to address collinearity
    drop_features = [ 'L_KNEE_min_power', 'L_HIP_max_power']
    
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
