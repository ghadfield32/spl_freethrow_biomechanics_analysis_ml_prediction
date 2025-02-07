
"""
Automated Categorization Module
This script automates the categorization of continuous variables into bins 
with specified labels and applies transformations to multiple columns.

To use:
1. Define a bin configuration dictionary with the desired bins and labels.
2. Pass your DataFrame and configuration to the `transform_features_with_bins` function.

Author: Your Name
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os

def categorize_column(df, column_name, bins, labels, new_column_name=None, debug=False):
    """
    Categorizes a column into bins with specified labels.

    Args:
        df (DataFrame): The dataset to transform.
        column_name (str): Name of the column to bin.
        bins (list): Bin edges for categorization.
        labels (list): Labels corresponding to each bin.
        new_column_name (str): Optional; name of the new column. Defaults to "<column_name>_category".
        debug (bool): If True, prints debugging information.

    Returns:
        Series: The newly categorized column as a pandas Series.
    """
    try:
        if new_column_name is None:
            new_column_name = f"{column_name}_category"

        # Apply binning
        categorized_column = pd.cut(df[column_name], bins=bins, labels=labels)

        if debug:
            print(f"\nBinning applied to '{column_name}' -> New column: '{new_column_name}'")
            print(pd.DataFrame({column_name: df[column_name], new_column_name: categorized_column}).head())

        return categorized_column
    except KeyError:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return pd.Series(index=df.index)  # Return an empty series if the column is missing
    except Exception as e:
        print(f"Unexpected error while categorizing '{column_name}': {e}")
        return pd.Series(index=df.index)  # Return an empty series if there's an error

def transform_features_with_bins(df, bin_config, debug=False):
    """
    Applies binning transformations to multiple columns based on the provided configuration.

    Args:
        df (DataFrame): The dataset to transform.
        bin_config (dict): Configuration dictionary where keys are column names and values are
                           dictionaries with 'bins', 'labels', and optionally 'new_column_name'.
        debug (bool): If True, prints debugging information.

    Returns:
        DataFrame: A new DataFrame containing only the categorized columns.
    """
    categorized_df = pd.DataFrame(index=df.index)  # Initialize an empty DataFrame with the same index
    for column, config in bin_config.items():
        bins = config['bins']
        labels = config['labels']
        new_column_name = config.get('new_column_name', f"{column}_category")  # Default new column name
        categorized_df[new_column_name] = categorize_column(df, column, bins, labels, debug=debug)

    return categorized_df

def load_default_bin_config(config_path=None):
    """
    Loads the default bin configuration. If not present, creates and saves a default configuration.

    Args:
        config_path (str): Path to save/load the bin configuration. Defaults to '../../data/model/pipeline/category_bin_config.pkl'.

    Returns:
        dict: The bin configuration dictionary.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../data/model/pipeline/category_bin_config.pkl')
    
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            bin_config = pickle.load(f)
    else:
        # Define default bin configuration
        bin_config = {
            'player_height_in_meters': {
                'bins': [0, 1.80, 2.00, np.inf],
                'labels': ["Short", "Average", "Tall"]
            },
            'player_weight__in_kg': {
                'bins': [0, 75, 95, np.inf],
                'labels': ["Lightweight", "Average", "Heavy"]
            },
            'player_estimated_wingspan_cm': {
                'bins': [0, 190, 220, np.inf],
                'labels': ["Small", "Medium", "Large"]
            },
            'player_estimated_standing_reach_cm': {
                'bins': [0, 230, 250, np.inf],
                'labels': ["Short", "Average", "Tall"]
            },
            'player_estimated_hand_length_cm': {
                'bins': [0, 20, 25, np.inf],
                'labels': ["Small", "Medium", "Large"]
            }
        }
        # Save the default bin configuration
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'wb') as f:
            pickle.dump(bin_config, f)
    
    return bin_config

if __name__ == "__main__":
    # Example usage for testing
    debug = True

    # Define the path to save the bin configuration
    config_path = '../../data/model/pipeline/category_bin_config.pkl'

    # Load the category bin configuration
    bin_config = load_default_bin_config(config_path=config_path)

    file_path = "../../data/processed/final_ml_dataset.csv"
    final_ml_df = pd.read_csv(file_path)

    # Transform player features using the configuration
    categorized_columns_df = transform_features_with_bins(final_ml_df, bin_config, debug=debug)

    # Combine the original DataFrame with the categorized columns
    final_ml_df_categoricals = pd.concat([final_ml_df, categorized_columns_df], axis=1)

    # Debugging output
    if debug:
        print("\nFinal DataFrame with Categorized Features:")
        print(final_ml_df_categoricals.columns)
