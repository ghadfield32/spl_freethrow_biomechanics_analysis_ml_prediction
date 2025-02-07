#../../src/freethrow_predictions/ml
"""
Automated Categorization Module
This script automates the categorization of continuous variables into bins 
with specified labels and applies transformations to multiple columns.

To use:
1. Define a bin configuration dictionary with the desired bins and labels.
2. Pass your DataFrame and configuration to the `transform_features_with_bins` function.

Author: Your Name
"""

# Imports
import pandas as pd
import numpy as np
import pickle
import logging


# Function Definitions
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


if __name__ == "__main__":
    # Debugging mode
    debug = True

    # Example bin configuration
    category_bin_config = {
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
    
    # Save the category bin configuration
    with open('../../data/model/pipeline/category_bin_config.pkl', 'wb') as f:
        pickle.dump(category_bin_config, f)

    # Load the category bin configuration
    with open('../../data/model/pipeline/category_bin_config.pkl', 'rb') as f:
        loaded_category_bin_config = pickle.load(f)

    file_path = "../../data/processed/final_ml_dataset.csv"
    #import ml dataset from spl_dataset_prep
    final_ml_df = pd.read_csv(file_path)
    

    # Step 1: Transform player features using the configuration
    categorized_columns_df = transform_features_with_bins(final_ml_df, loaded_category_bin_config, debug=debug)

    # Step 2: Combine the original DataFrame with the categorized columns
    final_ml_df_categoricals = pd.concat([final_ml_df, categorized_columns_df], axis=1)

    # Debugging output
    if debug:
        print("\nFinal DataFrame with Categorized Features:")
        print(final_ml_df_categoricals.columns)

