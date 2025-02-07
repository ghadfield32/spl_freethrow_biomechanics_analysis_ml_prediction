
import logging
import pandas as pd
from typing import Optional

class ShapFeedbackHandler:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the ShapFeedbackHandler with an optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def expand_specific_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the 'specific_feedback' column from dictionaries to separate columns.

        Enhanced Steps:
        1. Apply pd.Series to 'specific_feedback' to create separate columns.
        2. Identify numeric, categorical, and classification columns.
        3. Convert numeric columns to float, coercing errors to NaN.
        4. Fill missing values in categorical and classification columns with appropriate placeholders.
        5. Assign default units ('units') where unit information is missing.
        6. Merge the expanded feedback columns back into the original DataFrame.

        :param df: DataFrame containing 'specific_feedback' column.
        :return: DataFrame with expanded 'shap_' columns.
        """
        if 'specific_feedback' not in df.columns:
            self.logger.error("'specific_feedback' column not found in DataFrame.")
            raise KeyError("'specific_feedback' column not found.")
        
        self.logger.info("Expanding 'specific_feedback' into separate columns.")
        try:
            feedback_df = df['specific_feedback'].apply(pd.Series)
            self.logger.debug(f"Feedback DataFrame shape after expansion: {feedback_df.shape}")
            
            # Identify column types based on suffixes
            numeric_cols = [
                col for col in feedback_df.columns
                if 'unit_change' in col or 'goal' in col or 'min' in col or 'max' in col or 'importance' in col
            ]
            categorical_cols = [
                col for col in feedback_df.columns
                if 'direction' in col or 'impact' in col
            ]
            classification_cols = [
                col for col in feedback_df.columns
                if 'classification' in col
            ]
            unit_cols = [
                col for col in feedback_df.columns
                if 'unit' in col and 'goal' not in col  # Avoid '_unit_goal' if exists
            ]

            # Convert numeric columns to float
            for col in numeric_cols:
                feedback_df[col] = pd.to_numeric(feedback_df[col], errors='coerce')
                bad_count = feedback_df[col].isna().sum()
                self.logger.debug(f"Column '{col}': {bad_count} rows failed numeric parse => NaN.")

            # Fill missing in categorical columns
            for col in categorical_cols:
                feedback_df[col] = feedback_df[col].fillna('No feedback available')

            # Fill missing in classification columns
            for col in classification_cols:
                feedback_df[col] = feedback_df[col].fillna('No data')

            # Fill missing in unit columns with 'units'
            for col in unit_cols:
                feedback_df[col] = feedback_df[col].fillna('units')

            # Merge back into the original DataFrame without the 'specific_feedback' column
            df_expanded = pd.concat([df.drop(columns=['specific_feedback']), feedback_df], axis=1)
            self.logger.debug(f"Final dtypes after expansion:\n{df_expanded.dtypes}")
            self.logger.info("'specific_feedback' expanded successfully.")
            return df_expanded
        except Exception as e:
            self.logger.error(f"Failed to expand 'specific_feedback': {e}")
            raise

if __name__ == "__main__":
    print("ShapFeedbackHandler class for expanding specific feedback into separate columns.")
