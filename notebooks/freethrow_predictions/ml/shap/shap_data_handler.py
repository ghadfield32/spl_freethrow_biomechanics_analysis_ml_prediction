
import logging
import pickle
from pathlib import Path
from typing import Any
import shap

class ShapDataHandler:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the ShapDataHandler with an optional logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def save_shap_values(self, shap_values: Any, save_path: Path):
        """
        Save SHAP values to a file using pickle.
        """
        self.logger.info(f"Saving SHAP values to {save_path}...")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(shap_values, f)
            self.logger.info(f"SHAP values saved successfully to {save_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save SHAP values: {e}")
            raise

    def load_shap_values(self, load_path: Path) -> Any:
        """
        Load SHAP values from a pickle file.
        """
        self.logger.info(f"Loading SHAP values from {load_path}...")
        try:
            with open(load_path, "rb") as f:
                shap_values = pickle.load(f)
            self.logger.info(f"SHAP values loaded successfully from {load_path}.")
            return shap_values
        except Exception as e:
            self.logger.error(f"Failed to load SHAP values: {e}")
            raise

if __name__ == "__main__":
    print("ShapDataHandler class for saving and loading SHAP values using pickle.")
