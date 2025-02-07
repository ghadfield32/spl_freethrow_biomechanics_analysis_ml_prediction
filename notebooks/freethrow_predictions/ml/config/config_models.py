from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from omegaconf import OmegaConf
from pathlib import Path
class FeaturesConfig(BaseModel):
    ordinal_categoricals: List[str] = []
    nominal_categoricals: List[str] = []
    numericals: List[str]
    y_variable: List[str]

class PathsConfig(BaseModel):
    data_dir: str = "../../data/preprocessor"  # Unified base for preprocessor outputs
    raw_data: str = "../processed/final_ml_dataset.csv"
    processed_data_dir: str = "processed"
    features_metadata_file: str = "features_info/final_ml_df_selected_features_columns.pkl"
    predictions_output_dir: str = "../../data/predictions"
    config_file: str = "../../data/model/preprocessor_config/preprocessor_config.yaml"
    log_dir: str = "../../data/preprocessor/logs"
    model_save_base_dir: str = "../../data/model"
    transformers_save_base_dir: str = "../../data/preprocessor/transformers"
    plots_output_dir: str = "../../data/preprocessor/plots"
    training_output_dir: str = "../../data/preprocessor/training_output"
    log_file: Optional[str] = "../../data/preprocessor/prediction.log"


class ModelsConfig(BaseModel):
    selected_models: List[str] = Field(default_factory=lambda: ["XGBoost", "Random Forest", "Decision Tree, CatBoost"])
    selection_metric: str = "Log Loss"
    Tree_Based_Classifier: Optional[Dict] = {}

class LoggingConfig(BaseModel):
    level: str = "INFO"
    debug: bool = False

class AppConfig(BaseModel):
    # Make sure this is now a plain list
    model_types: List[str]
    model_sub_types: Dict[str, List[str]]
    features: FeaturesConfig
    paths: PathsConfig
    models: ModelsConfig
    logging: LoggingConfig = LoggingConfig()   # default provided if missing
    execution: Optional[Dict] = {}


def load_config(config_path: Path) -> AppConfig:
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print("DEBUG: Loaded configuration dictionary:", cfg_dict)
    app_config = AppConfig(**cfg_dict)
    print(f"[Config Loader] ✅ Successfully loaded configuration from {config_path}")
    return app_config


if __name__ == "__main__":
    import json
    config = load_config(Path('../../data/model/preprocessor_config/preprocessor_config.yaml'))
    print(json.dumps(config.model_dump(), indent=2))
