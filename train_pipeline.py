# End-to-end training pipeline
# Runs data generation, validation, feature engineering, model training, and evaluation

from pathlib import Path
import logging
import yaml

from data_ingestion import generate_and_save_data
from data_validation import validate_data_file
from feature_engineering import engineer_features
from baseline_model import train_baseline_model
from advanced_model import train_advanced_model
from model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml"):
    logger.info("Starting fraud detection pipeline...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate data
    raw_data_path = config.get('data', {}).get('raw_data_path', 'data/raw/transactions.csv')
    
    if Path(raw_data_path).exists():
        try:
            Path(raw_data_path).unlink()
        except Exception as e:
            logger.warning(f"Could not delete existing file: {e}")
    
    logger.info("Generating transaction data...")
    generate_and_save_data(
        output_path=raw_data_path,
        n_transactions=100000,
        config_path=config_path
    )
    
    # Validate data
    logger.info("Validating data...")
    is_valid, validation_results = validate_data_file(raw_data_path)
    
    if not is_valid:
        logger.error("Data validation failed")
        return
    
    # Feature engineering
    processed_data_path = config.get('data', {}).get('processed_data_path', 'data/processed/features.csv')
    
    if not Path(processed_data_path).exists():
        logger.info("Creating features...")
        engineer_features(
            input_path=raw_data_path,
            output_path=processed_data_path,
            config_path=config_path,
            is_training=True
        )
    
    # Train baseline
    logger.info("Training baseline model...")
    baseline_model, baseline_scaler, baseline_metrics = train_baseline_model(
        data_path=processed_data_path,
        config_path=config_path,
        model_save_path="models/baseline_model.pkl"
    )
    
    # Train advanced
    logger.info("Training advanced model...")
    advanced_model, advanced_metrics = train_advanced_model(
        data_path=processed_data_path,
        config_path=config_path,
        model_save_path="models/advanced_model.pkl"
    )
    
    # Register models
    registry = ModelRegistry(registry_path="models/")
    
    registry.register_model(
        model_name="baseline",
        model_path="models/baseline_model.pkl",
        version="v1.0",
        metrics=baseline_metrics,
        metadata={"model_type": "logistic_regression"}
    )
    
    registry.register_model(
        model_name="advanced",
        model_path="models/advanced_model.pkl",
        version="v1.0",
        metrics=advanced_metrics,
        metadata={"model_type": "xgboost"}
    )
    
    best_model = registry.get_best_model(metric="pr_auc")
    if best_model:
        logger.info(f"Best model: {best_model['model_name']} (PR-AUC: {best_model['metrics']['pr_auc']:.4f})")
    
    logger.info("Pipeline complete")


if __name__ == "__main__":
    run_pipeline()
