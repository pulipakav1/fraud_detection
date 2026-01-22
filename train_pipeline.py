"""
End-to-End Training Pipeline

Runs the complete ML pipeline:
1. Data generation
2. Data validation
3. Feature engineering
4. Model training (baseline + advanced)
5. Model evaluation
6. Model registration
"""

import sys
from pathlib import Path
import logging
import yaml

from src.data.ingestion import generate_and_save_data
from src.data.validation import validate_data_file
from src.features.engineering import engineer_features
from src.models.baseline import train_baseline_model
from src.models.advanced import train_advanced_model
from src.models.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml"):
    """
    Run the complete ML pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION ML PIPELINE")
    logger.info("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Generate data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: DATA GENERATION")
    logger.info("=" * 60)
    raw_data_path = config.get('data', {}).get('raw_data_path', 'data/raw/transactions.csv')
    
    if not Path(raw_data_path).exists():
        logger.info("Generating synthetic transaction data...")
        generate_and_save_data(
            output_path=raw_data_path,
            n_transactions=100000,
            config_path=config_path
        )
    else:
        logger.info(f"Raw data already exists at {raw_data_path}, skipping generation")
    
    # Step 2: Validate data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: DATA VALIDATION")
    logger.info("=" * 60)
    is_valid, validation_results = validate_data_file(raw_data_path)
    
    if not is_valid:
        logger.error("Data validation failed. Please check the data.")
        return
    
    # Step 3: Feature engineering
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 60)
    processed_data_path = config.get('data', {}).get('processed_data_path', 'data/processed/features.csv')
    
    if not Path(processed_data_path).exists():
        logger.info("Engineering features...")
        engineer_features(
            input_path=raw_data_path,
            output_path=processed_data_path,
            config_path=config_path,
            is_training=True
        )
    else:
        logger.info(f"Processed data already exists at {processed_data_path}, skipping feature engineering")
    
    # Step 4: Train baseline model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: TRAIN BASELINE MODEL")
    logger.info("=" * 60)
    baseline_model, baseline_scaler, baseline_metrics = train_baseline_model(
        data_path=processed_data_path,
        config_path=config_path,
        model_save_path="models/baseline_model.pkl"
    )
    
    # Step 5: Train advanced model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: TRAIN ADVANCED MODEL")
    logger.info("=" * 60)
    advanced_model, advanced_metrics = train_advanced_model(
        data_path=processed_data_path,
        config_path=config_path,
        model_save_path="models/advanced_model.pkl"
    )
    
    # Step 6: Register models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: MODEL REGISTRY")
    logger.info("=" * 60)
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
    
    # Get best model
    best_model = registry.get_best_model(metric="pr_auc")
    if best_model:
        logger.info(f"\n🏆 Best Model: {best_model['model_name']} v{best_model['version']}")
        logger.info(f"   PR-AUC: {best_model['metrics']['pr_auc']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Start inference API: python -m src.inference.api")
    logger.info("2. Test API: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @examples/transaction_example.json")


if __name__ == "__main__":
    run_pipeline()
