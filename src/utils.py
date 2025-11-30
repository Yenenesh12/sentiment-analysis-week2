"""
Utility functions for the project
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, Any

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('project.log'),
            logging.StreamHandler()
        ]
    )

def save_metrics(metrics: Dict[str, Any], filename: str):
    """Save metrics to JSON file"""
    try:
        with open(f"data/results/{filename}", 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved to data/results/{filename}")
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")

def load_metrics(filename: str) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    try:
        with open(f"data/results/{filename}", 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading metrics: {str(e)}")
        return {}

def validate_data_quality(df: pd.DataFrame, min_reviews: int = 1200, max_missing_pct: float = 5.0) -> bool:
    """Validate data quality against project requirements"""
    total_reviews = len(df)
    missing_data_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    meets_requirements = (
        total_reviews >= min_reviews and
        missing_data_pct < max_missing_pct
    )
    
    logging.info(f"Data Quality Check: {total_reviews} reviews, {missing_data_pct:.1f}% missing data")
    logging.info(f"Meets requirements: {meets_requirements}")
    
    return meets_requirements