# config/banks_config.py
"""
Configuration file for Bank Reviews Analysis Project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Play Store App IDs
APP_IDS = {
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'Awash': os.getenv('AWASHPAY_APP_ID', 'com.sc.awashpay'),
    'Amharabank': os.getenv('AMHARABANK_APP_ID', 'com.amharabank.Aba_mobile_banking')
}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'Awash': 'Awash Bank',
    'Amharabank': 'Amhara Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 400)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# File Paths
DATA_PATHS = {
    'raw': 'data/raw data',
    'processed': 'data/processed data',
    'raw_reviews': 'data/raw data/reviews_raw.csv',
    'processed_reviews': 'data/processed data/reviews_processed.csv',
    'sentiment_results': 'data/processed data/reviews_with_sentiment.csv',
    'final_results': 'data/processed data/reviews_final.csv',
    'themes_results': 'data/processed data/thematic_analysis.csv'
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'min_review_length': 5,
    'theme_keywords': {
        'login_issues': ['login', 'sign in', 'password', 'account access', 'authentication'],
        'transaction_issues': ['transfer', 'transaction', 'payment', 'send money', 'slow', 'failed'],
        'app_performance': ['crash', 'slow', 'freeze', 'bug', 'error', 'not working'],
        'ui_ux': ['interface', 'design', 'layout', 'user friendly', 'easy to use', 'complicated'],
        'customer_service': ['support', 'help', 'service', 'response', 'contact'],
        'features': ['feature', 'function', 'option', 'missing', 'should have']
    }
}