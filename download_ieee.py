#!/usr/bin/env python
"""Download IEEE-CIS Fraud Detection dataset from Kaggle"""

import kaggle
import os
import sys

os.chdir(r'c:\Users\User\Documents\gmdh-predictive-engine')

try:
    print("Downloading IEEE-CIS Fraud Detection dataset...")
    kaggle.api.competition_download_files('ieee-fraud-detection', path='.')
    print("✓ Dataset downloaded successfully")
    
    # Verify key files
    if os.path.exists('train_transaction.csv'):
        print("✓ train_transaction.csv found")
    if os.path.exists('train_identity.csv'):
        print("✓ train_identity.csv found")
        
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
