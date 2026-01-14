#!/usr/bin/env python3
"""
04_train_final_model.py - Train and save final model with best configuration
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime

print("="*80)
print("04_TRAIN_FINAL_MODEL - Train Best Configuration")
print("="*80)

# Setup
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load best configuration
with open('best_seed_results.json', 'r') as f:
    best_config = json.load(f)

# Load threshold from config (or use default from best_config)
config_path = Path('config.json')
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Handle new config structure with presets
    if 'current' in config:
        current_preset = config.get('current', 'maximize_combined')
        if current_preset in config.get('presets', {}):
            threshold = config['presets'][current_preset]['threshold']
            print(f"[INFO] Using threshold from preset '{current_preset}': {threshold*100:.2f}%")
        elif current_preset == 'custom' and 'custom' in config:
            threshold = config['custom']['threshold']
            print(f"[INFO] Using custom threshold from config.json: {threshold*100:.2f}%")
        else:
            threshold = best_config['best_threshold']
            print(f"[INFO] Using threshold from best seed: {threshold*100:.2f}%")
    elif 'threshold' in config:
        # Old format
        threshold = config['threshold']
        print(f"[INFO] Using threshold from config.json: {threshold*100:.2f}%")
    else:
        threshold = best_config['best_threshold']
        print(f"[INFO] Using threshold from best seed: {threshold*100:.2f}%")
else:
    threshold = best_config['best_threshold']
    print(f"[INFO] Using threshold from best seed: {threshold*100:.2f}%")
    print(f"[INFO] To customize threshold, edit config.json")

# Load data
df_raw = pd.read_csv("main_tavi_training.csv")
X_clean = pd.read_csv("cleaned_features.csv")
y = df_raw['target'].values

# Prepare features
feat_names = best_config['feature_names']
X_subset = X_clean[feat_names].copy()
X_subset = X_subset.fillna(X_subset.median())

# Train on full dataset with best seed
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y, test_size=0.3, stratify=y, random_state=best_config['seed']
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    penalty='l2',
    class_weight='balanced',
    C=best_config['best_C'],
    random_state=42,
    max_iter=10000
)
model.fit(X_train_sc, y_train)

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': feat_names,
    'threshold': threshold,  # Use threshold from config
    'config': best_config
}

joblib.dump(model_data, 'final_model.pkl')
print(f"\n[OK] Model saved: final_model.pkl")

# Save model info
model_info = {
    'features': best_config['features'],
    'feature_names': feat_names,
    'C': best_config['best_C'],
    'threshold': threshold,  # Use threshold from config
    'seed': best_config['seed'],
    'metrics': best_config['metrics'],
    'fairness_score': best_config['fairness_score'],
    'performance_score': best_config['performance_score'],
    'combined_score': best_config['combined_score'],
    'threshold_source': 'config.json' if config_path.exists() else 'best_seed'
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"[OK] Model info saved: model_info.json")

# Test prediction function
def predict_risk(fcc, masa_vs_sc):
    """Predict risk for a new patient"""
    # Load model
    model_data = joblib.load('final_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    # Load threshold from config if it exists, otherwise use model default
    if Path('config.json').exists():
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Handle new config structure with presets
        if 'current' in config:
            current_preset = config.get('current', 'maximize_combined')
            if current_preset in config.get('presets', {}):
                threshold = config['presets'][current_preset]['threshold']
            elif current_preset == 'custom' and 'custom' in config:
                threshold = config['custom']['threshold']
            else:
                threshold = model_data['threshold']
        elif 'threshold' in config:
            # Old format
            threshold = config['threshold']
        else:
            threshold = model_data['threshold']
    else:
        threshold = model_data['threshold']
    
    # Prepare input
    X_input = pd.DataFrame({
        feat_names[0]: [fcc],
        feat_names[1]: [masa_vs_sc]
    })
    
    # Scale
    X_input_sc = scaler.transform(X_input)
    
    # Predict
    prob = model.predict_proba(X_input_sc)[0, 1]
    risk_flag = prob > threshold
    
    return {
        'risk_probability': float(prob),
        'risk_percentage': float(prob * 100),
        'risk_flag': bool(risk_flag),
        'risk_level': 'HIGH RISK' if risk_flag else 'LOW RISK',
        'threshold_used': float(threshold)
    }

# Test with example
print(f"\n[INFO] Testing prediction function...")
test_result = predict_risk(fcc=65.0, masa_vs_sc=200.0)
print(f"  Example: FCC=65.0, Masa VS/SC=200.0")
print(f"  Risk: {test_result['risk_percentage']:.2f}% ({test_result['risk_level']})")

print(f"\n[OK] Final model ready for deployment!")
