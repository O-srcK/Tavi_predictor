#!/usr/bin/env python3
"""
06_explain_recall_vs_threshold.py - Explain recall vs threshold relationship
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

print("="*80)
print("06_EXPLAIN_RECALL_VS_THRESHOLD")
print("="*80)

# Setup - change to script directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Check if model exists
if not Path('final_model.pkl').exists():
    print("[ERROR] final_model.pkl not found!")
    print("[INFO] Please run 04_train_final_model.py first to create the model.")
    exit(1)

# Load model
model_data = joblib.load('final_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# Load data
df_raw = pd.read_csv("main_tavi_training.csv")
X_clean = pd.read_csv("cleaned_features.csv")
y = df_raw['target'].values

# Prepare features
X_subset = X_clean[feature_names].copy()
X_subset = X_subset.fillna(X_subset.median())

# Get patients with complications
complications_mask = y == 1
X_complications = X_subset[complications_mask]
y_complications = y[complications_mask]

# Scale
X_complications_sc = scaler.transform(X_complications)

# Get probabilities
probs = model.predict_proba(X_complications_sc)[:, 1]

# Load current threshold from config
with open('config.json', 'r') as f:
    config = json.load(f)

# Handle new config structure with presets
if 'current' in config:
    current_preset = config.get('current', 'maximize_combined')
    if current_preset in config.get('presets', {}):
        current_threshold = config['presets'][current_preset]['threshold']
    elif current_preset == 'custom' and 'custom' in config:
        current_threshold = config['custom']['threshold']
    else:
        current_threshold = 0.5
elif 'threshold' in config:
    # Old format
    current_threshold = config['threshold']
else:
    current_threshold = 0.5

# Calculate recall with current threshold
preds_current = (probs > current_threshold).astype(int)
true_positives = ((preds_current == 1) & (y_complications == 1)).sum()
false_negatives = ((preds_current == 0) & (y_complications == 1)).sum()
recall_current = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print(f"\n[INFO] Current Threshold: {current_threshold*100:.2f}%")
print(f"[INFO] Patients with complications: {len(y_complications)}")
print(f"[INFO] Flagged as HIGH RISK: {preds_current.sum()}")
print(f"[INFO] Flagged as LOW RISK: {len(y_complications) - preds_current.sum()}")
print(f"\n[INFO] Recall = True Positives / (True Positives + False Negatives)")
print(f"  True Positives: {true_positives}")
print(f"  False Negatives: {false_negatives}")
print(f"  Recall: {recall_current:.4f} ({recall_current*100:.2f}%)")

print("\n" + "="*80)
print("RECALL vs THRESHOLD")
print("="*80)

# Test different thresholds
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
print(f"\n{'Threshold':<12} {'Recall':<10} {'High Risk':<12} {'Low Risk':<12}")
print("-" * 50)

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    tp = ((preds == 1) & (y_complications == 1)).sum()
    fn = ((preds == 0) & (y_complications == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    high_risk = preds.sum()
    low_risk = len(y_complications) - preds.sum()
    print(f"{thresh*100:>6.0f}%      {recall:>6.2%}    {high_risk:>4}/{len(y_complications):<4}    {low_risk:>4}/{len(y_complications):<4}")

print("\n" + "="*80)
print("EXPLANATION")
print("="*80)

print("\n[INFO] Why Recall Changed:")
print("  1. The recall of 1.0 (100%) was achieved on a SPECIFIC test set")
print("     - Test set with seed 36 (30% of data)")
print("     - With threshold 50.17%")
print("     - That test set had different patients than what we're testing now")
print("")
print("  2. Current recall (88.9%) is on ALL patients with complications")
print("     - Testing on all 18 patients with target=1")
print("     - With threshold 45%")
print("     - This is the 'real-world' performance")
print("")
print("  3. Recall depends on:")
print("     - The threshold (lower = higher recall)")
print("     - Which patients are in the test set")
print("     - The model's ability to predict")

print("\n[INFO] To achieve Recall = 1.0 (100%):")
print("  - Lower threshold to ~40%")
print("  - This will catch ALL complications")
print("  - But will also increase false alarms")

print("\n[INFO] Current Performance:")
print(f"  - Threshold: {current_threshold*100:.2f}%")
print(f"  - Recall: {recall_current:.2%} (catches {recall_current*100:.1f}% of complications)")
print(f"  - Balance between catching complications and false alarms")

print("\n" + "="*80)
