#!/usr/bin/env python3
"""
05_validate_on_actual_complications.py - Check if patients with target=1 get high risk
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
# ============================================================================
# FIXED: 05_validate_on_actual_complications.py
# ============================================================================

print("\n\n")
print("="*80)
print("05_VALIDATE_ON_ACTUAL_COMPLICATIONS - Fixed Version")
print("="*80)

# Load model
model_data = joblib.load('final_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
threshold = model_data['threshold']
feature_names = model_data['feature_names']

# Load best seed config
with open('best_seed_results.json', 'r') as f:
    best_config = json.load(f)

# Load data
df_raw = pd.read_csv("main_tavi_training.csv")
X_clean = pd.read_csv("cleaned_features.csv")
y = df_raw['target'].values

# Prepare features
X_subset = X_clean[feature_names].copy()

# ============================================================================
# FIX: Split data the SAME way as during training, use TEST set only
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y,
    test_size=0.3,
    stratify=y,
    random_state=best_config['seed']  # Same seed!
)

# Get complications in TEST set only
test_complications_mask = y_test == 1
n_test_complications = test_complications_mask.sum()

print(f"\n[INFO] Total complications in TEST set: {n_test_complications}")
print(f"[INFO] Total TEST patients: {len(y_test)}")
print(f"[INFO] Complication rate in test: {n_test_complications/len(y_test)*100:.1f}%")

# Get features for test complications
X_test_complications = X_test[test_complications_mask].copy()
y_test_complications = y_test[test_complications_mask]

print(f"\n[INFO] Testing predictions on {len(X_test_complications)} patients with actual complications (from TEST set)...")

# Impute using TRAINING statistics
X_test_complications_filled = X_test_complications.fillna(X_train.median())

# Scale
X_test_complications_sc = scaler.transform(X_test_complications_filled)

# Predict
probs = model.predict_proba(X_test_complications_sc)[:, 1]
preds = (probs > threshold).astype(int)
# ============================================================================

# Calculate metrics
high_risk_count = (probs > threshold).sum()
low_risk_count = (probs <= threshold).sum()

print(f"\n[INFO] Results on TEST SET complications:")
print(f"  Patients flagged as HIGH RISK: {high_risk_count}/{len(X_test_complications)} ({high_risk_count/len(X_test_complications)*100:.1f}%)")
print(f"  Patients flagged as LOW RISK: {low_risk_count}/{len(X_test_complications)} ({low_risk_count/len(X_test_complications)*100:.1f}%)")

print(f"\n[INFO] Risk Distribution:")
print(f"  Mean risk: {probs.mean()*100:.2f}%")
print(f"  Median risk: {np.median(probs)*100:.2f}%")
print(f"  Min risk: {probs.min()*100:.2f}%")
print(f"  Max risk: {probs.max()*100:.2f}%")
print(f"  Threshold: {threshold*100:.2f}%")

# Show some examples
print(f"\n[INFO] Examples of TEST patients with complications:")
print(f"{'Index':<8} {'FCC':<10} {'Masa VS/SC':<15} {'Risk %':<10} {'Flagged':<10}")
print("-" * 60)

# Get original indices in full dataset
test_indices = np.where(np.isin(np.arange(len(y)), X_test.index))[0]
complication_test_indices = test_indices[test_complications_mask]

for i in range(min(10, len(X_test_complications))):
    orig_idx = X_test_complications.index[i]
    fcc = X_test_complications.iloc[i][feature_names[0]]
    masa = X_test_complications.iloc[i][feature_names[1]]
    risk_pct = probs[i] * 100
    flagged = "HIGH RISK" if probs[i] > threshold else "LOW RISK"
    print(f"{orig_idx:<8} {fcc:<10.1f} {masa:<15.1f} {risk_pct:<10.2f} {flagged:<10}")

# Summary
if high_risk_count == len(X_test_complications):
    print(f"\n[OK] ALL test complications correctly flagged as HIGH RISK!")
elif high_risk_count > len(X_test_complications) * 0.8:
    print(f"\n[OK] Most test complications ({high_risk_count/len(X_test_complications)*100:.1f}%) flagged as HIGH RISK")
else:
    print(f"\n[WARN] Only {high_risk_count/len(X_test_complications)*100:.1f}% of test complications flagged as HIGH RISK")
    print(f"  Model may miss some complications in practice")

# Show missed cases
if low_risk_count > 0:
    missed_mask = probs <= threshold
    print(f"\n[INFO] TEST complications flagged as LOW RISK (False Negatives):")
    print(f"{'Index':<8} {'FCC':<10} {'Masa VS/SC':<15} {'Risk %':<10}")
    print("-" * 50)
    
    missed_indices = X_test_complications.index[missed_mask]
    for idx in missed_indices[:10]:
        row_idx = np.where(X_test_complications.index == idx)[0][0]
        fcc = X_test_complications.iloc[row_idx][feature_names[0]]
        masa = X_test_complications.iloc[row_idx][feature_names[1]]
        risk_pct = probs[row_idx] * 100
        print(f"{idx:<8} {fcc:<10.1f} {masa:<15.1f} {risk_pct:<10.2f}")

print("\n" + "="*80)
print("[INFO] All metrics are on TEST SET complications only")
print("[INFO] This represents true performance on unseen patients")
print("="*80)
