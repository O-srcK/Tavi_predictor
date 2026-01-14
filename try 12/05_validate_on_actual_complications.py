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
from sklearn.preprocessing import StandardScaler

print("="*80)
print("05_VALIDATE_ON_ACTUAL_COMPLICATIONS")
print("="*80)

# Setup
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load model
model_data = joblib.load('final_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
threshold = model_data['threshold']
feature_names = model_data['feature_names']

# Load data
df_raw = pd.read_csv("main_tavi_training.csv")
X_clean = pd.read_csv("cleaned_features.csv")
y = df_raw['target'].values

# Get patients with complications (target=1)
complications_mask = y == 1
n_complications = complications_mask.sum()

print(f"\n[INFO] Total patients with complications (target=1): {n_complications}")
print(f"[INFO] Total patients: {len(y)}")
print(f"[INFO] Complication rate: {n_complications/len(y)*100:.1f}%")

# Prepare features
X_subset = X_clean[feature_names].copy()
X_subset = X_subset.fillna(X_subset.median())

# Get data for patients with complications
X_complications = X_subset[complications_mask]
y_complications = y[complications_mask]

print(f"\n[INFO] Testing predictions on {len(X_complications)} patients with actual complications...")

# Scale (using the scaler from training)
X_complications_sc = scaler.transform(X_complications)

# Predict
probs = model.predict_proba(X_complications_sc)[:, 1]
preds = (probs > threshold).astype(int)

# Calculate metrics
high_risk_count = (probs > threshold).sum()
low_risk_count = (probs <= threshold).sum()

print(f"\n[INFO] Results:")
print(f"  Patients flagged as HIGH RISK: {high_risk_count}/{len(X_complications)} ({high_risk_count/len(X_complications)*100:.1f}%)")
print(f"  Patients flagged as LOW RISK: {low_risk_count}/{len(X_complications)} ({low_risk_count/len(X_complications)*100:.1f}%)")

print(f"\n[INFO] Risk Distribution:")
print(f"  Mean risk: {probs.mean()*100:.2f}%")
print(f"  Median risk: {np.median(probs)*100:.2f}%")
print(f"  Min risk: {probs.min()*100:.2f}%")
print(f"  Max risk: {probs.max()*100:.2f}%")
print(f"  Threshold: {threshold*100:.2f}%")

# Show some examples
print(f"\n[INFO] Examples of patients with complications:")
print(f"{'Index':<8} {'FCC':<10} {'Masa VS/SC':<15} {'Risk %':<10} {'Flagged':<10}")
print("-" * 60)
for i in range(min(10, len(X_complications))):
    idx = np.where(complications_mask)[0][i]
    fcc = X_complications.iloc[i][feature_names[0]]
    masa = X_complications.iloc[i][feature_names[1]]
    risk_pct = probs[i] * 100
    flagged = "HIGH RISK" if probs[i] > threshold else "LOW RISK"
    print(f"{idx:<8} {fcc:<10.1f} {masa:<15.1f} {risk_pct:<10.2f} {flagged:<10}")

# Check if all are high risk
if high_risk_count == len(X_complications):
    print(f"\n[OK] ALL patients with complications are correctly flagged as HIGH RISK!")
elif high_risk_count > len(X_complications) * 0.8:
    print(f"\n[OK] Most patients with complications ({high_risk_count/len(X_complications)*100:.1f}%) are flagged as HIGH RISK")
else:
    print(f"\n[WARN] Only {high_risk_count/len(X_complications)*100:.1f}% of patients with complications are flagged as HIGH RISK")
    print(f"  This means the model may miss some complications")

# Show patients that were missed
if low_risk_count > 0:
    missed_mask = probs <= threshold
    missed_indices = np.where(complications_mask)[0][missed_mask]
    print(f"\n[INFO] Patients with complications but LOW RISK prediction:")
    print(f"{'Index':<8} {'FCC':<10} {'Masa VS/SC':<15} {'Risk %':<10}")
    print("-" * 50)
    for idx in missed_indices[:10]:  # Show first 10
        row_idx = np.where(np.where(complications_mask)[0] == idx)[0][0]
        fcc = X_complications.iloc[row_idx][feature_names[0]]
        masa = X_complications.iloc[row_idx][feature_names[1]]
        risk_pct = probs[row_idx] * 100
        print(f"{idx:<8} {fcc:<10.1f} {masa:<15.1f} {risk_pct:<10.2f}")

print("\n" + "="*80)
