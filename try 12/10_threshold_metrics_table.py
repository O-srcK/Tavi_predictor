#!/usr/bin/env python3
"""
10_threshold_metrics_table.py - Formatted Metrics Table
Format: AUC, Threshold, Recall, Accuracy (all as 0-100 with 2 decimals)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

print("="*80)
print("10_THRESHOLD_METRICS_TABLE - Test Set Metrics")
print("="*80)

# Check if model exists
if not Path('final_model.pkl').exists():
    print("[ERROR] final_model.pkl not found!")
    exit(1)

# Load model
model_data = joblib.load('final_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
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

# Split data the SAME way as during training
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y,
    test_size=0.3,
    stratify=y,
    random_state=best_config['seed']
)

print(f"\n[INFO] Using seed {best_config['seed']} (same as training)")
print(f"[INFO] Test set size: {len(X_test)} patients ({len(X_test)/len(y)*100:.1f}%)")
print(f"[INFO] Complications in test set: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

# Impute TEST data using TRAINING statistics
X_test_filled = X_test.fillna(X_train.median())

# Scale TEST data only
X_test_scaled = scaler.transform(X_test_filled)

# Get probabilities on TEST set only
probs = model.predict_proba(X_test_scaled)[:, 1]

# AUC is constant (doesn't depend on threshold)
auc = roc_auc_score(y_test, probs) * 100  # Convert to 0-100

# Test different thresholds
thresholds = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65]

results = []

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    
    acc = accuracy_score(y_test, preds) * 100  # Convert to 0-100
    rec = recall_score(y_test, preds, zero_division=0) * 100  # Convert to 0-100
    
    results.append({
        'AUC': auc,
        'Threshold': thresh * 100,  # Convert to 0-100
        'Recall': rec,
        'Accuracy': acc
    })

# Create DataFrame
df = pd.DataFrame(results)

# Display table
print("\n" + "="*80)
print("METRICS ON TEST SET ONLY")
print("="*80)
print(f"\n{'AUC':<10} {'Threshold':<12} {'Recall':<12} {'Accuracy':<12}")
print("-" * 50)

for _, row in df.iterrows():
    # Highlight certain thresholds if needed
    marker = ""
    if abs(row['Threshold'] - 42.0) < 0.1:
        marker = " *"
    elif abs(row['Threshold'] - 50.0) < 0.1:
        marker = " **"
    
    print(f"{row['AUC']:<10.2f} {row['Threshold']:<12.2f} {row['Recall']:<12.2f} {row['Accuracy']:<12.2f}{marker}")

print("\n* = 42% threshold (original safe mode)")
print("** = 50% threshold (recommended)")
print("="*80)

# Save to CSV with proper formatting
df_output = df.copy()
df_output = df_output.round(2)  # Round to 2 decimal places
df_output.to_csv('threshold_metrics_table.csv', index=False, float_format='%.2f')

print(f"\n[OK] Table saved to: threshold_metrics_table.csv")
print(f"[INFO] Format: AUC, Threshold, Recall, Accuracy (all 0-100 scale, 2 decimals)")
print(f"[INFO] All metrics calculated on TEST SET only")

# Also print a summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Model AUC: {auc:.2f}")
print(f"Test set size: {len(y_test)} patients")
print(f"Complications: {y_test.sum()} patients ({y_test.mean()*100:.2f}%)")
print("="*80)
