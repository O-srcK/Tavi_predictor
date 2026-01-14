#!/usr/bin/env python3
"""
10_threshold_metrics_table.py - Show final 3 stats (AUC, Accuracy, Recall) by threshold
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

print("="*80)
print("10_THRESHOLD_METRICS_TABLE - Final 3 Stats by Threshold")
print("="*80)

# Setup
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

# Scale
X_scaled = scaler.transform(X_subset)

# Get probabilities
probs = model.predict_proba(X_scaled)[:, 1]

# AUC is constant (doesn't depend on threshold)
auc = roc_auc_score(y, probs)

# Test different thresholds
thresholds = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65]

results = []

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds, zero_division=0)
    
    results.append({
        'threshold': thresh,
        'threshold_pct': f"{thresh*100:.0f}%",
        'auc': auc,  # Constant
        'accuracy': acc,
        'recall': rec
    })

df = pd.DataFrame(results)

print("\n" + "="*80)
print("FINAL 3 STATS BY THRESHOLD")
print("="*80)
print(f"\nModel AUC (constant): {auc:.4f} ({auc*100:.2f}%)")
print("\n" + "="*80)
print(f"{'Threshold':<12} {'AUC':<12} {'Accuracy':<12} {'Recall':<12}")
print("-" * 50)

for _, row in df.iterrows():
    # Highlight 42% threshold
    if abs(row['threshold'] - 0.42) < 0.01:
        marker = " [SAFE THRESHOLD]"
    else:
        marker = ""
    
    print(f"{row['threshold_pct']:<12} {row['auc']*100:>10.2f}%  {row['accuracy']*100:>10.2f}%  {row['recall']*100:>10.2f}%{marker}")

print("="*80)

# Highlight the 42% threshold specifically
safe_row = df[df['threshold'] == 0.42].iloc[0] if len(df[df['threshold'] == 0.42]) > 0 else None

if safe_row is not None:
    print("\n" + "="*80)
    print("FINAL 3 STATS AT 42% THRESHOLD (SAFE MODE)")
    print("="*80)
    print(f"\n  AUC:      {safe_row['auc']*100:.2f}% (constant - model property)")
    print(f"  Accuracy: {safe_row['accuracy']*100:.2f}%")
    print(f"  Recall:   {safe_row['recall']*100:.2f}%")
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print(f"  - AUC {safe_row['auc']*100:.2f}%: Model can distinguish between high/low risk patients")
    print(f"  - Accuracy {safe_row['accuracy']*100:.2f}%: Overall correctness of predictions")
    print(f"  - Recall {safe_row['recall']*100:.2f}%: Catches {safe_row['recall']*100:.2f}% of all complications")
    print("="*80)

# Save to CSV for easy reference
df_output = df.copy()
df_output['auc'] = df_output['auc'] * 100
df_output['accuracy'] = df_output['accuracy'] * 100
df_output['recall'] = df_output['recall'] * 100
df_output = df_output.rename(columns={
    'threshold_pct': 'Threshold',
    'auc': 'AUC (%)',
    'accuracy': 'Accuracy (%)',
    'recall': 'Recall (%)'
})
df_output = df_output[['Threshold', 'AUC (%)', 'Accuracy (%)', 'Recall (%)']]
df_output.to_csv('threshold_metrics_table.csv', index=False)
print(f"\n[OK] Table saved to: threshold_metrics_table.csv")

print("\n" + "="*80)
