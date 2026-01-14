#!/usr/bin/env python3
"""
07_threshold_impact_analysis.py - Analyze how threshold affects all metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score, 
    f1_score, confusion_matrix
)

print("="*80)
print("07_THRESHOLD_IMPACT_ANALYSIS - How Threshold Affects All Metrics")
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

# Scale
X_scaled = scaler.transform(X_subset)

# Get probabilities
probs = model.predict_proba(X_scaled)[:, 1]

# Test different thresholds
thresholds = np.arange(0.30, 0.70, 0.05)

results = []

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    
    # Calculate metrics
    recall = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Counts
    total_high_risk = (preds == 1).sum()
    total_low_risk = (preds == 0).sum()
    
    results.append({
        'threshold': thresh,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_high_risk': total_high_risk,
        'total_low_risk': total_low_risk
    })

df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("THRESHOLD IMPACT ON ALL METRICS")
print("="*80)

print(f"\n{'Thresh':<8} {'Recall':<8} {'Precision':<10} {'Accuracy':<10} {'F1':<8} {'Spec':<8} {'High Risk':<10}")
print("-" * 80)
for _, row in df_results.iterrows():
    print(f"{row['threshold']*100:>5.0f}%   {row['recall']:>6.2%}   {row['precision']:>8.2%}   "
          f"{row['accuracy']:>8.2%}   {row['f1']:>6.2%}   {row['specificity']:>6.2%}   "
          f"{row['total_high_risk']:>4}/{len(y):<4}")

print("\n" + "="*80)
print("DETAILED BREAKDOWN")
print("="*80)

# Show current threshold (45%)
current_thresh = 0.45
# Find closest threshold
closest_idx = (df_results['threshold'] - current_thresh).abs().idxmin()
current_row = df_results.iloc[closest_idx]

print(f"\n[INFO] Current Threshold: {current_thresh*100:.0f}%")
print(f"  Recall: {current_row['recall']:.2%} (catches {current_row['recall']:.1%} of complications)")
print(f"  Precision: {current_row['precision']:.2%} (of those flagged, {current_row['precision']:.1%} actually have complications)")
print(f"  Accuracy: {current_row['accuracy']:.2%} (overall correctness)")
print(f"  F1 Score: {current_row['f1']:.2%} (balance of recall and precision)")
print(f"  Specificity: {current_row['specificity']:.2%} (correctly identifies low-risk patients)")
print(f"\n  Confusion Matrix:")
print(f"    True Positives: {current_row['true_positives']} (correctly flagged complications)")
print(f"    True Negatives: {current_row['true_negatives']} (correctly identified as low risk)")
print(f"    False Positives: {current_row['false_positives']} (flagged but no complication - false alarms)")
print(f"    False Negatives: {current_row['false_negatives']} (missed complications)")
print(f"\n  Total Cases for Doctors:")
print(f"    High Risk Cases: {current_row['total_high_risk']}/{len(y)} ({current_row['total_high_risk']/len(y)*100:.1f}%)")

print("\n" + "="*80)
print("HOW THRESHOLD AFFECTS EACH METRIC")
print("="*80)

print("\n[INFO] 1. RECALL (Sensitivity)")
print("  - Definition: Of all patients with complications, how many did we catch?")
print("  - Effect: Lower threshold -> Higher recall (catches more complications)")
print("  - Trade-off: More false alarms")

print("\n[INFO] 2. PRECISION")
print("  - Definition: Of all patients flagged as HIGH RISK, how many actually have complications?")
print("  - Effect: Lower threshold -> Lower precision (more false alarms)")
print("  - Trade-off: More patients to investigate, but fewer are real complications")

print("\n[INFO] 3. ACCURACY")
print("  - Definition: Overall correctness (correct predictions / total)")
print("  - Effect: Usually peaks at a specific threshold (not always at extremes)")
print("  - Trade-off: Balance between catching complications and avoiding false alarms")

print("\n[INFO] 4. SPECIFICITY")
print("  - Definition: Of all patients WITHOUT complications, how many did we correctly identify?")
print("  - Effect: Higher threshold -> Higher specificity (fewer false alarms)")
print("  - Trade-off: But may miss more complications")

print("\n[INFO] 5. F1 SCORE")
print("  - Definition: Harmonic mean of recall and precision")
print("  - Effect: Balances recall and precision")
print("  - Trade-off: Good overall metric for balanced performance")

print("\n[INFO] 6. FALSE POSITIVE RATE")
print("  - Definition: Rate of false alarms (healthy patients flagged as high risk)")
print("  - Effect: Lower threshold -> Higher false positive rate")
print("  - Impact: More work for doctors (more patients to investigate)")

print("\n[INFO] 7. FALSE NEGATIVE RATE")
print("  - Definition: Rate of missed complications")
print("  - Effect: Lower threshold -> Lower false negative rate")
print("  - Impact: Fewer missed complications (better for patient safety)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n[INFO] For Maximum Safety (Catch All Complications):")
best_recall = df_results.loc[df_results['recall'].idxmax()]
print(f"  Threshold: {best_recall['threshold']*100:.0f}%")
print(f"  Recall: {best_recall['recall']:.2%}")
print(f"  But: {best_recall['false_positives']} false alarms ({best_recall['false_positive_rate']:.2%} false positive rate)")

print("\n[INFO] For Balanced Performance:")
best_f1 = df_results.loc[df_results['f1'].idxmax()]
print(f"  Threshold: {best_f1['threshold']*100:.0f}%")
print(f"  F1: {best_f1['f1']:.2%}, Recall: {best_f1['recall']:.2%}, Precision: {best_f1['precision']:.2%}")

print("\n[INFO] For Fewer False Alarms:")
best_precision = df_results.loc[df_results['precision'].idxmax()]
print(f"  Threshold: {best_precision['threshold']*100:.0f}%")
print(f"  Precision: {best_precision['precision']:.2%}")
print(f"  But: Recall only {best_precision['recall']:.2%} (misses more complications)")

print("\n[INFO] Current Setting (45%):")
print(f"  Good balance: {current_row['recall']:.1%} recall, {current_row['precision']:.1%} precision")
print(f"  Doctors need to review: {current_row['total_high_risk']} patients ({current_row['total_high_risk']/len(y)*100:.1f}% of all patients)")

print("\n" + "="*80)
