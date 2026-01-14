#!/usr/bin/env python3
"""
08_find_optimal_thresholds.py - Find optimal thresholds for different objectives
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

print("="*80)
print("08_FIND_OPTIMAL_THRESHOLDS")
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

# Test many thresholds
thresholds = np.arange(0.30, 0.70, 0.01)

results = []

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    
    recall = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    
    # Combined score (recall + accuracy)
    combined = recall + accuracy
    
    results.append({
        'threshold': thresh,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'combined': combined
    })

df = pd.DataFrame(results)

# Find optimal thresholds
# 1. Maximize recall (find threshold that gives recall >= 0.99, then pick highest accuracy among those)
max_recall_thresh = df.loc[df['recall'].idxmax()]
# Find threshold with recall >= 0.99 and highest accuracy
recall_99 = df[df['recall'] >= 0.99]
if len(recall_99) > 0:
    max_recall_opt = recall_99.loc[recall_99['accuracy'].idxmax()]
    threshold_max_recall = max_recall_opt['threshold']
else:
    threshold_max_recall = max_recall_thresh['threshold']

# 2. Maximize accuracy (but require recall > 0)
# Find threshold with highest accuracy where recall > 0
valid_acc = df[df['recall'] > 0]
if len(valid_acc) > 0:
    max_acc_thresh = valid_acc.loc[valid_acc['accuracy'].idxmax()]
    threshold_max_accuracy = max_acc_thresh['threshold']
else:
    # Fallback: use threshold with best accuracy among those with recall > 0.1
    valid_acc = df[df['recall'] > 0.1]
    if len(valid_acc) > 0:
        max_acc_thresh = valid_acc.loc[valid_acc['accuracy'].idxmax()]
        threshold_max_accuracy = max_acc_thresh['threshold']
    else:
        max_acc_thresh = df.loc[df['accuracy'].idxmax()]
        threshold_max_accuracy = max_acc_thresh['threshold']

# 3. Maximize combined (recall + accuracy)
max_combined_thresh = df.loc[df['combined'].idxmax()]
threshold_max_combined = max_combined_thresh['threshold']

# 4. Maximize F1 (balance)
max_f1_thresh = df.loc[df['f1'].idxmax()]
threshold_max_f1 = max_f1_thresh['threshold']

# 5. Youden's J (sensitivity + specificity - 1, equivalent to recall + specificity - 1)
df['youden'] = df['recall'] + (1 - df['precision'] * df['recall'] / (df['precision'] + 1e-10)) - 1
# Actually, let's calculate specificity properly
from sklearn.metrics import confusion_matrix
youden_scores = []
for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    youden_scores.append(youden)
df['youden'] = youden_scores
max_youden_thresh = df.loc[df['youden'].idxmax()]
threshold_youden = max_youden_thresh['threshold']

print("\n[INFO] Optimal Thresholds Found:")
print(f"\n1. MAXIMIZE RECALL (Safety First):")
print(f"   Threshold: {threshold_max_recall*100:.2f}%")
recall_preds = (probs > threshold_max_recall).astype(int)
recall_recall = recall_score(y, recall_preds, zero_division=0)
recall_acc = accuracy_score(y, recall_preds)
recall_prec = precision_score(y, recall_preds, zero_division=0)
print(f"   Recall: {recall_recall:.2%}")
print(f"   Accuracy: {recall_acc:.2%}")
print(f"   Precision: {recall_prec:.2%}")

print(f"\n2. MAXIMIZE ACCURACY:")
print(f"   Threshold: {threshold_max_accuracy*100:.2f}%")
acc_preds = (probs > threshold_max_accuracy).astype(int)
acc_recall = recall_score(y, acc_preds, zero_division=0)
acc_acc = accuracy_score(y, acc_preds)
acc_prec = precision_score(y, acc_preds, zero_division=0)
print(f"   Recall: {acc_recall:.2%}")
print(f"   Accuracy: {acc_acc:.2%}")
print(f"   Precision: {acc_prec:.2%}")

print(f"\n3. MAXIMIZE COMBINED (Recall + Accuracy):")
print(f"   Threshold: {threshold_max_combined*100:.2f}%")
comb_preds = (probs > threshold_max_combined).astype(int)
comb_recall = recall_score(y, comb_preds, zero_division=0)
comb_acc = accuracy_score(y, comb_preds)
comb_prec = precision_score(y, comb_preds, zero_division=0)
print(f"   Recall: {comb_recall:.2%}")
print(f"   Accuracy: {comb_acc:.2%}")
print(f"   Precision: {comb_prec:.2%}")

print(f"\n4. MAXIMIZE F1 (Balance):")
print(f"   Threshold: {threshold_max_f1*100:.2f}%")
f1_preds = (probs > threshold_max_f1).astype(int)
f1_recall = recall_score(y, f1_preds, zero_division=0)
f1_acc = accuracy_score(y, f1_preds)
f1_prec = precision_score(y, f1_preds, zero_division=0)
f1_f1 = f1_score(y, f1_preds, zero_division=0)
print(f"   Recall: {f1_recall:.2%}")
print(f"   Accuracy: {f1_acc:.2%}")
print(f"   Precision: {f1_prec:.2%}")
print(f"   F1 Score: {f1_f1:.2%}")

print(f"\n5. YOUDEN'S J (Optimal ROC Point):")
print(f"   Threshold: {threshold_youden*100:.2f}%")
youden_preds = (probs > threshold_youden).astype(int)
youden_recall = recall_score(y, youden_preds, zero_division=0)
youden_acc = accuracy_score(y, youden_preds)
youden_prec = precision_score(y, youden_preds, zero_division=0)
print(f"   Recall: {youden_recall:.2%}")
print(f"   Accuracy: {youden_acc:.2%}")
print(f"   Precision: {youden_prec:.2%}")

# Save to config
config = {
    "presets": {
        "maximize_recall": {
            "name": "Maximize Recall (Safety First)",
            "description": "Catches the most complications. Best for patient safety.",
            "threshold": float(threshold_max_recall),
            "threshold_percentage": float(threshold_max_recall * 100),
            "recall": float(recall_recall),
            "accuracy": float(recall_acc),
            "precision": float(recall_prec)
        },
        "maximize_accuracy": {
            "name": "Maximize Accuracy",
            "description": "Highest overall correctness. Fewer false alarms.",
            "threshold": float(threshold_max_accuracy),
            "threshold_percentage": float(threshold_max_accuracy * 100),
            "recall": float(acc_recall),
            "accuracy": float(acc_acc),
            "precision": float(acc_prec)
        },
        "maximize_combined": {
            "name": "Maximize Combined (Recall + Accuracy)",
            "description": "Balances catching complications and overall correctness.",
            "threshold": float(threshold_max_combined),
            "threshold_percentage": float(threshold_max_combined * 100),
            "recall": float(comb_recall),
            "accuracy": float(comb_acc),
            "precision": float(comb_prec)
        },
        "maximize_f1": {
            "name": "Maximize F1 (Balance)",
            "description": "Balances recall and precision using F1 score.",
            "threshold": float(threshold_max_f1),
            "threshold_percentage": float(threshold_max_f1 * 100),
            "recall": float(f1_recall),
            "accuracy": float(f1_acc),
            "precision": float(f1_prec),
            "f1": float(f1_f1)
        },
        "youden": {
            "name": "Youden's J (Optimal ROC)",
            "description": "Optimal point on ROC curve (sensitivity + specificity).",
            "threshold": float(threshold_youden),
            "threshold_percentage": float(threshold_youden * 100),
            "recall": float(youden_recall),
            "accuracy": float(youden_acc),
            "precision": float(youden_prec)
        }
    },
    "current": "maximize_combined",
    "custom": {
        "threshold": 0.45,
        "threshold_percentage": 45.0,
        "description": "Custom threshold set by user"
    },
    "last_updated": "2026-01-13"
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n[OK] Optimal thresholds saved to config.json")
print(f"[OK] Current preset: {config['current']}")

print("\n" + "="*80)
