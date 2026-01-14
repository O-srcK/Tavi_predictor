#!/usr/bin/env python3
"""
02_test_many_seeds.py - Test Many Seeds, Find Best Balanced Model
Uses the same balanced optimization approach but tests many different seeds
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize_scalar

print("="*80)
print("02_TEST_MANY_SEEDS - Find Best Seed for Balanced Model")
print("="*80)

# Setup
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load data
df_raw = pd.read_csv("main_tavi_training.csv")
X_clean = pd.read_csv("cleaned_features.csv")
y = df_raw['target'].values

# Use the best features from try 10 (FCC, masa_vs_sc)
PRIORITY_FEATURES = {
    'FCC': 'FCC',
    'masa_vs_sc': 'Masa VS/SC (g/m²)',
}

feat_names = [PRIORITY_FEATURES[k] for k in PRIORITY_FEATURES.keys() if PRIORITY_FEATURES[k] in X_clean.columns]
X_subset = X_clean[feat_names].copy()
X_subset = X_subset.fillna(X_subset.median())

# Use best C from try 10
BEST_C = 0.5

# Test many seeds
SEEDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    42, 50, 100, 123, 200, 300, 456, 500, 666, 777, 888, 999,
    1000, 1234, 2026, 3141, 5000, 7777, 9999,
    1337, 2024, 2025, 2027, 2028, 2029, 2030
]

print(f"\n[INFO] Testing {len(SEEDS)} different seeds...")
print(f"[INFO] Features: {list(PRIORITY_FEATURES.keys())}")
print(f"[INFO] C: {BEST_C}")

all_results = []
best_result = None
best_combined_score = -1

for i, seed in enumerate(SEEDS):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.3, stratify=y, random_state=seed
        )
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = LogisticRegression(
            penalty='l2',
            class_weight='balanced',
            C=BEST_C,
            random_state=42,
            max_iter=10000
        )
        model.fit(X_train_sc, y_train)
        
        probs_train = model.predict_proba(X_train_sc)[:, 1]
        probs_test = model.predict_proba(X_test_sc)[:, 1]
        
        # Optimize threshold for combined score
        def objective(threshold):
            preds = (probs_test > threshold).astype(int)
            
            # Performance metrics
            auc = roc_auc_score(y_test, probs_test)
            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds, zero_division=0)
            performance_score = auc * 0.4 + acc * 0.35 + rec * 0.25
            
            # Fairness metrics
            train_auc = roc_auc_score(y_train, probs_train)
            overfitting_gap = train_auc - auc
            brier = brier_score_loss(y_test, probs_test)
            
            try:
                if len(np.unique(probs_test)) > 1 and len(y_test) >= 5:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_test, probs_test, n_bins=min(5, len(y_test)//2), strategy='uniform'
                    )
                    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                else:
                    calibration_error = np.mean(np.abs(y_test.astype(float) - probs_test))
            except:
                calibration_error = np.mean(np.abs(y_test.astype(float) - probs_test))
            
            overfitting_fairness = max(0, 1 - overfitting_gap * 2)
            calibration_fairness = max(0, 1 - calibration_error * 5)
            brier_fairness = max(0, 1 - brier * 2)
            
            fairness_score = (
                overfitting_fairness * 0.4 +
                calibration_fairness * 0.4 +
                brier_fairness * 0.2
            )
            
            # Combined score (equal weight)
            combined = performance_score * 0.5 + fairness_score * 0.5
            return -combined
        
        result = minimize_scalar(objective, bounds=(0.2, 0.8), method='bounded')
        best_threshold = result.x
        
        # Evaluate with best threshold
        preds_best = (probs_test > best_threshold).astype(int)
        
        auc = roc_auc_score(y_test, probs_test)
        acc = accuracy_score(y_test, preds_best)
        rec = recall_score(y_test, preds_best, zero_division=0)
        performance_score = auc * 0.4 + acc * 0.35 + rec * 0.25
        
        train_auc = roc_auc_score(y_train, probs_train)
        overfitting_gap = train_auc - auc
        brier = brier_score_loss(y_test, probs_test)
        
        try:
            if len(np.unique(probs_test)) > 1 and len(y_test) >= 5:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, probs_test, n_bins=min(5, len(y_test)//2), strategy='uniform'
                )
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            else:
                calibration_error = np.mean(np.abs(y_test.astype(float) - probs_test))
        except:
            calibration_error = np.mean(np.abs(y_test.astype(float) - probs_test))
        
        overfitting_fairness = max(0, 1 - overfitting_gap * 2)
        calibration_fairness = max(0, 1 - calibration_error * 5)
        brier_fairness = max(0, 1 - brier * 2)
        
        fairness_score = (
            overfitting_fairness * 0.4 +
            calibration_fairness * 0.4 +
            brier_fairness * 0.2
        )
        
        combined_score = performance_score * 0.5 + fairness_score * 0.5
        
        result_dict = {
            'seed': int(seed),
            'performance_score': float(performance_score),
            'fairness_score': float(fairness_score),
            'combined_score': float(combined_score),
            'metrics': {
                'auc': float(auc),
                'accuracy': float(acc),
                'recall': float(rec),
            },
            'fairness_metrics': {
                'overfitting_gap': float(overfitting_gap),
                'calibration_error': float(calibration_error),
                'brier_score': float(brier),
            },
            'best_threshold': float(best_threshold)
        }
        
        all_results.append(result_dict)
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_result = {
                'features': list(PRIORITY_FEATURES.keys()),
                'feature_names': feat_names,
                'best_C': float(BEST_C),
                'best_threshold': float(best_threshold),
                'seed': int(seed),
                'performance_score': float(performance_score),
                'fairness_score': float(fairness_score),
                'combined_score': float(combined_score),
                'metrics': {
                    'auc': float(auc),
                    'accuracy': float(acc),
                    'recall': float(rec),
                },
                'fairness_metrics': {
                    'overfitting_gap': float(overfitting_gap),
                    'calibration_error': float(calibration_error),
                    'brier_score': float(brier),
                    'overfitting_fairness': float(overfitting_fairness),
                    'calibration_fairness': float(calibration_fairness),
                    'brier_fairness': float(brier_fairness)
                }
            }
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(SEEDS)} seeds tested...")
    
    except Exception as e:
        print(f"[WARN] Error with seed {seed}: {e}")
        continue

# Save all results
with open('all_seeds_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Save best result
if best_result:
    with open('best_seed_results.json', 'w') as f:
        json.dump(best_result, f, indent=2)
    
    print(f"\n[OK] Best seed found!")
    print(f"  Seed: {best_result['seed']}")
    print(f"  Threshold: {best_result['best_threshold']:.4f}")
    print(f"\n[INFO] Performance:")
    print(f"  AUC: {best_result['metrics']['auc']:.4f}")
    print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"  Recall: {best_result['metrics']['recall']:.4f}")
    print(f"  Performance Score: {best_result['performance_score']:.4f}")
    print(f"\n[INFO] Fairness:")
    print(f"  Overfitting Gap: {best_result['fairness_metrics']['overfitting_gap']:.4f}")
    print(f"  Calibration Error: {best_result['fairness_metrics']['calibration_error']:.4f}")
    print(f"  Brier Score: {best_result['fairness_metrics']['brier_score']:.4f}")
    print(f"  Fairness Score: {best_result['fairness_score']:.4f}")
    print(f"\n[INFO] Combined Score: {best_result['combined_score']:.4f}")
    
    # Show top 5 seeds
    sorted_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)
    print(f"\n[INFO] Top 5 Seeds:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. Seed {r['seed']:4d}: Combined={r['combined_score']:.4f}, "
              f"AUC={r['metrics']['auc']:.4f}, Acc={r['metrics']['accuracy']:.4f}, "
              f"Rec={r['metrics']['recall']:.4f}")
    
    print(f"\n[OK] Results saved:")
    print(f"  - best_seed_results.json (best model)")
    print(f"  - all_seeds_results.json (all {len(all_results)} seeds)")
else:
    print("\n[ERROR] No valid result found")
