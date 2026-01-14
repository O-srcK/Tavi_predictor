#!/usr/bin/env python3
"""
09_model_recall_accuracy_functions.py - Model Recall and Accuracy as Functions of Threshold
Fits mathematical functions to predict recall(t) and accuracy(t)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import recall_score, accuracy_score, precision_score
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*80)
print("09_MODEL_RECALL_ACCURACY_FUNCTIONS")
print("="*80)

# Setup - change to script directory
script_dir = Path(__file__).parent.absolute()
import os
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

# Generate data points: calculate recall and accuracy at many thresholds
thresholds = np.arange(0.30, 0.70, 0.01)
recall_values = []
accuracy_values = []
precision_values = []

print("\n[INFO] Calculating recall and accuracy at different thresholds...")

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    recall = recall_score(y, preds, zero_division=0)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    
    recall_values.append(recall)
    accuracy_values.append(accuracy)
    precision_values.append(precision)

recall_values = np.array(recall_values)
accuracy_values = np.array(accuracy_values)
precision_values = np.array(precision_values)

print(f"[OK] Calculated {len(thresholds)} data points")

# Define function forms to try
def recall_func_linear(t, a, b):
    """Linear: recall(t) = a*t + b"""
    return a * t + b

def recall_func_poly2(t, a, b, c):
    """Quadratic: recall(t) = a*t^2 + b*t + c"""
    return a * t**2 + b * t + c

def recall_func_poly3(t, a, b, c, d):
    """Cubic: recall(t) = a*t^3 + b*t^2 + c*t + d"""
    return a * t**3 + b * t**2 + c * t + d

def recall_func_sigmoid(t, a, b, c, d):
    """Sigmoid-like: recall(t) = a / (1 + exp(b*(t-c))) + d"""
    return a / (1 + np.exp(b * (t - c))) + d

def accuracy_func_linear(t, a, b):
    """Linear: accuracy(t) = a*t + b"""
    return a * t + b

def accuracy_func_poly2(t, a, b, c):
    """Quadratic: accuracy(t) = a*t^2 + b*t + c"""
    return a * t**2 + b * t + c

def accuracy_func_poly3(t, a, b, c, d):
    """Cubic: accuracy(t) = a*t^3 + b*t^2 + c*t + d"""
    return a * t**3 + b * t**2 + c * t + d

# Fit recall function
print("\n[INFO] Fitting recall function...")
recall_fits = {}

try:
    popt, _ = curve_fit(recall_func_poly3, thresholds, recall_values, maxfev=5000)
    recall_fits['poly3'] = {'params': popt, 'func': recall_func_poly3, 'name': 'Cubic'}
    recall_pred = recall_func_poly3(thresholds, *popt)
    recall_r2 = 1 - np.sum((recall_values - recall_pred)**2) / np.sum((recall_values - np.mean(recall_values))**2)
    recall_fits['poly3']['r2'] = recall_r2
    print(f"  Cubic: R² = {recall_r2:.4f}")
except:
    pass

try:
    popt, _ = curve_fit(recall_func_poly2, thresholds, recall_values, maxfev=5000)
    recall_fits['poly2'] = {'params': popt, 'func': recall_func_poly2, 'name': 'Quadratic'}
    recall_pred = recall_func_poly2(thresholds, *popt)
    recall_r2 = 1 - np.sum((recall_values - recall_pred)**2) / np.sum((recall_values - np.mean(recall_values))**2)
    recall_fits['poly2']['r2'] = recall_r2
    print(f"  Quadratic: R² = {recall_r2:.4f}")
except:
    pass

# Fit accuracy function
print("\n[INFO] Fitting accuracy function...")
accuracy_fits = {}

try:
    popt, _ = curve_fit(accuracy_func_poly3, thresholds, accuracy_values, maxfev=5000)
    accuracy_fits['poly3'] = {'params': popt, 'func': accuracy_func_poly3, 'name': 'Cubic'}
    accuracy_pred = accuracy_func_poly3(thresholds, *popt)
    accuracy_r2 = 1 - np.sum((accuracy_values - accuracy_pred)**2) / np.sum((accuracy_values - np.mean(accuracy_values))**2)
    accuracy_fits['poly3']['r2'] = accuracy_r2
    print(f"  Cubic: R² = {accuracy_r2:.4f}")
except:
    pass

try:
    popt, _ = curve_fit(accuracy_func_poly2, thresholds, accuracy_values, maxfev=5000)
    accuracy_fits['poly2'] = {'params': popt, 'func': accuracy_func_poly2, 'name': 'Quadratic'}
    accuracy_pred = accuracy_func_poly2(thresholds, *popt)
    accuracy_r2 = 1 - np.sum((accuracy_values - accuracy_pred)**2) / np.sum((accuracy_values - np.mean(accuracy_values))**2)
    accuracy_fits['poly2']['r2'] = accuracy_r2
    print(f"  Quadratic: R² = {accuracy_r2:.4f}")
except:
    pass

# Select best fits
best_recall_fit = max(recall_fits.items(), key=lambda x: x[1]['r2']) if recall_fits else None
best_accuracy_fit = max(accuracy_fits.items(), key=lambda x: x[1]['r2']) if accuracy_fits else None

print("\n" + "="*80)
print("FITTED FUNCTIONS")
print("="*80)

if best_recall_fit:
    fit_name, fit_data = best_recall_fit
    params = fit_data['params']
    print(f"\n[INFO] Recall(t) - Best Fit: {fit_data['name']} (R² = {fit_data['r2']:.4f})")
    if fit_name == 'poly3':
        print(f"  recall(t) = {params[0]:.2f}*t³ + {params[1]:.2f}*t² + {params[2]:.2f}*t + {params[3]:.2f}")
    elif fit_name == 'poly2':
        print(f"  recall(t) = {params[0]:.2f}*t² + {params[1]:.2f}*t + {params[2]:.2f}")

if best_accuracy_fit:
    fit_name, fit_data = best_accuracy_fit
    params = fit_data['params']
    print(f"\n[INFO] Accuracy(t) - Best Fit: {fit_data['name']} (R² = {fit_data['r2']:.4f})")
    if fit_name == 'poly3':
        print(f"  accuracy(t) = {params[0]:.2f}*t³ + {params[1]:.2f}*t² + {params[2]:.2f}*t + {params[3]:.2f}")
    elif fit_name == 'poly2':
        print(f"  accuracy(t) = {params[0]:.2f}*t² + {params[1]:.2f}*t + {params[2]:.2f}")

# Save functions
functions_data = {
    'recall_function': {
        'type': best_recall_fit[1]['name'] if best_recall_fit else None,
        'parameters': best_recall_fit[1]['params'].tolist() if best_recall_fit else None,
        'r2': float(best_recall_fit[1]['r2']) if best_recall_fit else None,
        'formula': None
    },
    'accuracy_function': {
        'type': best_accuracy_fit[1]['name'] if best_accuracy_fit else None,
        'parameters': best_accuracy_fit[1]['params'].tolist() if best_accuracy_fit else None,
        'r2': float(best_accuracy_fit[1]['r2']) if best_accuracy_fit else None,
        'formula': None
    },
    'data_points': {
        'thresholds': thresholds.tolist(),
        'recall': recall_values.tolist(),
        'accuracy': accuracy_values.tolist(),
        'precision': precision_values.tolist()
    }
}

# Add formulas
if best_recall_fit:
    params = best_recall_fit[1]['params']
    if best_recall_fit[0] == 'poly3':
        functions_data['recall_function']['formula'] = f"{params[0]:.6f}*t^3 + {params[1]:.6f}*t^2 + {params[2]:.6f}*t + {params[3]:.6f}"
    elif best_recall_fit[0] == 'poly2':
        functions_data['recall_function']['formula'] = f"{params[0]:.6f}*t^2 + {params[1]:.6f}*t + {params[2]:.6f}"

if best_accuracy_fit:
    params = best_accuracy_fit[1]['params']
    if best_accuracy_fit[0] == 'poly3':
        functions_data['accuracy_function']['formula'] = f"{params[0]:.6f}*t^3 + {params[1]:.6f}*t^2 + {params[2]:.6f}*t + {params[3]:.6f}"
    elif best_accuracy_fit[0] == 'poly2':
        functions_data['accuracy_function']['formula'] = f"{params[0]:.6f}*t^2 + {params[1]:.6f}*t + {params[2]:.6f}"

with open('threshold_functions.json', 'w') as f:
    json.dump(functions_data, f, indent=2)

print(f"\n[OK] Functions saved to threshold_functions.json")

# Test predictions
print("\n[INFO] Testing function predictions:")
test_thresholds = [0.40, 0.45, 0.50, 0.55]
print(f"\n{'Threshold':<12} {'Actual Recall':<15} {'Predicted Recall':<18} {'Actual Acc':<15} {'Predicted Acc':<18}")
print("-" * 80)

for t in test_thresholds:
    # Actual
    preds = (probs > t).astype(int)
    actual_recall = recall_score(y, preds, zero_division=0)
    actual_acc = accuracy_score(y, preds)
    
    # Predicted
    if best_recall_fit:
        pred_recall = best_recall_fit[1]['func'](t, *best_recall_fit[1]['params'])
    else:
        pred_recall = None
    
    if best_accuracy_fit:
        pred_acc = best_accuracy_fit[1]['func'](t, *best_accuracy_fit[1]['params'])
    else:
        pred_acc = None
    
    pred_recall_str = f"{pred_recall:.2%}" if pred_recall is not None else "N/A"
    pred_acc_str = f"{pred_acc:.2%}" if pred_acc is not None else "N/A"
    print(f"{t*100:>6.0f}%      {actual_recall:>6.2%}         {pred_recall_str:>18}   "
          f"{actual_acc:>6.2%}         {pred_acc_str:>18}")

# Plot
print("\n[INFO] Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot recall
ax1.scatter(thresholds, recall_values, alpha=0.5, label='Actual Data', s=20)
if best_recall_fit:
    t_smooth = np.linspace(thresholds.min(), thresholds.max(), 200)
    recall_smooth = best_recall_fit[1]['func'](t_smooth, *best_recall_fit[1]['params'])
    ax1.plot(t_smooth, recall_smooth, 'r-', label=f"Fitted {best_recall_fit[1]['name']} (R²={best_recall_fit[1]['r2']:.3f})", linewidth=2)
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Recall')
ax1.set_title('Recall as Function of Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot accuracy
ax2.scatter(thresholds, accuracy_values, alpha=0.5, label='Actual Data', s=20)
if best_accuracy_fit:
    t_smooth = np.linspace(thresholds.min(), thresholds.max(), 200)
    accuracy_smooth = best_accuracy_fit[1]['func'](t_smooth, *best_accuracy_fit[1]['params'])
    ax2.plot(t_smooth, accuracy_smooth, 'b-', label=f"Fitted {best_accuracy_fit[1]['name']} (R²={best_accuracy_fit[1]['r2']:.3f})", linewidth=2)
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy as Function of Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_functions.png', dpi=150, bbox_inches='tight')
print(f"[OK] Visualization saved: threshold_functions.png")

print("\n[INFO] Yes, recall and accuracy ARE functions of threshold!")
print("  - We can model them mathematically")
print("  - The fitted functions can predict recall/accuracy at any threshold")
print("  - This allows analytical optimization (finding optimal thresholds)")
print("  - The functions show the trade-off relationship clearly")

print("\n" + "="*80)
