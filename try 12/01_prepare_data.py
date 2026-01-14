#!/usr/bin/env python3
"""
01_prepare_data.py - Data Preparation
"""

import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path

print("="*80)
print("01_PREPARE_DATA - try 11")
print("="*80)

# Setup
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Copy original CSV
source_csv = Path("../main_tavi_training.csv")
target_csv = Path("main_tavi_training.csv")

if not source_csv.exists():
    raise FileNotFoundError(f"Source CSV not found: {source_csv}")

shutil.copy(source_csv, target_csv)
print(f"[OK] Copied: {source_csv} -> {target_csv}")

# Load data
df_raw = pd.read_csv(target_csv)
y = df_raw['target']

print(f"\n[INFO] Dataset Info:")
print(f"  Total patients: {len(df_raw)}")
print(f"  Events: {y.sum()} ({y.mean():.1%})")

# Load cleaned features
cleaned_path = Path("../try 6/v2/cleaned_features.csv")
if cleaned_path.exists():
    print(f"\n[OK] Using cleaned features: {cleaned_path}")
    X_clean = pd.read_csv(cleaned_path)
    X_clean.to_csv("cleaned_features.csv", index=False)
    print(f"[OK] Copied cleaned features")
else:
    missing_pct = df_raw.isnull().sum() / len(df_raw)
    keep_cols = missing_pct[missing_pct <= 0.30].index.tolist()
    if 'target' in keep_cols:
        keep_cols.remove('target')
    X_clean = df_raw[keep_cols].copy()
    X_clean.to_csv("cleaned_features.csv", index=False)
    print(f"[OK] Created cleaned features: {len(keep_cols)} features")

print(f"\n[OK] Data preparation complete!")
