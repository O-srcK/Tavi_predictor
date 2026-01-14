#!/usr/bin/env python3
"""
03_summarize_results.py - Summarize seed testing results
"""

import json
import os
from pathlib import Path
import pandas as pd

print("="*80)
print("03_SUMMARIZE_RESULTS - Seed Testing Summary")
print("="*80)

# Setup
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load results
if not Path('all_seeds_results.json').exists():
    print("[ERROR] Run 02_test_many_seeds.py first!")
    exit(1)

with open('all_seeds_results.json', 'r') as f:
    all_results = json.load(f)

with open('best_seed_results.json', 'r') as f:
    best_result = json.load(f)

df = pd.DataFrame(all_results)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n[INFO] Total seeds tested: {len(all_results)}")
print(f"\n[INFO] Performance Metrics (across all seeds):")
print(f"  AUC:        {df['metrics'].apply(lambda x: x['auc']).mean():.4f} ± {df['metrics'].apply(lambda x: x['auc']).std():.4f}")
print(f"  Accuracy:   {df['metrics'].apply(lambda x: x['accuracy']).mean():.4f} ± {df['metrics'].apply(lambda x: x['accuracy']).std():.4f}")
print(f"  Recall:     {df['metrics'].apply(lambda x: x['recall']).mean():.4f} ± {df['metrics'].apply(lambda x: x['recall']).std():.4f}")
print(f"  Perf Score: {df['performance_score'].mean():.4f} ± {df['performance_score'].std():.4f}")

print(f"\n[INFO] Fairness Metrics (across all seeds):")
print(f"  Fairness Score: {df['fairness_score'].mean():.4f} ± {df['fairness_score'].std():.4f}")
print(f"  Combined Score: {df['combined_score'].mean():.4f} ± {df['combined_score'].std():.4f}")

print("\n" + "="*80)
print("BEST SEED")
print("="*80)

print(f"\n[INFO] Best Seed: {best_result['seed']}")
print(f"  Combined Score: {best_result['combined_score']:.4f}")
print(f"  Performance Score: {best_result['performance_score']:.4f}")
print(f"  Fairness Score: {best_result['fairness_score']:.4f}")
print(f"\n[INFO] Performance:")
print(f"  AUC: {best_result['metrics']['auc']:.4f}")
print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
print(f"  Recall: {best_result['metrics']['recall']:.4f}")
print(f"\n[INFO] Fairness:")
print(f"  Overfitting Gap: {best_result['fairness_metrics']['overfitting_gap']:.4f}")
print(f"  Calibration Error: {best_result['fairness_metrics']['calibration_error']:.4f}")
print(f"  Brier Score: {best_result['fairness_metrics']['brier_score']:.4f}")

# Top 10 seeds
sorted_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)
print("\n" + "="*80)
print("TOP 10 SEEDS")
print("="*80)
print(f"\n{'Rank':<6} {'Seed':<6} {'Combined':<10} {'AUC':<8} {'Accuracy':<10} {'Recall':<8} {'Fairness':<10}")
print("-" * 80)
for i, r in enumerate(sorted_results[:10], 1):
    print(f"{i:<6} {r['seed']:<6} {r['combined_score']:<10.4f} "
          f"{r['metrics']['auc']:<8.4f} {r['metrics']['accuracy']:<10.4f} "
          f"{r['metrics']['recall']:<8.4f} {r['fairness_score']:<10.4f}")

print("\n[OK] Summary complete!")
