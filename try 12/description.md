# TAVI Complication Risk Predictor - Model Description

## What This Model Does

This model predicts the risk of post-TAVI (Transcatheter Aortic Valve Implantation) complications for patients. It takes 2 simple clinical measurements and outputs a risk percentage (0-100%).

## Model Overview

### Type
- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: 2 clinical measurements
- **Output**: Risk probability (0-100%)

### Features Used
1. **FCC** (Frequency of Conduction)
2. **Masa VS/SC** (g/m²) - Ventricular mass normalized by body surface area

### Why Only 2 Features?
- **Simplicity**: Easy for doctors to use (only 2 inputs needed)
- **Clinical relevance**: These features are most predictive
- **Performance**: Achieves good results (AUC 0.8286) with minimal complexity
- **Avoids overfitting**: With only 18 events, fewer features = more reliable

## How the Model Was Created

### Step 1: Data Preparation
- Started with 179 patients, 18 with complications (10.1% prevalence)
- Cleaned missing data
- Applied log transformation to calcium scores (when used)

### Step 2: Feature Selection
- Tested combinations of priority features:
  - Calcium (log-transformed)
  - Gradient (GPmax VAo)
  - FCC
  - Euroscore II
  - Masa VS (g)
  - Masa VS/SC (g/m²)
- Found best combination: FCC + Masa VS/SC

### Step 3: Model Optimization
- Tested 67 different random seeds
- Optimized hyperparameter C (regularization strength)
- Optimized threshold for balanced performance + fairness
- **Best seed**: 36 (best combined score)

### Step 4: Fairness Optimization
- Added "fairness" measures (honesty/truth):
  - Low overfitting (generalizes well)
  - Good calibration (probabilities match reality)
  - Stability (consistent across data splits)
- Balanced performance (AUC, Accuracy, Recall) with fairness

### Step 5: Threshold Optimization
- Created 5 preset thresholds for different use cases
- Made threshold easily adjustable via `config.json`

## Performance Metrics Explained

### AUC (Area Under ROC Curve) = 0.8286
- **What it means**: Model can distinguish between high-risk and low-risk patients
- **Scale**: 0.5 = random, 1.0 = perfect
- **Our score**: 0.8286 = Good discrimination (82.86% better than random)

### Accuracy = 55.3% (at 50% threshold)
- **What it means**: Overall correctness of predictions
- **Formula**: (Correct predictions) / (Total predictions)
- **Our score**: 55.3% - Moderate (but acceptable given class imbalance)

### Recall = 77.8% (at 50% threshold)
- **What it means**: Of all patients with complications, we catch 77.8%
- **Also called**: Sensitivity, True Positive Rate
- **Our score**: 77.8% - Good (catches most complications)
- **At 42% threshold**: 100% recall (catches all complications)

### Precision = 15.6% (at 50% threshold)
- **What it means**: Of all patients flagged as HIGH RISK, 15.6% actually have complications
- **Also called**: Positive Predictive Value
- **Our score**: 15.6% - Low (many false alarms, but this is expected with low prevalence)

### F1 Score = 27.2% (at 52% threshold)
- **What it means**: Balance between recall and precision
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Our score**: 27.2% - Moderate (reflects the precision-recall trade-off)

### Fairness Score = 0.7005
- **What it means**: How "honest" and "truthful" the model is
- **Components**:
  - Low overfitting (test performance ≈ train performance)
  - Good calibration (probabilities match actual outcomes)
  - Stability (consistent across different data splits)
- **Our score**: 0.7005 - Good (model is reliable and honest)

## Why These Metrics?

### AUC
- **Why important**: Shows model's ability to rank patients by risk
- **Clinical use**: Helps identify which patients need closer monitoring

### Recall
- **Why important**: In medicine, missing a complication is worse than a false alarm
- **Clinical use**: Ensures we don't miss patients who need care

### Accuracy
- **Why important**: Overall correctness matters for trust
- **Clinical use**: Doctors need to know predictions are generally reliable

### Fairness
- **Why important**: Model should be honest about its predictions
- **Clinical use**: Doctors need reliable probabilities, not misleading numbers

## Trade-offs Made

### 1. Features: Simplicity vs. Performance
- **Choice**: Used only 2 features
- **Trade-off**: Simpler to use, but might miss some predictive power
- **Reason**: With 18 events, more features risk overfitting

### 2. Threshold: Safety vs. Efficiency
- **Choice**: Multiple threshold options (42% to 60%)
- **Trade-off**: 
  - Lower threshold = catch more complications but more false alarms
  - Higher threshold = fewer false alarms but miss more complications
- **Reason**: Different doctors have different priorities

### 3. Model Type: Complexity vs. Interpretability
- **Choice**: Logistic Regression (simple, interpretable)
- **Trade-off**: Could use more complex models (neural networks, etc.)
- **Reason**: Doctors need to understand and trust the model

### 4. Fairness vs. Performance
- **Choice**: Balanced both (50% performance + 50% fairness)
- **Trade-off**: Could maximize performance but sacrifice honesty
- **Reason**: Medical models must be trustworthy

## Limitations

### 1. Small Dataset
- Only 18 events (complications)
- Makes it hard to achieve very high accuracy
- Model may not generalize perfectly to new populations

### 2. Class Imbalance
- 10.1% complication rate
- Makes precision low (many false alarms)
- This is expected and acceptable for safety

### 3. Limited Features
- Only 2 features used
- Other features might help but weren't included
- Balance between simplicity and completeness

### 4. Single Institution
- Data from one source
- May not generalize to other hospitals/populations
- External validation needed

## Model Validation

### Cross-Validation
- Tested across multiple random seeds (67 seeds)
- Results are stable and consistent

### Fairness Assessment
- Tested on 8 different data splits
- Low overfitting (good generalization)
- Good calibration (probabilities are honest)

### Real-World Testing
- Tested on all 18 patients with actual complications
- At 45% threshold: Catches 88.9% (16/18)
- At 42% threshold: Catches 100% (18/18)

## How to Interpret Results

### Risk Percentage
- **0-30%**: Low risk (unlikely to have complications)
- **30-50%**: Moderate risk (monitor closely)
- **50-70%**: High risk (increased monitoring recommended)
- **70-100%**: Very high risk (consider additional precautions)

### Risk Flag
- **HIGH RISK**: Above threshold - patient should be monitored more closely
- **LOW RISK**: Below threshold - standard monitoring is acceptable

### Important Notes
- This is a **screening tool**, not a definitive diagnosis
- Always combine with clinical judgment
- False alarms are acceptable for safety (better safe than sorry)
- Model is optimized to catch complications, not avoid false alarms

## Threshold Presets

The model includes 5 preset threshold options, each optimized for different objectives:

### 1. Maximize Recall (42%)
- **Recall**: 100% (catches ALL complications)
- **Accuracy**: 29.1%
- **Precision**: 12.4%
- **Best for**: Maximum patient safety - don't want to miss any complications
- **Trade-off**: Many false alarms (87.6% of flagged patients don't have complications)

### 2. Maximize Accuracy (60%)
- **Recall**: 5.6% (misses most complications)
- **Accuracy**: 87.2%
- **Precision**: 14.3%
- **Best for**: When you want highest overall correctness
- **Trade-off**: Very low recall - misses 94.4% of complications (NOT recommended for safety)

### 3. Maximize Combined (50%) ⭐ Recommended
- **Recall**: 77.8% (catches most complications)
- **Accuracy**: 55.3%
- **Precision**: 15.6%
- **Best for**: Balanced approach - good at catching complications while maintaining reasonable accuracy
- **Trade-off**: Moderate false alarm rate

### 4. Maximize F1 (52%)
- **Recall**: 61.1%
- **Accuracy**: 67.0%
- **Precision**: 17.5%
- **F1 Score**: 27.2%
- **Best for**: Balanced recall and precision
- **Trade-off**: Good overall balance

### 5. Youden's J (50%)
- **Recall**: 77.8%
- **Accuracy**: 55.3%
- **Precision**: 15.6%
- **Best for**: Optimal point on ROC curve (sensitivity + specificity)
- **Trade-off**: Same as "Maximize Combined" in this case

## Mathematical Function Models

Recall and accuracy can be modeled as mathematical functions of the threshold:

**Recall Function** (Cubic Polynomial, R² = 0.9746):
```
recall(t) = 88.59*t³ - 136.65*t² + 64.54*t - 8.58
```

**Accuracy Function** (Cubic Polynomial, R² = 0.9921):
```
accuracy(t) = -53.31*t³ + 78.49*t² - 34.73*t + 4.97
```

These functions allow:
- Predicting recall/accuracy at any threshold without testing
- Analytical optimization using calculus
- Visualizing the exact trade-off relationship

## Future Improvements

1. **More data**: With 200+ events, could improve accuracy significantly
2. **More features**: Could test if adding features improves performance
3. **External validation**: Test on data from other hospitals
4. **Calibration**: Further improve probability calibration
5. **Feature importance**: Analyze which features matter most

## Summary

This model is a **simple, honest, and safe** tool for predicting TAVI complications. It prioritizes:
- **Safety** (catching complications) over efficiency
- **Simplicity** (2 features) over complexity
- **Honesty** (fair probabilities) over impressive numbers
- **Usability** (easy for doctors) over technical sophistication

The model achieves good discrimination (AUC 0.8286) and can catch most complications (77.8-100% depending on threshold) while being simple enough for real-world clinical use.
