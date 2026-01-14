# TAVI Complication Risk Predictor - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install flask pandas scikit-learn joblib numpy scipy matplotlib
```

### 2. Train the Model
```bash
python 04_train_final_model.py
```

### 3. Start the Web Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to: `http://localhost:5000`

## Using the Web Interface

### Step 1: Select Threshold Preset
Choose from 5 preset options or use custom:
- **Maximize Recall (42%)** - Catches all complications (100% recall)
- **Maximize Accuracy (60%)** - Highest correctness (but low recall)
- **Maximize Combined (50%)** - ⭐ Recommended balanced option
- **Maximize F1 (52%)** - Balanced recall and precision
- **Youden's J (50%)** - Optimal ROC point
- **Custom** - Enter your own threshold (0-100%)

### Step 2: Enter Patient Data
- **FCC** (Frequency of Conduction): Enter the patient's FCC value
- **Masa VS/SC (g/m²)**: Enter the normalized ventricular mass value

### Step 3: Get Prediction
Click "Calculate Risk" to see:
- Risk percentage (0-100%)
- Risk level (HIGH RISK or LOW RISK)
- Detailed breakdown

## Understanding the Results

### Risk Percentage
- **0-100%**: Probability of post-TAVI complication
- Higher = more likely to have complications

### Risk Level
- **HIGH RISK**: Risk percentage is above the selected threshold
- **LOW RISK**: Risk percentage is below the selected threshold

### Threshold
- The cutoff point for flagging patients
- Lower threshold = more patients flagged (more sensitive)
- Higher threshold = fewer patients flagged (more specific)

## Threshold Selection Guide

### For Maximum Safety
- Use **"Maximize Recall"** (42%)
- Catches 100% of complications
- But: 87.6% of flagged patients are false alarms

### For Balanced Use
- Use **"Maximize Combined"** (50%) ⭐
- Catches 77.8% of complications
- 55.3% overall accuracy
- Good balance between safety and efficiency

### For Fewer False Alarms
- Use **"Maximize F1"** (52%)
- 61.1% recall, 67.0% accuracy
- Better precision (fewer false alarms)

## Modifying Threshold

### Option 1: Web Interface
- Select preset from dropdown
- Or choose "Custom" and enter value

### Option 2: Edit config.json
```json
{
  "current": "maximize_combined",  ← Change this
  "custom": {
    "threshold": 0.45  ← Or set custom value
  }
}
```

## API Usage

### Predict Risk
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"fcc": 65.0, "masa_vs_sc": 200.0, "preset": "maximize_combined"}'
```

### Get Model Info
```bash
curl http://localhost:5000/model_info
```

### Set Preset
```bash
curl -X POST http://localhost:5000/set_preset \
  -H "Content-Type: application/json" \
  -d '{"preset": "maximize_recall"}'
```

## Files Overview

- `app.py` - Web application server
- `final_model.pkl` - Trained model (created by step 2)
- `config.json` - Threshold configuration
- `templates/index.html` - Web interface
- `threshold_functions.json` - Mathematical functions for recall/accuracy

## Troubleshooting

### Model not found
- Run `python 04_train_final_model.py` first

### Port already in use
- Change port in `app.py`: `app.run(port=5001)`

### Predictions seem wrong
- Check that FCC and Masa VS/SC values are in correct units
- Verify threshold preset is appropriate for your use case

## Threshold Presets Explained

### For Maximum Safety
- Use **"Maximize Recall"** (42%)
- Catches 100% of complications
- But: 87.6% of flagged patients are false alarms

### For Balanced Use (Recommended)
- Use **"Maximize Combined"** (50%) ⭐
- Catches 77.8% of complications
- 55.3% overall accuracy
- Good balance between safety and efficiency

### For Fewer False Alarms
- Use **"Maximize F1"** (52%)
- 61.1% recall, 67.0% accuracy
- Better precision (fewer false alarms)

### Other Presets
- **Maximize Accuracy (60%)**: Highest correctness but very low recall (5.6%) - not recommended for safety
- **Youden's J (50%)**: Optimal ROC point, same as "Maximize Combined" in this case
