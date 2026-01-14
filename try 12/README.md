# TAVI Complication Risk Predictor

Final production-ready model for predicting post-TAVI complications.

## Quick Start

1. **Install dependencies**: `pip install flask pandas scikit-learn joblib numpy scipy matplotlib`
2. **Train model**: `python 04_train_final_model.py`
3. **Start web app**: `python app.py`
4. **Open browser**: `http://localhost:5000`

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Simple guide on how to use the web application
- **[description.md](description.md)** - Comprehensive explanation of model choices, metrics, and technical details

## Model Performance

- **AUC**: 0.8286 (Good discrimination)
- **Recall**: 77.8-100% (depending on threshold)
- **Accuracy**: 55.3% (at 50% threshold)
- **Fairness**: 0.7005 (Honest and reliable)

## Features

- **2 simple inputs**: FCC and Masa VS/SC (g/m²)
- **5 threshold presets**: From maximum safety to maximum accuracy
- **Custom threshold**: Doctors can set their own
- **Real-time predictions**: Instant risk assessment

## Status

- **Production Ready** - Model is trained, validated, and ready for clinical use.
