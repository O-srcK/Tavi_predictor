#!/usr/bin/env python3
"""
app.py - Flask web application for TAVI risk prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

app = Flask(__name__)

# Load model
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

model_data = joblib.load('final_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# Load threshold from config.json (easy to modify)
def load_threshold(preset_name=None):
    """Load threshold from config.json, or use model default"""
    config_path = Path('config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # If preset specified, use it
        if preset_name and preset_name in config.get('presets', {}):
            return config['presets'][preset_name]['threshold']
        
        # Otherwise use current preset
        current_preset = config.get('current', 'maximize_combined')
        if current_preset in config.get('presets', {}):
            return config['presets'][current_preset]['threshold']
        elif 'custom' in config:
            return config['custom']['threshold']
        elif 'threshold' in config:
            return config['threshold']  # Old format
    return model_data['threshold']

threshold = load_threshold()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get preset if specified, otherwise use current
        preset_name = data.get('preset', None)
        current_threshold = load_threshold(preset_name)
        
        data = request.json
        
        # Get input values
        fcc = float(data.get('fcc', 0))
        masa_vs_sc = float(data.get('masa_vs_sc', 0))
        
        # Validate inputs
        if fcc < 0 or masa_vs_sc < 0:
            return jsonify({'error': 'Values must be non-negative'}), 400
        
        # Prepare input
        X_input = pd.DataFrame({
            feature_names[0]: [fcc],
            feature_names[1]: [masa_vs_sc]
        })
        
        # Scale
        X_input_sc = scaler.transform(X_input)
        
        # Predict
        prob = model.predict_proba(X_input_sc)[0, 1]
        risk_flag = prob > current_threshold
        
        result = {
            'risk_probability': float(prob),
            'risk_percentage': float(prob * 100),
            'risk_flag': bool(risk_flag),
            'risk_level': 'HIGH RISK' if risk_flag else 'LOW RISK',
            'threshold_used': float(current_threshold),
            'inputs': {
                'fcc': fcc,
                'masa_vs_sc': masa_vs_sc
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information"""
    current_threshold = load_threshold()
    
    # Load config to get preset info
    config_path = Path('config.json')
    presets_info = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            presets_info = config.get('presets', {})
            current_preset = config.get('current', 'maximize_combined')
    
    return jsonify({
        'features': model_data['config']['features'],
        'feature_names': feature_names,
        'threshold': float(current_threshold),
        'metrics': model_data['config']['metrics'],
        'presets': presets_info,
        'current_preset': current_preset if config_path.exists() else None
    })

@app.route('/set_preset', methods=['POST'])
def set_preset():
    """Set the current preset threshold"""
    try:
        data = request.json
        preset_name = data.get('preset')
        
        config_path = Path('config.json')
        if not config_path.exists():
            return jsonify({'error': 'config.json not found'}), 404
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if preset_name == 'custom':
            # Set custom threshold
            custom_threshold = float(data.get('threshold', 0.45))
            config['custom']['threshold'] = custom_threshold
            config['custom']['threshold_percentage'] = custom_threshold * 100
            config['current'] = 'custom'
        elif preset_name in config.get('presets', {}):
            config['current'] = preset_name
        else:
            return jsonify({'error': f'Unknown preset: {preset_name}'}), 400
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({'success': True, 'current_preset': config['current']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("TAVI Risk Prediction Web App")
    print("="*80)
    print("\n[INFO] Starting server...")
    print("[INFO] Open http://localhost:5000 in your browser")
    print("\n[INFO] Press Ctrl+C to stop the server")
    print("="*80)
    app.run(debug=True, host='0.0.0.0', port=5000)
