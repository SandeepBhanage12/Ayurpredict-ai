#!/usr/bin/env python3
"""
Diagnostic script to check the model file format
"""
import joblib
import os

model_path = 'random_forest_model.pkl'

if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' not found!")
    exit(1)

print(f"Checking model file: {model_path}")
print("=" * 60)

try:
    loaded_data = joblib.load(model_path)
    
    print(f"[OK] File loaded successfully")
    print(f"   Type: {type(loaded_data)}")
    
    if isinstance(loaded_data, dict):
        print(f"\n[OK] Model is a dictionary (correct format)")
        print(f"   Keys: {list(loaded_data.keys())}")
        
        # Check for required components
        required_keys = ['model', 'scaler', 'vectorizer', 'encoders', 'feature_columns']
        missing = [k for k in required_keys if k not in loaded_data]
        
        if missing:
            print(f"\n[ERROR] Missing required components: {missing}")
        else:
            print(f"\n[OK] All required components present")
        
        # Check encoders
        if 'encoders' in loaded_data:
            encoders = loaded_data['encoders']
            print(f"\n   Encoders type: {type(encoders)}")
            if isinstance(encoders, dict):
                print(f"   [OK] Encoders is a dictionary")
                print(f"   Encoder keys: {list(encoders.keys())}")
            else:
                print(f"   [ERROR] Encoders should be a dictionary, got {type(encoders)}")
        
        # Check feature_columns
        if 'feature_columns' in loaded_data:
            fc = loaded_data['feature_columns']
            print(f"\n   Feature columns type: {type(fc)}")
            if isinstance(fc, list):
                print(f"   [OK] Feature columns is a list")
                print(f"   Number of features: {len(fc)}")
            else:
                print(f"   [WARNING] Feature columns type: {type(fc)}")
        
    elif isinstance(loaded_data, list):
        print(f"\n[ERROR] Model is a LIST (incorrect format)")
        print(f"   List length: {len(loaded_data)}")
        if loaded_data:
            print(f"   First element type: {type(loaded_data[0])}")
            print(f"   First element: {loaded_data[0] if len(str(loaded_data[0])) < 100 else '... (too long)'}")
        print(f"\n   [WARNING] The model file is in the wrong format.")
        print(f"   This usually happens if it was saved with an older version of the code.")
        print(f"   Solution: Retrain the model using 'python train_model.py'")
    else:
        print(f"\n[ERROR] Unexpected format: {type(loaded_data)}")
        print(f"   Expected: dict")
        print(f"   Solution: Retrain the model using 'python train_model.py'")
        
except Exception as e:
    print(f"\n[ERROR] Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
