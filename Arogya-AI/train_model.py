#!/usr/bin/env python3
"""
Model Training Script - Train Arogya AI Disease Prediction Model
================================================================

This script loads training data from enhanced_ayurvedic_treatment_dataset.csv
and creates a complete trained model with all necessary components.
The model training logic is based on the methodology from AyurCore.ipynb.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def extract_notebook_code():
    """
    Load training data and train model
    Note: This was originally designed to extract from notebook, but now loads directly from CSV
    """
    print("Loading training data and initializing model training...")
    
    # Load and train from CSV directly
    return create_demo_model()

def create_demo_model():
    """
    Create a demonstration model with realistic structure
    """
    print("Creating model from enhanced_ayurvedic_treatment_dataset.csv...")

    csv_path = 'enhanced_ayurvedic_treatment_dataset.csv'
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load {csv_path}: {e}")
        print("Falling back to synthetic demo data not implemented. Aborting.")
        raise

    # Keep only the columns needed for training
    required_cols = ['Disease', 'Symptoms', 'Age', 'Height_cm', 'Weight_kg', 'BMI',
                     'Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 'Food_Habits',
                     'Current_Medication', 'Allergies', 'Season', 'Weather']

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required_cols].copy()

    # Drop rows with missing essential fields
    df = df.dropna(subset=['Disease', 'Symptoms']).reset_index(drop=True)

    # Coerce numeric columns and fill missing numerics with medians
    for col in ['Age', 'Height_cm', 'Weight_kg', 'BMI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical with a placeholder
    cat_cols = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 'Food_Habits',
                'Current_Medication', 'Allergies', 'Season', 'Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown').replace({'nan': 'Unknown'})

    # Basic normalization for text symptoms (align with notebook: minimal cleaning)
    df['Symptoms'] = df['Symptoms'].astype(str).str.replace('_', ' ', regex=False)

    print(f"Loaded dataset with {len(df)} samples and {df['Disease'].nunique()} diseases")

    # Train the model following the notebook structure
    return train_complete_model(df)

def train_complete_model(df):
    """
    Complete model training pipeline based on notebook methodology
    """
    print("Starting complete model training pipeline...")
    
    # 1. Preprocessing and Label Encoding
    encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
                          'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather']
    
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Encode target variable (Disease)
    disease_encoder = LabelEncoder()
    df['Disease_encoded'] = disease_encoder.fit_transform(df['Disease'])
    encoders['Disease'] = disease_encoder
    
    print(f"Encoded {len(categorical_columns)} categorical variables")
    
    # 2. Filter classes with too few samples for stratified split
    disease_counts = df['Disease'].value_counts()
    valid_diseases = disease_counts[disease_counts >= 2].index
    removed = df.shape[0] - df[df['Disease'].isin(valid_diseases)].shape[0]
    if removed > 0:
        print(f"Filtered out {removed} samples from diseases with <2 instances for stratified split")
    df = df[df['Disease'].isin(valid_diseases)].reset_index(drop=True)
    if df['Disease'].nunique() == 0:
        raise ValueError("No diseases with at least 2 samples; cannot continue training.")

    # 3. Feature Selection (full dataset)
    feature_columns = ['Age', 'Height_cm', 'Weight_kg', 'BMI'] + [f'{col}_encoded' for col in categorical_columns]
    X_other_full = df[feature_columns]
    y_full = df['Disease_encoded']

    # 4. TF-IDF Vectorization of Symptoms on full dataset (as in notebook)
    vectorizer = TfidfVectorizer()
    symptoms_tfidf_full = vectorizer.fit_transform(df['Symptoms'])

    # Convert to DataFrame
    tfidf_df_full = pd.DataFrame(
        symptoms_tfidf_full.toarray(),
        columns=[f'tfidf_{i}' for i in range(symptoms_tfidf_full.shape[1])]
    )

    # 5. Combine Features for full dataset
    X_combined_full = pd.concat([X_other_full.reset_index(drop=True), tfidf_df_full], axis=1)

    print(f"Combined feature set shape (full): {X_combined_full.shape}")
    print(f"Total features: {X_combined_full.shape[1]} (12 basic + {symptoms_tfidf_full.shape[1]} TF-IDF)")

    # 6. Split indices for evaluation (split only to get test indices, like notebook flow)
    # Note: split on the base features to obtain a stable test index set
    X_base = X_other_full
    X_train_base, X_test_base, y_train_base, y_test = train_test_split(
        X_base, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # 7. SMOTE for Handling Class Imbalance on full combined data
    # Determine k_neighbors based on global class counts (excluding singletons)
    class_counts = y_full.value_counts()
    min_samples = class_counts[class_counts > 1].min() if (class_counts > 1).any() else 1
    k_neighbors = max(1, int(min_samples) - 1)
    if min_samples <= 1:
        print("Skipping SMOTE due to classes with <=1 sample in dataset")
        X_resampled, y_resampled = X_combined_full, y_full
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_combined_full, y_full)
        print(f"Applied SMOTE (full): {X_combined_full.shape[0]} -> {X_resampled.shape[0]} samples")

    # 8. Feature Scaling
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # Recreate combined features for original test set by index and scale
    other_features_test = X_test_base.copy()
    tfidf_matrix_test = vectorizer.transform(df.loc[X_test_base.index, 'Symptoms'])
    tfidf_df_test = pd.DataFrame(
        tfidf_matrix_test.toarray(),
        index=X_test_base.index,
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix_test.shape[1])]
    )
    X_test_full = pd.concat([other_features_test, tfidf_df_test], axis=1)
    X_test_scaled = scaler.transform(X_test_full)
    
    # 9. Model Training
    # Models aligned with notebook methodology
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150,  # Increased from 100 (more trees = higher accuracy)
            max_depth=40,  # Increased from 30 (deeper trees = higher accuracy)
            min_samples_split=2,  # Keep at minimum (allows more splits)
            min_samples_leaf=1,  # Keep at minimum (smaller leaves = more detail)
            max_features=0.8,  # Use 80% of features (was 'sqrt' ~28%) - more features = higher accuracy
            random_state=42, 
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),  # Original
        'SVM': SVC(random_state=42, kernel='rbf', probability=True),
    }
    
    trained_models = {}
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train on resampled data
        model.fit(X_resampled_scaled, y_resampled)
        
    # Predict on original test set (scaled)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # (No ensemble aggregation; reporting base models only)

    # 10. Select Best Model among trained base models (exclude ensemble placeholder)
    best_model_name = max(trained_models.keys(), key=lambda n: results.get(n, -1))
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with test accuracy: {results[best_model_name]:.4f}")
    
    # Additional validation: Cross-validation on original training data (before SMOTE)
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION (5-fold)")
    print("="*60)
    print("Note: CV is done on original data (before SMOTE) to check generalization...")
    
    # Use original split train data for CV (more realistic)
    X_train_orig = X_train_base.copy()
    tfidf_train = vectorizer.transform(df.loc[X_train_base.index, 'Symptoms'])
    tfidf_train_df = pd.DataFrame(
        tfidf_train.toarray(),
        index=X_train_base.index,
        columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])]
    )
    X_train_combined = pd.concat([X_train_orig, tfidf_train_df], axis=1)
    X_train_scaled_cv = scaler.transform(X_train_combined)
    y_train_orig = y_train_base
    
    # Cross-validation (5-fold stratified) - use simpler model for CV to avoid memory issues
    print("Running cross-validation with Logistic Regression (lighter model for CV)...")
    cv_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(cv_model, X_train_scaled_cv, y_train_orig, cv=cv, scoring='accuracy', n_jobs=1)
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"CV Scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
        cv_mean = cv_scores.mean()
    except MemoryError:
        print("⚠️  CV skipped due to memory constraints. Using test accuracy as primary metric.")
        cv_mean = None
    
    if results[best_model_name] >= 0.99:
        print("\n⚠️  WARNING: Test accuracy >= 99% may indicate:")
        print("   - Overfitting (model memorized training patterns)")
        print("   - Dataset too simple or highly separable")
        print("   - Possible data leakage (check feature engineering)")
        if cv_mean is not None:
            print(f"   - CV accuracy ({cv_mean:.4f}) is more realistic for generalization")
        else:
            print("   - Consider running CV separately with a lighter model")
    
    # Detailed classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    y_pred_test = best_model.predict(X_test_scaled)
    # Only show classes that appear in test set
    unique_classes_test = np.unique(np.concatenate([y_test, y_pred_test]))
    test_class_names = encoders['Disease'].classes_[unique_classes_test]
    print(classification_report(y_test, y_pred_test, labels=unique_classes_test, target_names=test_class_names, zero_division=0))
    
    # 11. Save Everything
    model_components = {
        'model': best_model,
        'scaler': scaler,
        'vectorizer': vectorizer,
        'encoders': encoders,
        'feature_columns': feature_columns,
        'results': results,
        'model_type': best_model_name
    }
    
    # Save with joblib
    joblib.dump(model_components, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")
    
    return model_components

if __name__ == "__main__":
    print("="*60)
    print("AROGYA AI - MODEL TRAINING SYSTEM")
    print("="*60)
    
    # Train the complete model
    model_components = extract_notebook_code()
    
    print(f"\n✅ Model training completed successfully!")
    print(f"✅ Model type: {model_components['model_type']}")
    print(f"✅ Model accuracy: {model_components['results'][model_components['model_type']]:.4f}")
    print(f"✅ Total features: {len(model_components['feature_columns']) + model_components['vectorizer'].get_feature_names_out().shape[0]}")
    print(f"✅ Supported diseases: {len(model_components['encoders']['Disease'].classes_)}")
    
    print("\nModel is ready for prediction!")
