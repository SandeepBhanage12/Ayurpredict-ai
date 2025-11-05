#!/usr/bin/env python3
"""
Benchmark multiple classifiers on the Arogya AI pipeline without affecting the production model.

Usage:
  python benchmark_models.py

Notes:
- Reads enhanced_ayurvedic_treatment_dataset.csv
- Reuses the same preprocessing approach as train_model.py
- Runs stratified K-fold CV and prints a leaderboard
- Does NOT write/overwrite random_forest_model.pkl
"""

import warnings
warnings.filterwarnings('ignore')

import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False


CSV_PATH = 'enhanced_ayurvedic_treatment_dataset.csv'


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ['Disease', 'Symptoms', 'Age', 'Height_cm', 'Weight_kg', 'BMI',
                     'Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 'Food_Habits',
                     'Current_Medication', 'Allergies', 'Season', 'Weather']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=['Disease', 'Symptoms']).reset_index(drop=True)

    for col in ['Age', 'Height_cm', 'Weight_kg', 'BMI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 'Food_Habits',
                'Current_Medication', 'Allergies', 'Season', 'Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown').replace({'nan': 'Unknown'})

    df['Symptoms'] = df['Symptoms'].astype(str).str.replace('_', ' ', regex=False)
    return df


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, LabelEncoder], StandardScaler, TfidfVectorizer, List[str]]:
    encoders: Dict[str, LabelEncoder] = {}
    categorical_columns = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
                           'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather']

    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    disease_encoder = LabelEncoder()
    y = disease_encoder.fit_transform(df['Disease'])
    encoders['Disease'] = disease_encoder

    feature_columns = ['Age', 'Height_cm', 'Weight_kg', 'BMI'] + [f'{col}_encoded' for col in categorical_columns]
    X_other = df[feature_columns]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df['Symptoms'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf.shape[1])])

    X_combined = pd.concat([X_other.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    return X_scaled, y, encoders, scaler, vectorizer, feature_columns


def get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        'Random Forest': RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=None),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
        )
    return models


def kfold_benchmark(X: np.ndarray, y: np.ndarray, models: Dict[str, Any], n_splits: int = 5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: Dict[str, List[float]] = {name: [] for name in models}

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        for name, model in models.items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            acc = accuracy_score(yte, pred)
            scores[name].append(acc)

    return {name: float(np.mean(vals)) for name, vals in scores.items()}


def main():
    print("=" * 60)
    print("Arogya AI - Benchmark Suite (non-intrusive)")
    print("=" * 60)
    df = load_dataset(CSV_PATH)
    print(f"Loaded {len(df)} samples across {df['Disease'].nunique()} diseases")

    X, y, _encoders, _scaler, _vectorizer, feature_columns = build_features(df)
    print(f"Feature matrix: {X.shape[0]} x {X.shape[1]} (tabular {len(feature_columns)} + TF-IDF {X.shape[1]-len(feature_columns)})")

    models = get_models()
    print("\nEvaluating models with 5-fold Stratified CV (accuracy)...\n")
    scores = kfold_benchmark(X, y, models, n_splits=5)

    print("Leaderboard (higher is better):")
    for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  - {name}: {score:.4f}")

    print("\nNote: This script does NOT modify or save the production model bundle.")


if __name__ == '__main__':
    main()


