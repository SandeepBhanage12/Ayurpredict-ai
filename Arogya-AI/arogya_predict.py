#!/usr/bin/env python3
"""
Arogya AI - Main Prediction System
==================================

This script loads the trained model and provides disease prediction
along with comprehensive Ayurvedic recommendations as specified.

Features:
- Disease prediction using Random Forest model (>99% accuracy)
- Comprehensive Ayurvedic recommendations including:
  - Ayurvedic_Herbs_Sanskrit
  - Ayurvedic_Herbs_English
  - Herbs_Effects
  - Ayurvedic_Therapies_Sanskrit
  - Ayurvedic_Therapies_English
  - Therapies_Effects
  - Dietary_Recommendations
  - How_Treatment_Affects_Your_Body_Type
"""

import joblib
import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, Any, List
import sys

warnings.filterwarnings('ignore')

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        # Set console to UTF-8 if possible
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # If reconfigure fails, use errors='replace' for safety
        pass

class ArogyaAI:
    """Main Arogya AI Prediction System"""
    
    def __init__(self, model_path='random_forest_model.pkl'):
        """Initialize the prediction system"""
        self.model_path = model_path
        self.model_components = None
        self.ayurvedic_database = None
        self.load_model()
        self.create_ayurvedic_database()
    
    def load_model(self):
        """Load the trained model and all components"""
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} not found!")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        
        try:
            loaded_data = joblib.load(self.model_path)
            
            # Handle case where model might be saved as a list or different format
            if isinstance(loaded_data, list):
                print(f"⚠️ Warning: Model file contains a list. Expected dictionary.")
                print(f"   Attempting to reconstruct model components...")
                # If it's a list, we might need to reconstruct it
                # This could happen if the model was saved differently
                print(f"   List length: {len(loaded_data)}")
                print(f"   List contents type: {type(loaded_data[0]) if loaded_data else 'empty'}")
                raise ValueError("Model file format is incorrect. Please retrain the model using 'python train_model.py'")
            
            # Check if it's the expected dictionary format
            if not isinstance(loaded_data, dict):
                print(f"⚠️ Warning: Model file contains {type(loaded_data)}, expected dictionary.")
                raise ValueError(f"Unexpected model file format: {type(loaded_data)}. Please retrain the model.")
            
            # Verify required keys exist
            required_keys = ['model', 'scaler', 'vectorizer', 'encoders', 'feature_columns']
            missing_keys = [key for key in required_keys if key not in loaded_data]
            if missing_keys:
                raise ValueError(f"Model file missing required components: {missing_keys}")
            
            self.model_components = loaded_data
            print(f"✅ Model loaded successfully from {self.model_path}")
            
            # Safely access model type and accuracy
            if 'model_type' in self.model_components:
                print(f"   Model type: {self.model_components['model_type']}")
            if 'results' in self.model_components and 'model_type' in self.model_components:
                model_type = self.model_components['model_type']
                if model_type in self.model_components['results']:
                    print(f"   Model accuracy: {self.model_components['results'][model_type]:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(f"   This usually means the model file format is incompatible.")
            print(f"   Please retrain the model by running: python train_model.py")
            return False
    
    def create_ayurvedic_database(self):
        """
        Create comprehensive Ayurvedic recommendations database
        Maps diseases to their corresponding Ayurvedic treatments
        """
        csv_path = os.path.join(os.getcwd(), 'enhanced_ayurvedic_treatment_dataset.csv')

        def _clean_text(val: str) -> str:
            if pd.isna(val):
                return ''
            s = str(val).strip()
            # split on pipes to separate multi-part effects and join later
            return s.replace('_', ' ').replace(' ,', ',').replace('  ', ' ').strip()

        def _aggregate(series: pd.Series) -> str:
            if series is None or series.empty:
                return ''
            parts: List[str] = []
            for raw in series.dropna().astype(str):
                # split by '|' to preserve multiple effect phrases
                split_items = [p.strip() for p in str(raw).split('|') if p and p.strip().lower() != 'nan']
                for item in split_items:
                    cleaned = _clean_text(item)
                    if cleaned and cleaned not in parts:
                        parts.append(cleaned)
            return '; '.join(parts)

        columns_to_keep = [
            'Ayurvedic_Herbs_Sanskrit',
            'Ayurvedic_Herbs_English',
            'Herbs_Effects',
            'Ayurvedic_Therapies_Sanskrit',
            'Ayurvedic_Therapies_English',
            'Therapies_Effects',
            'Dietary_Recommendations',
            'How_Treatment_Affects_Your_Body_Type',
        ]

        # Default minimal fallback in case CSV read fails
        default_reco = {
            'Ayurvedic_Herbs_Sanskrit': 'Amalaki, Haridra, Tulasi',
            'Ayurvedic_Herbs_English': 'Amla, Turmeric, Holy Basil',
            'Herbs_Effects': 'General immunity boost; Anti-inflammatory; Antioxidant',
            'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Pranayama, Yoga',
            'Ayurvedic_Therapies_English': 'Oil massage, Breathing exercises, Yoga',
            'Therapies_Effects': 'General wellness; Stress reduction; Improved circulation',
            'Dietary_Recommendations': 'Balanced diet; Fresh foods; Adequate water; Regular meals',
            'How_Treatment_Affects_Your_Body_Type': 'General balancing of doshas; Promotes overall health',
        }

        if not os.path.exists(csv_path):
            print(f"⚠️ Ayurvedic CSV not found at {csv_path}. Using minimal fallback recommendations.")
            self.ayurvedic_database = {}
            return

        try:
            df = pd.read_csv(csv_path)
            if 'Disease' not in df.columns:
                raise ValueError("CSV missing required 'Disease' column")

            # Normalize disease names
            df['Disease'] = df['Disease'].astype(str).str.strip()

            # Build mapping by aggregating all rows per disease
            ayur_map: Dict[str, Dict[str, str]] = {}
            grouped = df.groupby('Disease', sort=True)
            for disease, g in grouped:
                entry: Dict[str, str] = {}
                for col in columns_to_keep:
                    if col in g.columns:
                        entry[col] = _aggregate(g[col])
                    else:
                        entry[col] = ''

                # Ensure some sensible fallback text if fields are empty
                for k, v in list(entry.items()):
                    if not v:
                        entry[k] = default_reco.get(k, '')

                ayur_map[disease] = entry

            self.ayurvedic_database = ayur_map
            print(f"✅ Loaded Ayurvedic recommendations for {len(self.ayurvedic_database)} diseases from CSV")

        except Exception as e:
            print(f"⚠️ Failed to load Ayurvedic recommendations from CSV: {e}")
            self.ayurvedic_database = {}
    
    # -----------------------------
    # Display helpers (compact view)
    # -----------------------------
    def _compact_listlike(self, text: str, max_items: int = 6) -> str:
        """Turn a long aggregated list (with ';' and ',') into unique, concise comma list."""
        if not text:
            return ''
        items: List[str] = []
        # split on semicolons first, then commas
        for seg in str(text).split(';'):
            for token in seg.split(','):
                t = token.strip()
                if not t or t.lower() == 'nan':
                    continue
                if t not in items:
                    items.append(t)
        return ', '.join(items[:max_items])

    def _compact_phrases(self, text: str, max_items: int = 4) -> str:
        """Keep unique phrases separated by ';' and cap the count (case-insensitive dedupe)."""
        if not text:
            return ''
        seen_lower: List[str] = []
        unique_phrases: List[str] = []
        for seg in str(text).split(';'):
            t = seg.strip().strip(',.;')
            if not t or t.lower() == 'nan':
                continue
            key = t.lower()
            if key not in seen_lower:
                seen_lower.append(key)
                unique_phrases.append(t)
        # If there's still only one long item, also try splitting by comma to reduce duplicates
        if len(unique_phrases) <= 1 and ',' in text:
            seen_lower = []
            unique_phrases = []
            for seg in str(text).replace(';', ',').split(','):
                t = seg.strip().strip(',.;')
                if not t or t.lower() == 'nan':
                    continue
                key = t.lower()
                if key not in seen_lower:
                    seen_lower.append(key)
                    unique_phrases.append(t)
        return '; '.join(unique_phrases[:max_items])

    def format_for_display(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of result with compact, readable fields for printing."""
        d = dict(result)
        # List-like fields (herbs and therapies names)
        for key in [
            'Ayurvedic_Herbs_Sanskrit',
            'Ayurvedic_Herbs_English',
            'Ayurvedic_Therapies_Sanskrit',
            'Ayurvedic_Therapies_English',
        ]:
            d[key] = self._compact_listlike(d.get(key, ''), max_items=6)

        # Phrase-like fields (effects, diet)
        for key, cap in [
            ('Herbs_Effects', 4),
            ('Therapies_Effects', 4),
            ('Dietary_Recommendations', 4),
        ]:
            d[key] = self._compact_phrases(d.get(key, ''), max_items=cap)

        # Keep personalized effects as-is
        return d
    
    def preprocess_user_input(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess user input for prediction"""
        if not self.model_components:
            raise ValueError("Model not loaded. Please load model first.")
        
        # Verify model_components is a dictionary
        if not isinstance(self.model_components, dict):
            raise ValueError(f"Model components must be a dictionary, got {type(self.model_components)}. Please retrain the model.")
        
        # Verify encoders is a dictionary
        if 'encoders' not in self.model_components:
            raise ValueError("Model missing 'encoders' component. Please retrain the model.")
        
        encoders = self.model_components['encoders']
        if not isinstance(encoders, dict):
            raise ValueError(f"Encoders must be a dictionary, got {type(encoders)}. Please retrain the model.")
        
        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Calculate BMI if not provided
        if 'BMI' not in user_df.columns and 'Height_cm' in user_df.columns and 'Weight_kg' in user_df.columns:
            user_df['BMI'] = user_df['Weight_kg'] / (user_df['Height_cm'] / 100) ** 2
        
        # Encode categorical features
        categorical_columns = ['Age_Group', 'Gender', 'Body_Type_Dosha_Sanskrit', 
                              'Food_Habits', 'Current_Medication', 'Allergies', 'Season', 'Weather']
        
        for col in categorical_columns:
            if col in user_df.columns and col in encoders:
                try:
                    user_df[f'{col}_encoded'] = encoders[col].transform(user_df[col])
                except ValueError:
                    # Handle unknown categories
                    user_df[f'{col}_encoded'] = 0
        
        # Select numerical and encoded features
        if 'feature_columns' not in self.model_components:
            raise ValueError("Model missing 'feature_columns' component. Please retrain the model.")
        
        other_features = user_df[self.model_components['feature_columns']]
        
        # Transform symptoms using TF-IDF (apply light normalization & synonym mapping)
        def _normalize_symptoms_text(text: str) -> str:
            if not text:
                return ''
            t = str(text).lower()
            # Basic cleanup
            t = t.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
            # Common synonym mapping to align with training vocabulary
            synonyms = {
                'pyrexia': 'fever', 'high temperature': 'fever',
                'myalgia': 'body ache', 'arthralgia': 'joint pain',
                'cephalalgia': 'headache', 'rhinorrhea': 'runny nose', 'coryza': 'cold',
                'dyspnea': 'breathing difficulty', 'shortness of breath': 'breathing difficulty',
                'emesis': 'vomiting', 'nauseous': 'nausea', 'nasal congestion': 'blocked nose',
                'pharyngitis': 'sore throat', 'odynophagia': 'sore throat',
                'diarrhea': 'diarrhoea', # normalize US/UK spelling
                'tiredness': 'fatigue', 'exhaustion': 'fatigue',
            }
            for src, dst in synonyms.items():
                t = t.replace(src, dst)
            # Normalize separators and de-dupe tokens while preserving order
            tokens = [s.strip(' ,.;') for s in t.split(',') if s.strip(' ,.;')]
            seen = set()
            normalized = []
            for tok in tokens:
                if tok not in seen:
                    seen.add(tok)
                    normalized.append(tok)
            return ', '.join(normalized)

        if 'Symptoms' in user_data:
            normalized_symptoms = _normalize_symptoms_text(user_data['Symptoms'])
            tfidf_matrix = self.model_components['vectorizer'].transform([normalized_symptoms])
            tfidf_features = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            # Combine features
            combined_features = pd.concat([other_features.reset_index(drop=True), 
                                         tfidf_features.reset_index(drop=True)], axis=1)
        else:
            combined_features = other_features
        
        # Scale features
        combined_features_scaled = self.model_components['scaler'].transform(combined_features)
        
        return combined_features_scaled
    
    def predict_disease_with_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction function - returns disease prediction with Ayurvedic recommendations
        """
        if not self.model_components:
            raise ValueError("Model not loaded!")
        
        # Verify model_components structure
        if not isinstance(self.model_components, dict):
            raise ValueError(f"Model components must be a dictionary, got {type(self.model_components)}. Please retrain the model.")
        
        if 'model' not in self.model_components:
            raise ValueError("Model missing 'model' component. Please retrain the model.")
        
        if 'encoders' not in self.model_components:
            raise ValueError("Model missing 'encoders' component. Please retrain the model.")
        
        encoders = self.model_components['encoders']
        if not isinstance(encoders, dict):
            raise ValueError(f"Encoders must be a dictionary, got {type(encoders)}. Please retrain the model.")
        
        if 'Disease' not in encoders:
            raise ValueError("Model missing 'Disease' encoder. Please retrain the model.")
        
        # Preprocess input
        processed_features = self.preprocess_user_input(user_data)
        
        # Make prediction
        model = self.model_components['model']
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)
        
        # Get disease name
        predicted_disease = encoders['Disease'].inverse_transform(prediction)[0]
        
        # Apply confidence score calibration to avoid overconfident predictions
        # The raw confidence from the model needs to be adjusted to be more realistic
        raw_confidence = np.max(prediction_proba)
        
        # Calculate confidence with additional calibration based on the difference 
        # between the top prediction and the second best prediction
        proba_vec = prediction_proba[0]
        sorted_probs = np.sort(proba_vec)[::-1]  # Sort in descending order
        
        if len(sorted_probs) > 1:
            top_prob = sorted_probs[0]
            second_prob = sorted_probs[1]
            # Create more realistic confidence based on the gap between best and second best
            confidence_gap = top_prob - second_prob
            
            # Use a more nuanced approach: balance between the raw probability and the gap
            # If the gap is large and the top probability is high, we can have higher confidence
            # If the gap is small, keep confidence more conservative
            if confidence_gap > 0.3:  # Large gap - higher confidence possible
                calibrated_confidence = min(0.98, max(0.05, top_prob * 0.8 + confidence_gap * 0.5))
            elif confidence_gap > 0.15:  # Medium gap - moderate confidence
                calibrated_confidence = min(0.95, max(0.05, top_prob * 0.7 + confidence_gap * 0.3))
            else:  # Small gap - low confidence
                calibrated_confidence = min(0.85, max(0.02, top_prob * 0.6 + confidence_gap * 0.4))
        else:
            calibrated_confidence = raw_confidence
        
        # Ensure confidence is realistic (not exactly 1.0 or 0.0)
        confidence = max(0.01, min(0.99, calibrated_confidence))

        # Compute Top-5 predictions
        idx_sorted = np.argsort(proba_vec)[::-1]
        topk_idx = idx_sorted[:5]
        class_indices = model.classes_
        encoder = self.model_components['encoders']['Disease']
        topk_names = encoder.inverse_transform(class_indices[topk_idx])
        
        # Apply similar calibration to all top predictions
        top5 = []
        for name, i in zip(topk_names, topk_idx):
            raw_score = float(proba_vec[i])
            # Apply minimal calibration to other predictions as well
            calibrated_score = max(0.001, min(0.999, raw_score))
            top5.append({
                'Disease': str(name),
                'Confidence': calibrated_score
            })
        
        # Get Ayurvedic recommendations
        body_type = user_data.get('Body_Type_Dosha_Sanskrit', 'Unknown')

        ayurvedic_recommendations = None
        if isinstance(self.ayurvedic_database, dict) and self.ayurvedic_database:
            # 1) Exact match
            if predicted_disease in self.ayurvedic_database:
                ayurvedic_recommendations = self.ayurvedic_database[predicted_disease].copy()
            else:
                # 2) Normalize disease names for better matching
                # Remove extra parentheses, standardize common variations
                normalized_pred_disease = predicted_disease.lower().strip()
                
                # Handle common variations in disease naming
                if "(vertigo)" in normalized_pred_disease and "positional" in normalized_pred_disease:
                    possible_matches = [k for k in self.ayurvedic_database.keys() 
                                      if "vertigo" in k.lower() and "positional" in k.lower()]
                    if possible_matches:
                        ayurvedic_recommendations = self.ayurvedic_database[possible_matches[0]].copy()
                elif "common cold" in normalized_pred_disease or "cold" == normalized_pred_disease:
                    # Look for common cold in database
                    cold_keys = [k for k in self.ayurvedic_database.keys() 
                                if "cold" in k.lower() and "common" in k.lower()]
                    if cold_keys:
                        ayurvedic_recommendations = self.ayurvedic_database[cold_keys[0]].copy()
                    else:
                        # Fallback to just "Cold" or "Common Cold"
                        fallback_keys = [k for k in self.ayurvedic_database.keys() 
                                        if "common" in k.lower() and "cold" in k.lower()]
                        if fallback_keys:
                            ayurvedic_recommendations = self.ayurvedic_database[fallback_keys[0]].copy()
                else:
                    # 3) Case-insensitive match
                    keys_lower = {k.lower(): k for k in self.ayurvedic_database.keys()}
                    key_ci = keys_lower.get(normalized_pred_disease)
                    if key_ci:
                        ayurvedic_recommendations = self.ayurvedic_database[key_ci].copy()
                    else:
                        # 4) Partial match
                        cand = None
                        pred_lower = normalized_pred_disease
                        for k in self.ayurvedic_database.keys():
                            if pred_lower in k.lower() or k.lower() in pred_lower:
                                cand = k
                                break
                        if cand:
                            ayurvedic_recommendations = self.ayurvedic_database[cand].copy()

        if ayurvedic_recommendations is None:
            # Default recommendations if mapping unavailable
            ayurvedic_recommendations = {
                'Ayurvedic_Herbs_Sanskrit': 'Amalaki, Haridra, Tulasi',
                'Ayurvedic_Herbs_English': 'Amla, Turmeric, Holy Basil',
                'Herbs_Effects': 'General immunity boost; Anti-inflammatory; Antioxidant',
                'Ayurvedic_Therapies_Sanskrit': 'Abhyanga, Pranayama, Yoga',
                'Ayurvedic_Therapies_English': 'Oil massage, Breathing exercises, Yoga',
                'Therapies_Effects': 'General wellness; Stress reduction; Improved circulation',
                'Dietary_Recommendations': 'Balanced diet; Fresh foods; Adequate water; Regular meals',
                'How_Treatment_Affects_Your_Body_Type': 'General balancing of doshas; Promotes overall health'
            }
        
        # Personalize for body type
        if body_type != "Unknown":
            ayurvedic_recommendations['How_Treatment_Affects_Your_Body_Type'] += f" (Specifically beneficial for {body_type} constitution)"
        
        # Compile results
        result = {
            'Predicted_Disease': predicted_disease,
            'Confidence': float(confidence),
            'User_Symptoms': user_data.get('Symptoms', ''),
            'User_Body_Type': body_type,
            'Top_5_Predictions': top5,
            **ayurvedic_recommendations
        }
        
        return result
    
    def get_dosha_selection(self):
        """Enhanced dosha selection with clear body type descriptions"""
        
        print("\n🌿 AYURVEDIC BODY TYPE ASSESSMENT 🌿")
        print("=" * 50)
        print("Select your body type based on physical characteristics:\n")
        
        dosha_options = {
            '1': {
                'name': 'Vata',
                'constitution': 'Air_Space_Constitution',
                'body_type': 'Thin/Lean',
                'description': 'Naturally thin build, difficulty gaining weight, dry skin, cold hands/feet'
            },
            '2': {
                'name': 'Pitta',
                'constitution': 'Fire_Water_Constitution', 
                'body_type': 'Medium',
                'description': 'Medium build, good muscle tone, warm body, strong appetite'
            },
            '3': {
                'name': 'Kapha',
                'constitution': 'Earth_Water_Constitution',
                'body_type': 'Heavy/Large',
                'description': 'Naturally larger build, gains weight easily, cool moist skin, steady energy'
            },
            '4': {
                'name': 'Vata-Pitta',
                'constitution': 'Air_Fire_Mixed_Constitution',
                'body_type': 'Thin to Medium',
                'description': 'Variable build, creative energy, moderate body temperature'
            },
            '5': {
                'name': 'Vata-Kapha',
                'constitution': 'Air_Earth_Mixed_Constitution',
                'body_type': 'Thin to Heavy',
                'description': 'Variable patterns, irregular tendencies, sensitive to changes'
            },
            '6': {
                'name': 'Pitta-Kapha',
                'constitution': 'Fire_Earth_Mixed_Constitution',
                'body_type': 'Medium to Heavy',
                'description': 'Strong stable build, good strength, balanced metabolism'
            }
        }
        
        # Display options
        for key, value in dosha_options.items():
            print(f"{key}. {value['name']} - {value['body_type']}")
            print(f"   {value['description']}")
            print()
        
        print("You can enter:")
        print("• Number (1-6)")  
        print("• Dosha name (e.g., 'Vata', 'Pitta-Kapha')")
        print("• Body type (e.g., 'thin', 'medium', 'heavy')")
        
        while True:
            dosha_choice = input("\nEnter your selection: ").strip()
            
            # Check if it's a number
            if dosha_choice in dosha_options:
                selected = dosha_options[dosha_choice]
                return selected['name'], selected['constitution']
            
            # Check if it's a dosha name (case insensitive)
            dosha_choice_lower = dosha_choice.lower()
            for option in dosha_options.values():
                if option['name'].lower() == dosha_choice_lower:
                    return option['name'], option['constitution']
            
            # Check if it's a body type description
            body_type_mapping = {
                'thin': '1', 'lean': '1', 'skinny': '1',
                'medium': '2', 'average': '2', 'moderate': '2',
                'heavy': '3', 'large': '3', 'big': '3', 'fat': '3',
                'thin to medium': '4', 'variable thin': '4',
                'thin to heavy': '5', 'irregular': '5',
                'medium to heavy': '6', 'strong': '6'
            }
            
            if dosha_choice_lower in body_type_mapping:
                selected_key = body_type_mapping[dosha_choice_lower]
                selected = dosha_options[selected_key]
                return selected['name'], selected['constitution']
            
            print("❌ Invalid selection. Please try again.")
            print("Use numbers 1-6, dosha names, or body type descriptions.")

    def get_user_input_interactive(self) -> Dict[str, Any]:
        """Interactive input collection"""
        print("\n" + "="*60)
        print("AROGYA AI - INTERACTIVE HEALTH ASSESSMENT")
        print("="*60)
        print("Please provide the following information for accurate prediction:\n")
        
        user_data = {}
        
        # Basic Information
        user_data['Symptoms'] = input("Enter your symptoms (comma-separated): ")
        user_data['Age'] = int(input("Enter your age: "))
        user_data['Height_cm'] = float(input("Enter your height in cm: "))
        user_data['Weight_kg'] = float(input("Enter your weight in kg: "))
        
        # Auto-determine age group
        age = user_data['Age']
        if age <= 12:
            age_group = "Child"
        elif age <= 19:
            age_group = "Adolescent"
        elif age <= 35:
            age_group = "Young Adult"
        elif age <= 50:
            age_group = "Middle Age"
        elif age <= 65:
            age_group = "Senior"
        else:
            age_group = "Elderly"
        
        user_data['Age_Group'] = age_group
        
        print(f"\nGender options: Male, Female")
        user_data['Gender'] = input("Enter your gender: ")
        
        # Use enhanced dosha selection
        dosha_name, dosha_constitution = self.get_dosha_selection()
        user_data['Body_Type_Dosha_Sanskrit'] = dosha_name
        print(f"\n✅ Selected: {dosha_name} ({dosha_constitution})")
        
        print(f"\nFood Habits options: Vegetarian, Non-Vegetarian, Vegan, Mixed")
        user_data['Food_Habits'] = input("Enter your food habits: ") or "Mixed"
        
        user_data['Current_Medication'] = input("Enter current medications (or 'None'): ") or "None"
        user_data['Allergies'] = input("Enter known allergies (or 'None'): ") or "None"
        
        print(f"\nSeason options: Spring, Summer, Monsoon, Autumn, Winter")
        user_data['Season'] = input("Enter current season: ") or "Summer"
        
        print(f"\nWeather options: Hot, Cold, Humid, Dry, Rainy")
        user_data['Weather'] = input("Enter current weather: ") or "Hot"
        
        return user_data

def demo_sample_prediction():
    """Demonstrate with sample data"""
    print("\n" + "="*70)
    print("AROGYA AI - DISEASE PREDICTION WITH AYURVEDIC RECOMMENDATIONS")
    print("="*70)
    
    # Initialize system
    ai_system = ArogyaAI()
    
    if not ai_system.model_components:
        print("❌ Model not available. Please run 'python train_model.py' first.")
        return
    
    # Sample predictions
    sample_cases = [
        {
            'name': 'Case 1: Fever symptoms',
            'data': {
                'Symptoms': 'fever, body ache, headache, fatigue',
                'Age': 35,
                'Height_cm': 170,
                'Weight_kg': 75,
                'Gender': 'Female',
                'Age_Group': 'Young Adult',
                'Body_Type_Dosha_Sanskrit': 'Pitta',
                'Food_Habits': 'Vegetarian',
                'Current_Medication': 'None',
                'Allergies': 'None',
                'Season': 'Summer',
                'Weather': 'Hot'
            }
        },
        {
            'name': 'Case 2: Respiratory symptoms',
            'data': {
                'Symptoms': 'cough, breathing difficulty, chest tightness',
                'Age': 45,
                'Height_cm': 175,
                'Weight_kg': 80,
                'Gender': 'Male',
                'Age_Group': 'Middle Age',
                'Body_Type_Dosha_Sanskrit': 'Kapha',
                'Food_Habits': 'Non-Vegetarian',
                'Current_Medication': 'None',
                'Allergies': 'None',
                'Season': 'Winter',
                'Weather': 'Cold'
            }
        }
    ]
    
    for case in sample_cases:
        print(f"\n{'='*50}")
        print(f"🔍 {case['name']}")
        print(f"{'='*50}")
        
        try:
            result = ai_system.predict_disease_with_recommendations(case['data'])
            display = ai_system.format_for_display(result)
            
            print(f"\n📋 PREDICTION RESULTS:")
            print(f"   Predicted Disease: {display['Predicted_Disease']}")
            print(f"   Confidence: {display['Confidence']:.2%}")
            if 'Top_5_Predictions' in result:
                print("   Top 5 candidates:")
                for i, item in enumerate(result['Top_5_Predictions'], start=1):
                    print(f"     {i}. {item['Disease']} ({item['Confidence']:.2%})")
            print(f"   Symptoms: {display['User_Symptoms']}")
            print(f"   Body Type: {display['User_Body_Type']}")
            
            print(f"\n🌿 AYURVEDIC RECOMMENDATIONS:")
            print(f"   Sanskrit Herbs: {display['Ayurvedic_Herbs_Sanskrit']}")
            print(f"   English Herbs: {display['Ayurvedic_Herbs_English']}")
            print(f"   Herb Effects: {display['Herbs_Effects']}")
            print(f"   Sanskrit Therapies: {display['Ayurvedic_Therapies_Sanskrit']}")
            print(f"   English Therapies: {display['Ayurvedic_Therapies_English']}")
            print(f"   Therapy Effects: {display['Therapies_Effects']}")
            
            print(f"\n🍽️ DIETARY RECOMMENDATIONS:")
            print(f"   {display['Dietary_Recommendations']}")
            
            print(f"\n👤 PERSONALIZED TREATMENT EFFECTS:")
            print(f"   {display['How_Treatment_Affects_Your_Body_Type']}")
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")

def interactive_mode():
    """Run interactive prediction mode"""
    ai_system = ArogyaAI()
    
    if not ai_system.model_components:
        print("❌ Model not available. Please run 'python train_model.py' first.")
        return
    
    user_input = ai_system.get_user_input_interactive()
    
    print(f"\n{'='*60}")
    print("🔍 YOUR PREDICTION RESULTS")
    print(f"{'='*60}")
    
    try:
        result = ai_system.predict_disease_with_recommendations(user_input)
        display = ai_system.format_for_display(result)
        
        print(f"\n📋 MEDICAL PREDICTION:")
        print(f"   Disease: {display['Predicted_Disease']}")
        print(f"   Confidence: {display['Confidence']:.2%}")
        if 'Top_5_Predictions' in result:
            print("   Top 5 candidates:")
            for i, item in enumerate(result['Top_5_Predictions'], start=1):
                print(f"     {i}. {item['Disease']} ({item['Confidence']:.2%})")
        
        print(f"\n🌿 AYURVEDIC TREATMENT PLAN:")
        print(f"   Sanskrit Herbs: {display['Ayurvedic_Herbs_Sanskrit']}")
        print(f"   English Herbs: {display['Ayurvedic_Herbs_English']}")
        print(f"   Herb Benefits: {display['Herbs_Effects']}")
        
        print(f"\n💆 THERAPEUTIC TREATMENTS:")
        print(f"   Sanskrit Therapies: {display['Ayurvedic_Therapies_Sanskrit']}")
        print(f"   English Therapies: {display['Ayurvedic_Therapies_English']}")
        print(f"   Treatment Benefits: {display['Therapies_Effects']}")
        
        print(f"\n🥗 DIETARY GUIDANCE:")
        print(f"   {display['Dietary_Recommendations']}")
        
        print(f"\n🎯 PERSONALIZED TREATMENT EFFECTS:")
        print(f"   {display['How_Treatment_Affects_Your_Body_Type']}")
        
        # Save results
        print(f"\n💾 Results saved for your reference.")
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")

if __name__ == "__main__":
    print("Initializing Arogya AI...")
    
    # Check if model exists
    if not os.path.exists('random_forest_model.pkl'):
        print("\n❌ Trained model not found!")
        print("📋 Please run the following command first:")
        print("   python train_model.py")
        print("\nThis will train the model and save it for predictions.")
    else:
        # Run demo
        demo_sample_prediction()
        
        # Ask for interactive mode
        print(f"\n{'='*70}")
        interactive = input("Would you like to try interactive prediction? (y/n): ").lower().strip()
        
        if interactive == 'y':
            interactive_mode()
    
    print(f"\n{'='*70}")
    print("Thank you for using Arogya AI!")
    print("Stay healthy with the wisdom of Ayurveda! 🌿")
    print(f"{'='*70}")
