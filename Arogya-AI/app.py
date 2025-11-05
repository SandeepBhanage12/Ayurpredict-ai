#!/usr/bin/env python3
"""
Arogya AI - Interactive Frontend (Streamlit)
Run: streamlit run app.py
"""

import os
import sys
import textwrap
from typing import Dict, Any, List

import streamlit as st

# Ensure local imports work when running from this folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from arogya_predict import ArogyaAI


# -----------------------------
# UI Helpers
# -----------------------------
def _pill_list(items_text: str) -> None:
    if not items_text:
        return
    items = [x.strip() for x in items_text.split(',') if x.strip()]
    if not items:
        return
    cols = st.columns(min(4, max(1, len(items))))
    for idx, item in enumerate(items):
        with cols[idx % len(cols)]:
            st.markdown(
                f"<div style='display:inline-block;padding:6px 10px;margin:6px 0;"
                f"border-radius:16px;background:#EEF6EE;border:1px solid #D7EBD7;"
                f"color:#1E6F3F;font-weight:500;font-size:0.9rem'>{item}</div>",
                unsafe_allow_html=True,
            )


def _section_title(emoji: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div style='margin-top:14px;margin-bottom:6px'>"
        f"<div style='font-size:1.3rem;font-weight:700'>{emoji} {title}</div>"
        f"<div style='color:#5f6b7a;margin-top:2px'>{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _dosha_cards(default: str = "Pitta") -> str:
    st.write("Select your Ayurvedic constitution (Dosha):")
    doshas = [
        ("Vata", "Air_Space_Constitution", "Thin/Lean", "Naturally thin build, dry skin, cold hands/feet"),
        ("Pitta", "Fire_Water_Constitution", "Medium", "Warm body, strong appetite, good muscle tone"),
        ("Kapha", "Earth_Water_Constitution", "Heavy/Large", "Larger build, steady energy, cool moist skin"),
        ("Vata-Pitta", "Air_Fire_Mixed_Constitution", "Thin to Medium", "Variable build, creative energy"),
        ("Vata-Kapha", "Air_Earth_Mixed_Constitution", "Thin to Heavy", "Irregular patterns, sensitive to changes"),
        ("Pitta-Kapha", "Fire_Earth_Mixed_Constitution", "Medium to Heavy", "Strong stable build, balanced metabolism"),
    ]

    selected = st.session_state.get("dosha", default)
    cols = st.columns(3)

    for i, (name, _const, body, desc) in enumerate(doshas):
        with cols[i % 3]:
            is_active = selected == name
            bg = "#FFF7E8" if is_active else "#FFFFFF"
            bd = "#FECF66" if is_active else "#E6E8EB"
            st.markdown(
                f"<div style='border:2px solid {bd};border-radius:12px;padding:12px;margin-bottom:10px;"
                f"background:{bg};cursor:default'>"
                f"<div style='font-weight:700'>{name}</div>"
                f"<div style='color:#5f6b7a;font-size:0.9rem'>{body}</div>"
                f"<div style='color:#6b7280;font-size:0.85rem;margin-top:4px'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select", key=f"dosha_{name}"):
                selected = name

    st.session_state["dosha"] = selected
    return selected


def _format_percent(p: float) -> str:
    try:
        return f"{float(p)*100:.2f}%"
    except Exception:
        return "-"


def _example_cases() -> List[Dict[str, Any]]:
    return [
        {
            "label": "Fever-like symptoms (Young Adult, Pitta)",
            "data": {
                "Symptoms": "fever, body ache, headache, fatigue",
                "Age": 35,
                "Height_cm": 170.0,
                "Weight_kg": 75.0,
                "Gender": "Female",
                "Age_Group": "Young Adult",
                "Body_Type_Dosha_Sanskrit": "Pitta",
                "Food_Habits": "Vegetarian",
                "Current_Medication": "None",
                "Allergies": "None",
                "Season": "Summer",
                "Weather": "Hot",
            },
        },
        {
            "label": "Respiratory symptoms (Middle Age, Kapha)",
            "data": {
                "Symptoms": "cough, breathing difficulty, chest tightness",
                "Age": 45,
                "Height_cm": 175.0,
                "Weight_kg": 80.0,
                "Gender": "Male",
                "Age_Group": "Middle Age",
                "Body_Type_Dosha_Sanskrit": "Kapha",
                "Food_Habits": "Non-Vegetarian",
                "Current_Medication": "None",
                "Allergies": "None",
                "Season": "Winter",
                "Weather": "Cold",
            },
        },
    ]


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Arogya AI", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .stSelectbox > div > div { border-radius: 10px; }
    .stTextInput > div > div { border-radius: 10px; }
    .stNumberInput > div > div { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.55, 0.45])

with left:
    st.title("Arogya AI")
    st.caption("Disease prediction with personalized Ayurvedic recommendations")

    with st.expander("Try demo presets", expanded=False):
        demos = _example_cases()
        labels = [d["label"] for d in demos]
        choice = st.selectbox("Select a preset", ["—"] + labels, index=0)
        if choice != "—":
            data = next(d["data"] for d in demos if d["label"] == choice)
            for k, v in data.items():
                st.session_state[k] = v

    # Inputs
    _section_title("📝", "Symptoms", "Use comma-separated phrases")
    symptoms = st.text_area(
        "",
        value=st.session_state.get("Symptoms", ""),
        height=90,
        placeholder="e.g., fever, body ache, headache, fatigue",
    )

    _section_title("👤", "Profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.get("Age", 30)))
    with c2:
        height_cm = st.number_input("Height (cm)", min_value=60.0, max_value=250.0, value=float(st.session_state.get("Height_cm", 170.0)))
    with c3:
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=float(st.session_state.get("Weight_kg", 70.0)))
    with c4:
        gender = st.selectbox("Gender", ["Male", "Female"], index=(0 if st.session_state.get("Gender", "Male") == "Male" else 1))

    # Age group auto-derive with override
    if age <= 12:
        auto_age_group = "Child"
    elif age <= 19:
        auto_age_group = "Adolescent"
    elif age <= 35:
        auto_age_group = "Young Adult"
    elif age <= 50:
        auto_age_group = "Middle Age"
    elif age <= 65:
        auto_age_group = "Senior"
    else:
        auto_age_group = "Elderly"
    age_group = st.selectbox(
        "Age Group",
        ["Child", "Adolescent", "Young Adult", "Middle Age", "Senior", "Elderly"],
        index=["Child", "Adolescent", "Young Adult", "Middle Age", "Senior", "Elderly"].index(
            st.session_state.get("Age_Group", auto_age_group)
        ),
        help="Auto-selected based on age; you can override",
    )

    _section_title("🌿", "Dosha (Ayurvedic Constitution)")
    body_type_dosha = _dosha_cards(default=st.session_state.get("Body_Type_Dosha_Sanskrit", "Pitta"))

    _section_title("🍽️", "Lifestyle & Context")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        food_habits = st.selectbox(
            "Food Habits",
            ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"],
            index=["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"].index(
                st.session_state.get("Food_Habits", "Mixed")
            ),
        )
    with c6:
        current_med = st.text_input("Current Medication", value=st.session_state.get("Current_Medication", "None"))
    with c7:
        allergies = st.text_input("Allergies", value=st.session_state.get("Allergies", "None"))
    with c8:
        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Monsoon", "Autumn", "Winter"],
            index=["Spring", "Summer", "Monsoon", "Autumn", "Winter"].index(
                st.session_state.get("Season", "Summer")
            ),
        )
    weather = st.selectbox(
        "Weather",
        ["Hot", "Cold", "Humid", "Dry", "Rainy"],
        index=["Hot", "Cold", "Humid", "Dry", "Rainy"].index(st.session_state.get("Weather", "Hot")),
    )

    # Persist state
    for k, v in {
        "Symptoms": symptoms,
        "Age": age,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Gender": gender,
        "Age_Group": age_group,
        "Body_Type_Dosha_Sanskrit": body_type_dosha,
        "Food_Habits": food_habits,
        "Current_Medication": current_med,
        "Allergies": allergies,
        "Season": season,
        "Weather": weather,
    }.items():
        st.session_state[k] = v

    st.divider()
    predict_now = st.button("🔮 Predict Now", type="primary")

with right:
    _section_title("📊", "Prediction Results", "Top candidates and confidence")
    result_placeholder = st.empty()

    # Initialize model once
    if "_ai" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state["_ai"] = ArogyaAI()

    ai: ArogyaAI = st.session_state["_ai"]

    def _predict_and_render():
        if not ai.model_components:
            with result_placeholder.container():
                st.warning("Model not available. Please run 'python train_model.py' inside `Arogya-AI`.")
            return

        user: Dict[str, Any] = {
            "Symptoms": st.session_state.get("Symptoms", ""),
            "Age": st.session_state.get("Age", 30),
            "Height_cm": st.session_state.get("Height_cm", 170.0),
            "Weight_kg": st.session_state.get("Weight_kg", 70.0),
            "Gender": st.session_state.get("Gender", "Male"),
            "Age_Group": st.session_state.get("Age_Group", "Young Adult"),
            "Body_Type_Dosha_Sanskrit": st.session_state.get("Body_Type_Dosha_Sanskrit", "Pitta"),
            "Food_Habits": st.session_state.get("Food_Habits", "Mixed"),
            "Current_Medication": st.session_state.get("Current_Medication", "None"),
            "Allergies": st.session_state.get("Allergies", "None"),
            "Season": st.session_state.get("Season", "Summer"),
            "Weather": st.session_state.get("Weather", "Hot"),
        }

        try:
            result = ai.predict_disease_with_recommendations(user)
        except Exception as e:
            with result_placeholder.container():
                st.error(f"Prediction failed: {e}")
            return

        display = ai.format_for_display(result)

        with result_placeholder.container():
            st.markdown(
                f"<div style='padding:12px;border:1px solid #E6E8EB;border-radius:12px;background:#FFFFFF'>"
                f"<div style='font-size:1.1rem;font-weight:700'>Predicted Disease: {display['Predicted_Disease']}</div>"
                f"<div style='color:#5f6b7a;margin-top:2px'>Confidence: {_format_percent(display['Confidence'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if "Top_5_Predictions" in result and result["Top_5_Predictions"]:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                for i, item in enumerate(result["Top_5_Predictions"], start=1):
                    st.progress(min(0.999, float(item["Confidence"])) , text=f"{i}. {item['Disease']} ({_format_percent(item['Confidence'])})")

            _section_title("🌿", "Ayurvedic Herbs", "Sanskrit and English names")
            _pill_list(display.get("Ayurvedic_Herbs_Sanskrit", ""))
            _pill_list(display.get("Ayurvedic_Herbs_English", ""))

            _section_title("✨", "Herb Effects")
            st.write(display.get("Herbs_Effects", ""))

            _section_title("💆", "Therapies")
            _pill_list(display.get("Ayurvedic_Therapies_Sanskrit", ""))
            _pill_list(display.get("Ayurvedic_Therapies_English", ""))

            _section_title("🧪", "Therapy Effects")
            st.write(display.get("Therapies_Effects", ""))

            _section_title("🥗", "Dietary Recommendations")
            st.write(display.get("Dietary_Recommendations", ""))

            _section_title("👤", "Personalized Effects")
            st.info(display.get("How_Treatment_Affects_Your_Body_Type", ""))

    # Auto-update on change if symptoms entered; also respond to button
    if predict_now or st.session_state.get("Symptoms", "").strip():
        _predict_and_render()

st.markdown(
    """
    <div style='margin-top:24px;color:#94a3b8;font-size:0.9rem'>
    Disclaimer: This system is for educational and research purposes. Consult qualified healthcare professionals for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)



