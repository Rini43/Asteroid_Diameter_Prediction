import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Asteroid Diameter Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# PATHS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "asteroid_diameter_model.h5"
IMPUTER_PATH = MODEL_DIR / "asteroid_imputer.pkl"
SCALER_PATH = MODEL_DIR / "asteroid_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "asteroid_features.pkl"

# LOAD MODEL AND PREPROCESSING OBJECTS

@st.cache_resource
def load_artifacts():

    model = keras.models.load_model(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, imputer, scaler, features

# CHECK MODEL FILES

required_files = [MODEL_PATH, IMPUTER_PATH,
    SCALER_PATH, FEATURES_PATH]

missing_files = [str(file.name)
    for file in required_files
    if not file.exists()]

if missing_files:

    st.error("Required model files are missing.")
    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Make sure the `models` folder is in the same directory "
        "as app.py.")

    st.stop()

# LOAD ARTIFACTS

try:

    model, imputer, scaler, feature_names = load_artifacts()

except Exception as e:

    st.error("Could not load the trained model.")
    st.exception(e)
    st.stop()
    
def get_feature_descriptions():
    
    """
    Return descriptions of orbital and physical properties.
    """
    descriptions = {
        'a': 'Semi-major axis (AU) - Average distance from the Sun',
        'e': 'Eccentricity - Orbital shape (0=circular, 1=parabolic)',
        'i': 'Inclination (degrees) - Angle of orbital plane',
        'Omega': 'Longitude of ascending node (degrees)',
        'w': 'Argument of perihelion (degrees)',
        'ma': 'Mean anomaly (degrees)',
        'ad': 'Aphelion distance (AU) - Farthest point from Sun',
        'per': 'Orbital period (years)',
        'n': 'Mean motion (degrees/day)',
        'tp': 'Time of perihelion passage (JD)',
        'H': 'Absolute magnitude - Intrinsic brightness',
        'sigma_a': 'Uncertainty in semi-major axis',
        'sigma_e': 'Uncertainty in eccentricity',
        'sigma_i': 'Uncertainty in inclination',
        'sigma_om': 'Uncertainty in Omega',
        'sigma_w': 'Uncertainty in w',
        'sigma_ma': 'Uncertainty in mean anomaly',
        'sigma_ad': 'Uncertainty in aphelion distance',
        'sigma_n': 'Uncertainty in mean motion',
        'sigma_tp': 'Uncertainty in perihelion time',
        'sigma_per': 'Uncertainty in orbital period'
    }
    return descriptions

# HEADER

st.title("☄️ Asteroid Diameter Prediction")

st.markdown(
    """
    ### AI-powered asteroid diameter estimation

    Enter the available astronomical and orbital parameters
    below to estimate the asteroid's diameter in **kilometers**.
    """
)

st.divider()

# INFORMATION

with st.expander("ℹ️ About this application"):

    st.write(
        """
        This application uses a trained Deep Neural Network (DNN)
        to estimate asteroid diameter.

        The model was trained using astronomical and orbital
        parameters after preprocessing with:

        - Median imputation
        - StandardScaler
        - One-hot encoding
        - Deep Neural Network regression

        The prediction should be considered an estimate and not
        a direct astronomical measurement.
        """)

# DEFAULT VALUES

# The notebook trains the model with the features saved in
# asteroid_features.pkl.

# We create an input dictionary containing every expected feature.
# Missing values can safely be handled by the trained imputer.

input_data = {feature: np.nan for feature in feature_names}

# SIDEBAR

st.sidebar.header("Asteroid Parameters")
st.sidebar.caption(
    "Enter the values available for your asteroid.")

# IMPORTANT FEATURES

st.sidebar.subheader("Physical / Main Parameters")

if "H" in input_data:

    input_data["H"] = st.sidebar.number_input(
        "Absolute Magnitude (H)",
        min_value=-10.0, max_value=40.0,
        value=15.0, step=0.1,
        help="Absolute magnitude of the asteroid.")

if "a" in input_data:

    input_data["a"] = st.sidebar.number_input(
        "Semi-major axis (a)",
        min_value=0.01, max_value=100.0,
        value=2.5, step=0.01,
        help="Semi-major axis in astronomical units.")

if "e" in input_data:

    input_data["e"] = st.sidebar.number_input(
        "Eccentricity (e)",
        min_value=0.0, max_value=0.99,
        value=0.1, step=0.01)

if "i" in input_data:

    input_data["i"] = st.sidebar.number_input(
        "Inclination (i)",
        min_value=0.0, max_value=180.0,
        value=5.0, step=0.1)

if "q" in input_data:

    input_data["q"] = st.sidebar.number_input(
        "Perihelion distance (q)",
        min_value=0.0, max_value=100.0,
        value=2.0, step=0.01)

if "ad" in input_data:

    input_data["ad"] = st.sidebar.number_input(
        "Aphelion distance (ad)",
        min_value=0.0, max_value=200.0,
        value=3.0, step=0.01)

if "moid" in input_data:

    input_data["moid"] = st.sidebar.number_input(
        "MOID", min_value=0.0,
        max_value=100.0, value=1.0, step=0.01)

# ORBITAL PARAMETERS

with st.expander("🛰️ Orbital Parameters"):

    orbital_features = [
        "epoch", "epoch_mjd", "tp", "tp_cal",
        "per", "per_y", "ma", "maia", "n", "om",
        "w", "sigma_e", "sigma_a", "sigma_q","sigma_i",
        "sigma_om", "sigma_w", "sigma_ma", "sigma_ad",
        "sigma_n", "sigma_tp", "sigma_per",]

    available_orbital_features = [
        feature
        for feature in orbital_features
        if feature in input_data]

    cols = st.columns(3)

    for index, feature in enumerate(available_orbital_features):

        with cols[index % 3]:

            input_data[feature] = st.number_input(
                feature, value=0.0, format="%.6f",
                key=f"orbital_{feature}")

# OBSERVATIONAL PARAMETERS

with st.expander("🔭 Observation Parameters"):

    observation_features = [
        "data_arc", "n_obs_used", "n_del_obs_used",
        "n_dop_obs_used", "rms", "condition_code",
        "epoch_mjd"]

    available_observation_features = [
        feature
        for feature in observation_features
        if feature in input_data]

    cols = st.columns(3)

    for index, feature in enumerate(available_observation_features):

        with cols[index % 3]:

            input_data[feature] = st.number_input(
                feature, value=0.0, format="%.6f",
                key=f"observation_{feature}")

# NEO / PHA

with st.expander("🌍 Classification"):

    if "neo" in input_data:

        neo_value = st.selectbox(
            "Near-Earth Object (NEO)",
            ["No", "Yes"])

        input_data["neo"] = 1 if neo_value == "Yes" else 0

    if "pha" in input_data:

        pha_value = st.selectbox(
            "Potentially Hazardous Asteroid (PHA)",
            ["No", "Yes"])

        input_data["pha"] = 1 if pha_value == "Yes" else 0

# CLASS FEATURES

class_features = [
    feature
    for feature in feature_names
    if feature.startswith("class_")]

if class_features:

    with st.expander("🔬 Asteroid Class"):

        selected_class = st.selectbox(
            "Class",
            ["Default"] + class_features)

        # One-hot encoding equivalent to the notebook:
        # pd.get_dummies(..., drop_first=True)

        for feature in class_features:

            input_data[feature] = (
                1 if feature == selected_class else 0)

# PREDICTION

st.divider()
st.subheader("🔮 Prediction")

if st.button(
    "☄️ Predict Asteroid Diameter",
    type="primary",
    use_container_width=True):

    try:

        # Create dataframe

        input_df = pd.DataFrame([input_data])

        # IMPORTANT:
        # Force exact feature order used during training.

        input_df = input_df.reindex(
            columns=feature_names)

        # Convert everything to numeric

        input_df = input_df.apply(
            pd.to_numeric,
            errors="coerce")

        # Imputation

        input_imputed = imputer.transform(
            input_df)

        # Scaling

        input_scaled = scaler.transform(
            input_imputed)

        # Prediction

        prediction = model.predict(
            input_scaled, verbose=0)

        diameter = float(
            np.asarray(prediction).reshape(-1)[0])

        # Safety check

        if not np.isfinite(diameter):

            st.error(
                "The model returned an invalid prediction.")

        else:

            # Diameter cannot physically be negative.
            diameter = max(0.0, diameter)

            st.success(
                "Prediction completed successfully!")

            # RESULT

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Predicted Diameter",
                    f"{diameter:.3f} km")

            with col2:

                st.metric("Predicted Diameter",
                    f"{diameter * 1000:.1f} m")

            with col3:

                st.metric("Model Input Features",
                    len(feature_names))

            # Interpretation

            st.info(
                f"""
                The trained DNN estimates the asteroid diameter
                at approximately **{diameter:.3f} km**.

                This is an ML-based estimate and should not be
                treated as a direct astronomical measurement.
                """
            )

            # Show processed input

            with st.expander("🔍 View model input"):

                display_df = input_df.T
                display_df.columns = ["Value"]

                st.dataframe(display_df, use_container_width=True)

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)

# FOOTER

st.divider()
st.caption("Asteroid Diameter Prediction • Deep Learning Regression")


            
