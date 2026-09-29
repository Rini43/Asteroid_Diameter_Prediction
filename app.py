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

@st.cache_resource
def load_model_and_scaler():
    """
    Load the trained DNN model and scaler.
    """
    try:
        model_path = 'models/asteroid_diameter_dnn_model.h5'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/feature_columns.pkl'
        
        if not all([os.path.exists(p) for p in [model_path, scaler_path, features_path]]):
            st.error("❌ Model files not found. Please train the model first by running `python train_model.py`")
            st.stop()
        
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)
        
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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

def create_input_section():
    """
    Create input section for user to enter asteroid properties.
    """
    st.subheader("📊 Enter Asteroid Properties")
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    descriptions = get_feature_descriptions()
    input_values = {}
    
    # Note: This is a simplified version. In practice, you'd want to match
    # the exact features your model was trained on.
    orbital_features = [
        'a', 'e', 'i', 'Omega', 'w', 'ma', 'ad', 'per', 'n', 'tp'
    ]
    
    physical_features = [
        'H', 'sigma_a', 'sigma_e', 'sigma_i', 'sigma_om', 'sigma_w',
        'sigma_ma', 'sigma_ad', 'sigma_n', 'sigma_tp', 'sigma_per'
    ]
    
    with col1:
        st.markdown("### Orbital Properties")
        for feature in orbital_features:
            if feature in descriptions:
                help_text = descriptions[feature]
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=1.0,
                    help=help_text,
                    key=f"input_{feature}"
                )
    
    with col2:
        st.markdown("### Physical Properties")
        for feature in physical_features:
            if feature in descriptions:
                help_text = descriptions[feature]
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.1,
                    help=help_text,
                    key=f"input_{feature}"
                )
    
    return input_values

def make_prediction(model, scaler, feature_cols, input_values):
    """
    Make prediction using the trained model.
    """
    try:
        # Create input array with exact feature order
        input_array = np.array([input_values.get(feature, 0.0) for feature in feature_cols])
        
        # Scale input
        input_scaled = scaler.transform(input_array.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        predicted_diameter = prediction[0][0]
        
        return predicted_diameter
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_results(diameter):
    """
    Display prediction results with visualization.
    """
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Diameter (km)",
            value=f"{diameter:.2f}",
            delta=None
        )
    
    with col2:
        # Radius in km
        radius = diameter / 2
        st.metric(
            label="Estimated Radius (km)",
            value=f"{radius:.2f}"
        )
    
    with col3:
        # Volume estimate (assuming sphere)
        volume = (4/3) * np.pi * (radius ** 3)
        st.metric(
            label="Estimated Volume (km³)",
            value=f"{volume:.2e}"
        )
    
    # Size comparison
    st.markdown("---")
    st.subheader("📏 Size Comparison")
    
    comparisons = {
        "Moon": 3474,
        "Mt. Everest (base)": 0.01,
        "Chicxulub Crater (asteroid)": 10,
        "Ceres (dwarf planet)": 946
    }
    
    comparison_data = []
    for name, size in comparisons.items():
        ratio = diameter / size if size > 0 else 0
        comparison_data.append({
            "Object": name,
            "Diameter (km)": size,
            "Ratio to Prediction": f"{ratio:.2f}x"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

def main():
    # Header
    st.title("🌍 Asteroid Diameter Predictor")
    st.markdown("""
    This application uses a Deep Neural Network (DNN) trained on asteroid orbital and physical properties
    to predict asteroid diameter. The model learns patterns from real astronomical data to make accurate predictions.
    """)
    
    st.markdown("---")
    
    # Load model and scaler
    with st.spinner("Loading model..."):
        model, scaler, feature_cols = load_model_and_scaler()
    
    st.success("✅ Model loaded successfully!")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predictor", "📖 About", "⚙️ Model Info"])
    
    with tab1:
        # Input section
        input_values = create_input_section()
        
        # Prediction button
        col1, col2 = st.columns([2, 1])
        with col2:
            predict_button = st.button("🚀 Predict Diameter", use_container_width=True)
        
        if predict_button:
            with st.spinner("Making prediction..."):
                diameter = make_prediction(model, scaler, feature_cols, input_values)
            
            if diameter is not None and diameter > 0:
                display_results(diameter)
                
                # Additional insights
                st.markdown("---")
                st.subheader("💡 Insights")
                
                if diameter < 1:
                    st.info("This is a very small asteroid, likely a meteoroid.")
                elif diameter < 10:
                    st.info("This is a small asteroid, similar in size to a city block.")
                elif diameter < 100:
                    st.info("This is a medium-sized asteroid, capable of causing significant damage if it impacted Earth.")
                elif diameter < 500:
                    st.info("This is a large asteroid, similar in size to a major city.")
                else:
                    st.warning("This is a very large asteroid, similar in size to a dwarf planet!")
            else:
                st.error("Prediction failed. Please check your inputs.")
    
    with tab2:
        st.subheader("About This Application")
        st.markdown("""
        ### 🔬 Model Details
        
        This application uses a **Deep Neural Network (DNN)** to predict asteroid diameter based on:
        - **Orbital Properties**: Semi-major axis, eccentricity, inclination, etc.
        - **Physical Properties**: Absolute magnitude and orbital uncertainties
        
        ### 🎯 Target Variable
        - **Diameter**: The physical diameter of the asteroid in kilometers
        
        ### 📊 Model Architecture
        The DNN consists of:
        - 128 neurons (ReLU) → Dropout(0.2)
        - 64 neurons (ReLU) → Dropout(0.2)
        - 32 neurons (ReLU) → Dropout(0.1)
        - 16 neurons (ReLU)
        - 1 output neuron (Linear - for regression)
        
        ### 📚 Training Details
        - Optimizer: Adam
        - Loss Function: Mean Squared Error (MSE)
        - Early Stopping: Prevents overfitting
        - Feature Scaling: StandardScaler normalization
        """)
    
    with tab3:
        st.subheader("Model Performance & Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Features Used
            The model uses the following features for prediction:
            """)
            features_df = pd.DataFrame({
                "Feature": feature_cols[:len(feature_cols)//2]
            })
            st.dataframe(features_df, use_container_width=True)
        
        with col2:
            st.markdown("&nbsp;")
            features_df2 = pd.DataFrame({
                "Feature": feature_cols[len(feature_cols)//2:]
            })
            st.dataframe(features_df2, use_container_width=True)
        
        st.markdown("---")
        st.info("""
        📌 **To view detailed model performance metrics**, run the training script:
        ```bash
        python train_model.py
        ```
        This will display MSE, RMSE, MAE, and R² scores on the test set.
        """)

if __name__ == "__main__":
    main()
