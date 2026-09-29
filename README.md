# 🌍 Asteroid Diameter Prediction DNN

A Deep Neural Network application that predicts asteroid diameter from orbital and physical properties. Features a user-friendly Streamlit web interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project builds a comprehensive machine learning pipeline to:

1. **Load and preprocess** asteroid orbital and physical data
2. **Train a Deep Neural Network** to predict asteroid diameter
3. **Deploy as a web application** using Streamlit for easy interaction

### Key Features

- 🧠 Deep Neural Network with 4 hidden layers
- 📊 Uses 20+ orbital and physical properties as features
- 📈 Comprehensive data preprocessing and scaling
- 🎯 Early stopping to prevent overfitting
- 🌐 Interactive Streamlit web interface
- 📱 Real-time predictions and visualizations
- 📏 Size comparisons with known celestial objects

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rini43/Asteroid_Diameter_Prediction.git
   cd Asteroid_Diameter_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load the asteroid dataset (`dataset.csv`)
- Preprocess and scale the features
- Build and train the DNN model
- Evaluate performance on test data
- Save the model, scaler, and feature columns

**Output files:**
- `models/asteroid_diameter_dnn_model.h5` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_columns.pkl` - Feature column names

### 2. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
Asteroid_Diameter_Prediction/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── dataset.csv                     # Asteroid data (required)
├── models/                         # Directory for trained models
│   ├── asteroid_diameter_dnn_model.h5
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── README.md                       # This file
└── Exit_exam.ipynb                # Original Jupyter notebook (reference)
```

## Model Architecture

### Network Structure

```
Input Layer (21 features)
        ↓
Dense(128, relu) + Dropout(0.2)
        ↓
Dense(64, relu) + Dropout(0.2)
        ↓
Dense(32, relu) + Dropout(0.1)
        ↓
Dense(16, relu)
        ↓
Output Layer (1 neuron, linear)
```

### Hyperparameters

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 16
- **Epochs**: 100 (with early stopping)
- **Early Stopping**: Patience = 15 epochs
- **Validation Split**: 0.2

## Usage

### Web Application Interface

The Streamlit app provides three main tabs:

#### 1. 🔮 Predictor
- Input asteroid orbital and physical properties
- Get instant diameter predictions
- View size comparisons with known objects
- Get insights about the asteroid

#### 2. 📖 About
- Learn about the model architecture
- Understand training methodology
- View technical details

#### 3. ⚙️ Model Info
- View all features used by the model
- Understand what inputs the model requires
- Training performance information

### Example Input

**Orbital Properties:**
- Semi-major axis (a): 2.5 AU
- Eccentricity (e): 0.2
- Inclination (i): 10 degrees
- Longitude of ascending node (Omega): 45 degrees
- Argument of perihelion (w): 120 degrees
- Mean anomaly (ma): 200 degrees
- Aphelion distance (ad): 3.0 AU
- Orbital period (per): 3.96 years
- Mean motion (n): 0.3 degrees/day
- Time of perihelion (tp): 2451545 JD

**Physical Properties:**
- Absolute magnitude (H): 5.5
- Various sigma values (uncertainties): 1e-6 to 1e-4

### Command Line Usage

**Train only (no web interface):**
```bash
python train_model.py
```

**Run web app only (assumes model is already trained):**
```bash
streamlit run app.py
```

## Results

After training, the model will display:

```
==================================================
MODEL EVALUATION RESULTS
==================================================
Mean Squared Error (MSE): [value]
Root Mean Squared Error (RMSE): [value]
Mean Absolute Error (MAE): [value]
R² Score: [value]
==================================================
```

### Interpretation

- **RMSE**: Average prediction error in kilometers
- **MAE**: Mean absolute error (more robust to outliers)
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MSE**: Sum of squared errors

## Data Features

The model uses **21 orbital and physical features**:

### Orbital Features (10)
1. **a** - Semi-major axis (AU)
2. **e** - Eccentricity
3. **i** - Inclination (degrees)
4. **Omega** - Longitude of ascending node (degrees)
5. **w** - Argument of perihelion (degrees)
6. **ma** - Mean anomaly (degrees)
7. **ad** - Aphelion distance (AU)
8. **per** - Orbital period (years)
9. **n** - Mean motion (degrees/day)
10. **tp** - Time of perihelion passage (JD)

### Physical Properties (11)
11. **H** - Absolute magnitude
12. **sigma_a** - Uncertainty in semi-major axis
13. **sigma_e** - Uncertainty in eccentricity
14. **sigma_i** - Uncertainty in inclination
15. **sigma_om** - Uncertainty in Omega
16. **sigma_w** - Uncertainty in w
17. **sigma_ma** - Uncertainty in mean anomaly
18. **sigma_ad** - Uncertainty in aphelion distance
19. **sigma_n** - Uncertainty in mean motion
20. **sigma_tp** - Uncertainty in perihelion time
21. **sigma_per** - Uncertainty in orbital period

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Data preprocessing and metrics
- **Pandas/NumPy**: Data manipulation
- **Streamlit**: Web application framework
- **Matplotlib/Seaborn**: Visualization

## Dataset

The dataset (`dataset.csv`) should contain asteroid data with columns for:
- Orbital elements (a, e, i, Omega, w, ma, ad, per, n, tp)
- Physical properties (H, diameter)
- Uncertainties (sigma_* columns)

Example format:
```csv
id,spkid,full_name,pdes,name,prefix,neo,pha,H,diameter,a,e,i,Omega,w,ma,ad,per,n,tp,sigma_a,sigma_e,...
a0000001,2000001,1 Ceres,1,Ceres,NaN,N,N,3.40,939.400,...
```

## Tips for Best Results

1. **Data Quality**: Ensure your dataset has minimal missing values
2. **Feature Scaling**: The training script automatically scales features
3. **Model Retraining**: Retrain with new data if predictions seem off
4. **Input Validation**: The app validates all inputs automatically
5. **Error Analysis**: Check training metrics if predictions are inaccurate

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions or issues, please open a GitHub issue in the repository.

---

**Built with ❤️ for asteroid science enthusiasts**
