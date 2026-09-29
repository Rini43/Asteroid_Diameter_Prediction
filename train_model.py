import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_data(filepath):
    """
    Load asteroid data and prepare features for modeling.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Display data info
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df

def select_features(df):
    """
    Select orbital and physical properties as features.
    Target: diameter
    """
    # Features: orbital and physical properties
    # Exclude identifier columns and the target variable
    exclude_cols = ['id', 'spkid', 'full_name', 'pdes', 'name', 'prefix', 
                    'diameter', 'class', 'neo', 'pha']  # exclude categorical and identifiers
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nSelected {len(feature_cols)} features for modeling")
    print(f"Features: {feature_cols}")
    
    # Handle missing values
    df_clean = df[feature_cols + ['diameter']].copy()
    df_clean = df_clean.dropna()
    
    print(f"Data shape after removing NaN: {df_clean.shape}")
    
    X = df_clean[feature_cols].values
    y = df_clean['diameter'].values
    
    return X, y, feature_cols

def build_dnn_model(input_dim):
    """
    Build a Deep Neural Network for regression.
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X, y, model, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train the DNN model.
    """
    print("\nTraining model...")
    
    # Early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    # Update this path to your actual dataset location
    filepath = 'dataset.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        print("Please provide the asteroid dataset CSV file.")
        return
    
    df = load_and_prepare_data(filepath)
    X, y, feature_cols = select_features(df)
    
    # Split data
    print(f"\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Build model
    print(f"\nBuilding DNN model with {X_train_scaled.shape[1]} input features...")
    model = build_dnn_model(X_train_scaled.shape[1])
    print(model.summary())
    
    # Train model
    history = train_model(X_train_scaled, y_train, model)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model
    model.save('models/asteroid_diameter_dnn_model.h5')
    print("\nModel saved to models/asteroid_diameter_dnn_model.h5")
    
    # Save feature columns
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    print("Feature columns saved to models/feature_columns.pkl")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
