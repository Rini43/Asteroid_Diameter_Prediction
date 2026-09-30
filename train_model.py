import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration

DATA_PATH = "dataset.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "asteroid_diameter_model.keras"
)

IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "asteroid_imputer.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "asteroid_scaler.pkl"
)
FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "asteroid_features.pkl"
)

# Load data
def load_and_prepare_data(filepath):
    """
    Load asteroid data and prepare features for modeling.
    """
    print("=" * 70)
    print("Loading data...")
    print("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    df = pd.read_csv(filepath)
    
    # Display data info
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df
    
# Preprocessing

def select_features(df):
    """
    Select orbital and physical properties as features.
    Target: diameter
    """
    # Features: orbital and physical properties
    
    df = df.copy()
    
    # Encode binary columns

    if "neo" in df.columns:
        df["neo"] = df["neo"].map({
            "Y": 1,
            "N": 0
        })

    if "pha" in df.columns:
        df["pha"] = df["pha"].map({
            "Y": 1,
            "N": 0
        })
    # One-hot encode asteroid class

    if "class" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["class"],
            drop_first=True,
            dtype=int
        )
     # Exclude identifier columns and the target variable
     
    exclude_cols = [
        "id", "spkid", "full_name", "pdes",
        "name", "prefix", "orbit_id", "equinox",
        "epoch_cal", "albedo", "diameter_sigma"]

    existing_exclude_cols = [
        col for col in exclude_cols
        if col in df.columns
    ]

    df.drop(columns=existing_exclude_cols,
        inplace=True)
    
    # Remove rows where target is missing

    if "diameter" not in df.columns:
        raise ValueError(
            "Target column 'diameter' was not found."
        )

    df = df.dropna(
        subset=["diameter"]
    ).copy()

    print(
        f"Rows after removing missing diameter: "
        f"{len(df)}"
    )

    # Separate features and target

    X = df.drop(
        columns=["diameter"]
    )
    y = df["diameter"]

    # Ensure only numeric features remain

    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:

        print(
            "\nDropping non-numeric columns:"
        )
        print(non_numeric)

        X = X.drop(
            columns=non_numeric
        )

    print(
        f"\nNumber of input features: {X.shape[1]}"
    )

    print(
        f"Target samples: {len(y)}"
    )

    print(
        "\nFeatures:"
    )
    print(X.columns.tolist())

    return X, y

# Build DNN

def build_dnn_model(input_dim):
    """
    Build a Deep Neural Network for regression.
    """
    model = keras.Sequential([

        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),

        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu"),

        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")  # Output layer for regression
    ])

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=0.001),
        loss="mse",
        metrics=["mae","mse"]
    )

    model.summary()
    return model

# Evaluation

def evaluate_model(
    model, X_train, y_train,
    X_test, y_test):
        
    y_train_pred = model.predict(
        X_train, verbose=0).flatten()

    y_test_pred = model.predict(
        X_test, verbose=0).flatten()

    # Training metrics

    train_mse = mean_squared_error(
        y_train, y_train_pred)

    train_rmse = np.sqrt(train_mse)

    train_mae = mean_absolute_error(
        y_train,y_train_pred)

    train_r2 = r2_score(
        y_train, y_train_pred)

    # Test metrics

    test_mse = mean_squared_error(y_test,
        y_test_pred)

    test_rmse = np.sqrt(test_mse)

    test_mae = mean_absolute_error(
        y_test, y_test_pred)

    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTraining Results")
    print("-" * 50)

    print(f"MSE  : {train_mse:.4f}")
    print(f"RMSE : {train_rmse:.4f} km")
    print(f"MAE  : {train_mae:.4f} km")
    print(f"R²   : {train_r2:.4f}")
        
    print("\nTest Results")
    print("-" * 50)

    print(f"MSE  : {test_mse:.4f}")
    print(f"RMSE : {test_rmse:.4f} km")
    print(f"MAE  : {test_mae:.4f} km")
    print(f"R²   : {test_r2:.4f}")

    return {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2
    }

# Main

def main():

    print("\n")
    print("=" * 70)
    print("ASTEROID DIAMETER PREDICTION")
    print("=" * 70)

    # Create model directory

    os.makedirs(
        MODEL_DIR,
        exist_ok=True)

    # Load

    df = load_data(
        DATA_PATH)

    # Preprocess

    X, y = preprocess_data(
        df)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4: TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    print(
        f"Training samples: {X_train.shape}"
    )

    print(
        f"Testing samples : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    print("\nApplying median imputation...")

    numeric_cols = X_train.select_dtypes(
        include=[np.number]
    ).columns

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train[numeric_cols] = imputer.fit_transform(
        X_train[numeric_cols]
    )

    X_test[numeric_cols] = imputer.transform(
        X_test[numeric_cols]
    )

    # Verify missing values

    print(
        "Missing values in training:",
        X_train.isna().sum().sum()
    )

    print(
        "Missing values in testing :",
        X_test.isna().sum().sum()
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Save preprocessing objects
    # --------------------------------------------------------

    joblib.dump(
        imputer,
        IMPUTER_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        X_train.columns.tolist(),
        FEATURES_PATH
    )

    print(
        f"\nImputer saved: {IMPUTER_PATH}"
    )

    print(
        f"Scaler saved : {SCALER_PATH}"
    )

    print(
        f"Features saved: {FEATURES_PATH}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        X_train_scaled.shape[1]
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5: TRAIN MODEL")
    print("=" * 70)

    history = model.fit(

        # IMPORTANT:
        # Train using SCALED data.
        X_train_scaled,

        y_train,

        validation_split=0.20,

        epochs=50,

        batch_size=256,

        callbacks=[
            early_stop,
            reduce_lr
        ],

        verbose=1
    )

    print(
        "\nTraining completed!"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(

        model,

        X_train_scaled,
        y_train,

        X_test_scaled,
        y_test
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 7: SAVE MODEL")
    print("=" * 70)

    model.save(
        MODEL_PATH
    )

    print(
        f"Model saved: {MODEL_PATH}"
    )

    print("\n" + "=" * 70)
    print("ALL MODEL FILES SAVED SUCCESSFULLY")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"1. {MODEL_PATH}"
    )

    print(
        f"2. {IMPUTER_PATH}"
    )

    print(
        f"3. {SCALER_PATH}"
    )

    print(
        f"4. {FEATURES_PATH}"
    )

    print("\nFinal Test Metrics:")

    print(
        f"RMSE: {metrics['test_rmse']:.4f} km"
    )

    print(
        f"MAE : {metrics['test_mae']:.4f} km"
    )

    print(
        f"R²  : {metrics['test_r2']:.4f}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()






    
def train_model(X, y, model, epochs=10, batch_size=16, validation_split=0.2):
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
