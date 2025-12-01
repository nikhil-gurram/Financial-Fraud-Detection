import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess(path, use_smote=True):
    """
    Preprocess the credit-card fraud dataset.

    Parameters
    ----------
    path : str
        Path to CSV file.
    use_smote : bool
        Whether to apply SMOTE (default = True).

    Returns
    -------
    X_resampled, y_resampled        -> For ML models (balanced or unbalanced)
    X_train_scaled, y_train         -> DL training
    X_val_scaled, y_val             -> Validation
    X_test_scaled, y_test           -> Evaluation
    """

    # Load data
    data = pd.read_csv(path)

    # Features/target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # -----------------------------------------------------
    # 1. Train/Test split (70/30)
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # -----------------------------------------------------
    # 2. Split train → train + validation (80/20 of train)
    # -----------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=42
    )

    # -----------------------------------------------------
    # 3. Scale features (fit only on training)
    # -----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # -----------------------------------------------------
    # 4. Apply SMOTE (Toggle)
    # -----------------------------------------------------
    if use_smote:
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)
    else:
        # NO SMOTE → keep original training set
        X_resampled, y_resampled = X_train_scaled, y_train

    # -----------------------------------------------------
    # Return everything needed
    # -----------------------------------------------------
    return (
        X_resampled, y_resampled,    # Balanced or unbalanced(train)
        X_train_scaled, y_train,     # DL training
        X_val_scaled, y_val,         # Validation
        X_test_scaled, y_test        # Evaluation
    )
