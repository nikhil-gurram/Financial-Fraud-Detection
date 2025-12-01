# train_models.py
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, backend as K

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import load_and_preprocess  # keep your existing loader

# -------------------------
# Utility: Build Keras model
# -------------------------
def build_deep_mlp(input_dim, dropout=0.2):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(16)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# -------------------------
# Callback: compute F1 on validation
# -------------------------
from sklearn.metrics import f1_score

class F1Metrics(callbacks.Callback):
    def __init__(self, validation_data, patience=10, save_path=None):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.best_f1 = -1.0
        self.patience = patience
        self.wait = 0
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        y_pred_prob = self.model.predict(self.X_val, verbose=0).ravel()
        # use default 0.5 threshold for monitoring; threshold tuned separately
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        logs = logs or {}
        logs['val_f1'] = f1
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            if self.save_path:
                self.model.save(self.save_path, include_optimizer=False)
        else:
            self.wait += 1

# -------------------------
# Threshold tuning utility
# -------------------------
from sklearn.metrics import precision_recall_curve

def find_best_threshold(y_true, y_probs, metric="f1"):
    # Scan many thresholds and pick best by f1
    best_t = 0.5
    best_score = -1
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

# -------------------------
# Ensemble management
# -------------------------
import pickle

class EnsembleDL:
    """
    Soft-average ensemble wrapper. Use .save(path) to persist.
    When saved it writes:
      - models as keras files in models/dl/ensemble_model_i.h5
      - json metadata (thresholds) in path + '.meta'
    Loading should use EnsembleDL.load(path)
    """
    def __init__(self, model1, model2, model3, threshold_model2=None, threshold_final=0.5):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.threshold_model2 = threshold_model2
        self.threshold_final = threshold_final

    def predict_proba(self, X):
        # returns (N,) array of probabilities
        p1 = self.model1.predict(X).ravel()
        p2 = self.model2.predict(X).ravel()
        p3 = self.model3.predict(X).ravel()

        # if a special threshold was chosen for model2 during its training,
        # we still want to use raw probabilities in ensemble — average them.
        p_final = (p1 + p2 + p3) / 3.0
        return p_final

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= self.threshold_final).astype(int)

    def save(self, path_base):
        # path_base: e.g. "models/dl/dl_ensemble"
        os.makedirs(os.path.dirname(path_base), exist_ok=True)
        # save keras submodels to files next to path_base
        self.model1.save(path_base + "_m1.h5", include_optimizer=False)
        self.model2.save(path_base + "_m2.h5", include_optimizer=False)
        self.model3.save(path_base + "_m3.h5", include_optimizer=False)
        meta = {
            "threshold_model2": float(self.threshold_model2) if self.threshold_model2 is not None else None,
            "threshold_final": float(self.threshold_final)
        }
        with open(path_base + ".meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path_base):
        m1 = tf.keras.models.load_model(path_base + "_m1.h5")
        m2 = tf.keras.models.load_model(path_base + "_m2.h5")
        m3 = tf.keras.models.load_model(path_base + "_m3.h5")
        with open(path_base + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(m1, m2, m3, threshold_model2=meta.get("threshold_model2"), threshold_final=meta.get("threshold_final", 0.5))

# -------------------------
# TRAINING pipeline
# -------------------------
if __name__ == "__main__":
    os.makedirs("models/ml/with_smote", exist_ok=True)
    os.makedirs("models/ml/without_smote", exist_ok=True)
    os.makedirs("models/dl", exist_ok=True)

    print("Loading datasets (SMOTE and NON-SMOTE)...")
    # WITH SMOTE for ML training
    X_res_smote, y_res_smote, X_train_sm, y_train_sm, X_val_sm, y_val_sm, X_test_sm, y_test_sm = load_and_preprocess("data/creditcard - Copy.csv", use_smote=True)

    # WITHOUT SMOTE for DL training (paper uses DL on unbalanced)
    X_res_no, y_res_no, X_train_no, y_train_no, X_val_no, y_val_no, X_test_no, y_test_no = load_and_preprocess("data/creditcard - Copy.csv", use_smote=False)

    # -----------------------
    # ML MODELS (with and without SMOTE)
    # -----------------------
    ml_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1),
        "xgboost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=200,
            scale_pos_weight=(len(y_res_smote[y_res_smote==0]) / max(1, len(y_res_smote[y_res_smote==1])))
        )
    }

    print("\nTraining ML models (WITH SMOTE)...")
    for name, model in ml_models.items():
        model.fit(X_res_smote, y_res_smote)
        joblib.dump(model, f"models/ml/with_smote/{name}.pkl")
        print("Saved:", f"models/ml/with_smote/{name}.pkl")

    print("\nTraining ML models (NO SMOTE)...")
    for name, model in ml_models.items():
        model.fit(X_res_no, y_res_no)
        joblib.dump(model, f"models/ml/without_smote/{name}.pkl")
        print("Saved:", f"models/ml/without_smote/{name}.pkl")

    # -----------------------
    # DEEP LEARNING MODELS (No SMOTE) — Paper-style improvements
    # -----------------------
    input_dim = X_train_no.shape[1]
    print("\nBuilding DL model 1 (class-weighted)...")
    model1 = build_deep_mlp(input_dim)
    model1.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["AUC"])
    # compute class weights: keep w0=1, w1 = N_neg / N_pos (but limit extreme values)
    n_pos = y_train_no.sum()
    n_neg = len(y_train_no) - n_pos
    w1 = float(n_neg / max(1, n_pos))
    class_weight = {0: 1.0, 1: min(w1, 50.0)}  # clamp to avoid extreme gradients
    cb_f1 = F1Metrics(validation_data=(X_val_no, y_val_no), patience=10, save_path="models/dl/model1_classweight.best.h5")
    model1.fit(X_train_no, y_train_no, validation_data=(X_val_no, y_val_no),
               epochs=100, batch_size=1024, class_weight=class_weight,
               callbacks=[cb_f1], verbose=2)
    # load best if callback saved
    if os.path.exists("models/dl/model1_classweight.best.h5"):
        model1 = tf.keras.models.load_model("models/dl/model1_classweight.best.h5")
    model1.save("models/dl/model1_classweight.h5", include_optimizer=False)

    print("\nBuilding DL model 2 (threshold-tuned)...")
    model2 = build_deep_mlp(input_dim)
    model2.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["AUC"])
    cb2 = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model2.fit(X_train_no, y_train_no, validation_data=(X_val_no, y_val_no),
               epochs=100, batch_size=1024, callbacks=[cb2], verbose=2)
    # find best threshold on validation set
    val_probs_m2 = model2.predict(X_val_no).ravel()
    best_threshold_m2, best_f1_m2 = find_best_threshold(y_val_no, val_probs_m2)
    print("Best threshold for model2:", best_threshold_m2, "val F1:", best_f1_m2)
    model2.save("models/dl/model2_threshold.h5", include_optimizer=False)
    joblib.dump(best_threshold_m2, "models/dl/best_threshold.pkl")

    print("\nBuilding DL model 3 (F1-optimized training via callback)...")
    model3 = build_deep_mlp(input_dim)
    model3.compile(optimizer=optimizers.Adam(5e-4), loss="binary_crossentropy", metrics=["AUC"])
    cb3 = F1Metrics(validation_data=(X_val_no, y_val_no), patience=12, save_path="models/dl/model3_f1.best.h5")
    model3.fit(X_train_no, y_train_no, validation_data=(X_val_no, y_val_no),
               epochs=200, batch_size=1024, callbacks=[cb3], verbose=2)
    if os.path.exists("models/dl/model3_f1.best.h5"):
        model3 = tf.keras.models.load_model("models/dl/model3_f1.best.h5")
    model3.save("models/dl/model3_f1.h5", include_optimizer=False)

    # -----------------------
    # ENSEMBLE — soft average + tuned final threshold
    # -----------------------
    print("\nBuilding Ensemble...")
    # load saved models to ensure consistent saved weights
    m1 = tf.keras.models.load_model("models/dl/model1_classweight.h5")
    m2 = tf.keras.models.load_model("models/dl/model2_threshold.h5")
    m3 = tf.keras.models.load_model("models/dl/model3_f1.h5")

    # find a global ensemble threshold on validation by averaging probabilities
    p1_val = m1.predict(X_val_no).ravel()
    p2_val = m2.predict(X_val_no).ravel()
    p3_val = m3.predict(X_val_no).ravel()
    p_ensemble_val = (p1_val + p2_val + p3_val) / 3.0
    best_thr_ens, best_f1_ens = find_best_threshold(y_val_no, p_ensemble_val)
    print("Ensemble best threshold:", best_thr_ens, "F1:", best_f1_ens)

    ensemble = EnsembleDL(model1=m1, model2=m2, model3=m3, threshold_model2=best_threshold_m2, threshold_final=best_thr_ens)
    # save ensemble safely
    ensemble.save("models/dl/dl_ensemble")
    print("Saved ensemble to models/dl/dl_ensemble*_ and .meta.pkl")
    print("\nTRAINING COMPLETE")
