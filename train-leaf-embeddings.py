"""
FULL TRAINING SCRIPT — WITH SMOTE ADDED FOR ML MODELS
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_PATH = "data/creditcard - Copy.csv"
OUT_DIR = "models"

os.makedirs(os.path.join(OUT_DIR, "ml", "with_smote"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "ml", "without_smote"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "dl"), exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 100
BATCH = 1024


# =====================================================================
# SPLIT FUNCTION
# =====================================================================
def stratified_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(X, y))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    val_rel = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def find_best_threshold_by_f1(y_true, y_probs):
    best_t = 0.5
    best_f1 = -1
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# =====================================================================
# LOAD + SCALE DATA
# =====================================================================
df = pd.read_csv(DATA_PATH)
y = df["Class"]
X = df.drop(columns=["Class"])

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save scaler (used by evaluation)
joblib.dump(scaler, os.path.join(OUT_DIR, "ml", "without_smote", "scaler.pkl"))

X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X_scaled, y)

print("Dataset shapes:")
print("Train:", X_train.shape, y_train.sum())
print("Val:", X_val.shape, y_val.sum())
print("Test:", X_test.shape, y_test.sum())


# =====================================================================
# ML MODELS WITHOUT SMOTE (BASELINE)
# =====================================================================
print("\nTraining ML models WITHOUT SMOTE...")
ml_no_smote = [
    (LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1), "logistic_regression"),
    (RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1), "random_forest"),
    (xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=300), "xgboost"),
]

for clf, name in ml_no_smote:
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(OUT_DIR, "ml", "without_smote", f"{name}.pkl"))
    print(f"Saved: {name}")


# =====================================================================
# ML MODELS WITH SMOTE  (NEW PART YOU REQUESTED)
# =====================================================================
print("\n🔵 Training ML models WITH SMOTE...")

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

ml_with_smote = [
    (LogisticRegression(max_iter=1000), "logistic_regression"),
    (RandomForestClassifier(n_estimators=200, n_jobs=-1), "random_forest"),
    (xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=300), "xgboost"),
]

for clf, name in ml_with_smote:
    print(f"Training {name} (SMOTE)...")
    clf.fit(X_train_sm, y_train_sm)
    joblib.dump(clf, os.path.join(OUT_DIR, "ml", "with_smote", f"{name}.pkl"))
    print(f"Saved SMOTE model: {name}")


# =====================================================================
# TRAIN XGB FOR LEAF EMBEDDINGS (TRAIN ONLY)
# =====================================================================
print("\nTraining XGBoost leaf extractor...")
xgb_leaf_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=300)
xgb_leaf_model.fit(X_train, y_train)

joblib.dump(xgb_leaf_model, os.path.join(OUT_DIR, "ml", "without_smote", "xgb_leaf_model.pkl"))

leaf_train = xgb_leaf_model.apply(X_train)
leaf_val = xgb_leaf_model.apply(X_val)
leaf_test = xgb_leaf_model.apply(X_test)

leaf_train = np.array(leaf_train, dtype=np.int32)
leaf_val = np.array(leaf_val, dtype=np.int32)
leaf_test = np.array(leaf_test, dtype=np.int32)

n_trees = leaf_train.shape[1]

# Normalize leaf indices
mins = leaf_train.min(axis=0)
leaf_train_norm = leaf_train - mins
leaf_val_norm = leaf_val - mins
leaf_test_norm = leaf_test - mins
leaf_cardinalities = leaf_train_norm.max(axis=0) + 1

print("Leaf cardinalities sample:", leaf_cardinalities[:10])


# =====================================================================
# BUILD DL MODEL
# =====================================================================
def build_tabular_model(input_dim, n_trees, leaf_cardinalities, embedding_dim=8, dropout=0.2):
    num_in = layers.Input(shape=(input_dim,), name="numeric_input")

    emb_inputs = []
    emb_layers = []

    for i in range(n_trees):
        inp = layers.Input(shape=(1,), dtype='int32', name=f"leaf_in_{i}")
        emb = layers.Embedding(
            input_dim=leaf_cardinalities[i],
            output_dim=min(embedding_dim, max(2, leaf_cardinalities[i] // 10)),
            name=f"leaf_emb_{i}"
        )(inp)
        emb = layers.Reshape((emb.shape[-1],))(emb)
        emb_inputs.append(inp)
        emb_layers.append(emb)

    merged = layers.Concatenate()([num_in] + emb_layers)

    def block(x, units):
        y = layers.Dense(units)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Dropout(dropout)(y)
        return y

    x = block(merged, 256)
    x = layers.Add()([x, layers.Dense(256)(x)])
    x = block(x, 128)
    x = block(x, 64)
    x = block(x, 32)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=[num_in] + emb_inputs, outputs=out)


def make_inputs(X_df, leaf_norm):
    num = X_df.values.astype(np.float32)
    leaf_inputs = [leaf_norm[:, j].reshape(-1, 1) for j in range(leaf_norm.shape[1])]
    return [num] + leaf_inputs


train_inputs = make_inputs(X_train, leaf_train_norm)
val_inputs = make_inputs(X_val, leaf_val_norm)
test_inputs = make_inputs(X_test, leaf_test_norm)

input_dim = X_train.shape[1]

# CLASS WEIGHTS
pos = y_train.sum()
neg = len(y_train) - pos
class_weight = {0: 1.0, 1: min(50.0, neg / pos)}

# CALLBACKS
class F1Callback(callbacks.Callback):
    def __init__(self, val_data, save_path, patience=10):
        super().__init__()
        self.Xv, self.yv = val_data
        self.best = -1
        self.wait = 0
        self.patience = patience
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.Xv, verbose=0).ravel()
        _, f1 = find_best_threshold_by_f1(self.yv, probs)

        if f1 > self.best:
            self.best = f1
            self.wait = 0
            self.model.save(self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


# =====================================================================
# TRAIN DL MODELS (1–3)
# =====================================================================

auc_pr = tf.keras.metrics.AUC(curve="PR", name="auc_pr")

model1 = build_tabular_model(input_dim, n_trees, leaf_cardinalities)
model1.compile(optimizer="adam", loss="binary_crossentropy", metrics=[auc_pr])

model2 = build_tabular_model(input_dim, n_trees, leaf_cardinalities)
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=[auc_pr])

model3 = build_tabular_model(input_dim, n_trees, leaf_cardinalities)
model3.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="binary_crossentropy", metrics=[auc_pr])


print("\nTraining DL model1 (Class Weighted)...")
cb1 = F1Callback((val_inputs, y_val.values), os.path.join(OUT_DIR, "dl", "model1_classweight_best.keras"))
model1.fit(train_inputs, y_train.values,
           validation_data=(val_inputs, y_val.values),
           epochs=EPOCHS, batch_size=BATCH,
           class_weight=class_weight, callbacks=[cb1], verbose=2)

model1.save(os.path.join(OUT_DIR, "dl", "model1_classweight.keras"))


print("\nTraining DL model2 (Threshold model)...")
es2 = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model2.fit(train_inputs, y_train.values,
           validation_data=(val_inputs, y_val.values),
           epochs=EPOCHS, batch_size=BATCH,
           callbacks=[es2], verbose=2)

model2.save(os.path.join(OUT_DIR, "dl", "model2_threshold.keras"))

val_probs = model2.predict(val_inputs).ravel()
best_t2, best_f1_2 = find_best_threshold_by_f1(y_val, val_probs)
with open(os.path.join(OUT_DIR, "dl", "model2.best_threshold.pkl"), "wb") as f:
    pickle.dump({"best_threshold": best_t2, "val_f1": best_f1_2}, f)


print("\nTraining DL model3 (F1 optimized)...")
cb3 = F1Callback((val_inputs, y_val.values), os.path.join(OUT_DIR, "dl", "model3_f1_best.keras"))
model3.fit(train_inputs, y_train.values,
           validation_data=(val_inputs, y_val.values),
           epochs=EPOCHS, batch_size=BATCH,
           callbacks=[cb3], verbose=2)

model3.save(os.path.join(OUT_DIR, "dl", "model3_f1.keras"))


# =====================================================================
# ENSEMBLE
# =====================================================================
print("\nBuilding ensemble model...")

p1 = model1.predict(val_inputs).ravel()
p2 = model2.predict(val_inputs).ravel()
p3 = model3.predict(val_inputs).ravel()
p_ens = (p1 + p2 + p3) / 3

best_thr_ens, best_f1_ens = find_best_threshold_by_f1(y_val, p_ens)

tf.keras.models.save_model(model1, os.path.join(OUT_DIR, "dl", "dl_ensemble_m1.keras"))
tf.keras.models.save_model(model2, os.path.join(OUT_DIR, "dl", "dl_ensemble_m2.keras"))
tf.keras.models.save_model(model3, os.path.join(OUT_DIR, "dl", "dl_ensemble_m3.keras"))

with open(os.path.join(OUT_DIR, "dl", "dl_ensemble.meta.pkl"), "wb") as f:
    pickle.dump({"threshold_final": float(best_thr_ens)}, f)

print("\n======================================================")
print(" ALL TRAINING COMPLETE SUCCESSFULLY ")
print("======================================================")
