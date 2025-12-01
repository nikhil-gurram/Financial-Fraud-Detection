import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, precision_recall_curve
import joblib

# ------------------------------
# Helper: build base NN
# ------------------------------
def build_base_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ------------------------------
# Model 1: Class Weighted NN
# ------------------------------
def train_model_class_weight(X_train, y_train, X_val, y_val):
    model = build_base_model(X_train.shape[1])

    class_weights = {0: 1, 1: 30}  # higher weight for fraud

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=256,
              validation_data=(X_val, y_val),
              class_weight=class_weights,
              verbose=1)

    return model


# ------------------------------
# Model 2: Threshold-Tuned NN
# ------------------------------
def train_model_threshold(X_train, y_train, X_val, y_val):
    model = build_base_model(X_train.shape[1])
    model.fit(X_train, y_train,
              epochs=20,
              batch_size=256,
              validation_data=(X_val, y_val),
              verbose=1)

    # tune threshold using validation set
    y_prob = model.predict(X_val)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return model, best_threshold


# ------------------------------
# Model 3: F1 Optimization via Callback
# ------------------------------
class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val) > 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred)

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save("models/dl/model3_f1_best.h5")


def train_model_f1(X_train, y_train, X_val, y_val):
    model = build_base_model(X_train.shape[1])

    callback = F1Callback(X_val, y_val)

    model.fit(X_train, y_train,
              epochs=30,
              batch_size=256,
              validation_data=(X_val, y_val),
              callbacks=[callback],
              verbose=1)

    model = tf.keras.models.load_model("models/dl/model3_f1_best.h5")
    return model


# ------------------------------
# Ensemble Soft Voting
# ------------------------------
class EnsembleDL:
    def __init__(self, model1, model2, model3, threshold_model2=0.4, threshold_final=0.5):
        self.m1 = model1
        self.m2 = model2
        self.m3 = model3
        self.th_m2 = threshold_model2
        self.th_final = threshold_final

    def predict_proba(self, X):
        p1 = self.m1.predict(X)
        p2 = self.m2.predict(X)
        p3 = self.m3.predict(X)

        return (p1 + p2 + p3) / 3

    def predict(self, X):
        p_final = self.predict_proba(X)
        return (p_final >= self.th_final).astype(int)
