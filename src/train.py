import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

# Load preprocessed data
X_res, y_res, X_test, y_test = load_and_preprocess("data/creditcard.csv")

models = {
    "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "xgboost": XGBClassifier(scale_pos_weight=len(y_res[y_res==0]) / len(y_res[y_res==1]))
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_res, y_res)
    joblib.dump(model, f"models/{name}.pkl")
