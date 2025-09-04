import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from preprocess import load_and_preprocess

X_res, y_res, X_test, y_test = load_and_preprocess("data/creditcard.csv")

results = []

for name in ["logistic_regression", "random_forest", "xgboost"]:
    model = joblib.load(f"models/{name}.pkl")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, precision, recall, f1, auc])

df = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score", "ROC-AUC"])
print(df)
df.to_csv("model_comparison.csv", index=False)
