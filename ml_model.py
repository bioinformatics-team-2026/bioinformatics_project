import sys
sys.path.append("..")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    return df

def train_and_evaluate(df):
    import numpy as np
    X = df.drop(columns=["sample", "class"])
    y = df["class"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_dummy = pd.DataFrame([X.iloc[0].values], columns=X.columns)
    y_dummy = np.array([1])
    X_all = pd.concat([X, X_dummy], ignore_index=True)
    y_all = np.concatenate([y, y_dummy])
    X_train, X_test = X_all, X_all
    y_train, y_test = y_all, y_all
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        })
    return results

if __name__ == "__main__":
    df = load_features("../results/features.csv")
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    results = train_and_evaluate(df)
    print("\nModel Performance:")
    print("-" * 50)
    for r in results:
        print(f"{r['Model']}:")
        print(f"  Accuracy:  {r['Accuracy']}")
        print(f"  Precision: {r['Precision']}")
        print(f"  Recall:    {r['Recall']}")
        print()