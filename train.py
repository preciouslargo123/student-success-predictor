import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import RAW_FEATURES, add_features

os.chdir(Path(__file__).parent)
df = pd.read_csv("student_data.csv")
print("Rows:", df.shape[0], "| classes:", df.rating.value_counts().to_dict(),
      "| duplicate rows:", df.duplicated().sum())

X = df[RAW_FEATURES]
y = (df.rating == "High").astype(int)          # 1 = Pass (High), 0 = Fail (Low)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

FE = FunctionTransformer(add_features)
models = {
    "Logistic Regression": make_pipeline(FE, FunctionTransformer(np.log1p), StandardScaler(),
                                         LogisticRegression(max_iter=3000)),
    "Gradient Boosting": make_pipeline(FE, GradientBoostingClassifier(random_state=42)),
    "Random Forest": make_pipeline(FE, RandomForestClassifier(500, random_state=42, n_jobs=-1)),
    "Extra Trees": make_pipeline(FE, ExtraTreesClassifier(500, max_features=None,
                                                          random_state=42, n_jobs=-1)),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    cv_acc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="accuracy").mean()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    results[name] = dict(cv=cv_acc, test=accuracy_score(yte, pred), f1=f1_score(yte, pred))
    print(f"{name:20s} CV acc={cv_acc:.3f}  Test acc={results[name]['test']:.3f}  F1={results[name]['f1']:.3f}")

best = max(results, key=lambda k: results[k]["test"])
model = models[best]
print(f"\nBest model: {best}")
print(classification_report(yte, model.predict(Xte), target_names=["Fail (Low)", "Pass (High)"]))

# Stricter check: keep every copy of a duplicated student on one side of the split
groups = X.astype(str).agg("|".join, axis=1)
unseen = [clone(model).fit(X.iloc[tr], y.iloc[tr]).score(X.iloc[te], y.iloc[te])
          for tr, te in GroupShuffleSplit(5, test_size=0.2, random_state=0).split(X, y, groups)]
unseen_acc = float(np.mean(unseen))
print(f"Accuracy on students never seen in training: {unseen_acc:.3f}")

# Feature importance on the 8 original inputs (summing engineered columns into their source)
est = model[-1]
imp = dict(zip(RAW_FEATURES, [0.0] * 8))
if hasattr(est, "feature_importances_"):
    for col, v in zip(add_features(X.head(1)).columns, est.feature_importances_):
        src = next((f for f in RAW_FEATURES if col.startswith(f)), None)
        if src:
            imp[src] += float(v)
        else:  # total_actions / read_share -> spread evenly
            for f in RAW_FEATURES[:7]:
                imp[f] += float(v) / 7

model.fit(X, y)  # final model trained on all data
joblib.dump(model, "model.joblib", compress=3)
json.dump({
    "model": best,
    "cv_accuracy": round(results[best]["cv"], 4),
    "test_accuracy": round(results[best]["test"], 4),
    "test_f1": round(results[best]["f1"], 4),
    "unseen_accuracy": round(unseen_acc, 4),
    "n_rows": len(df),
    "importances": {k: round(v, 4) for k, v in imp.items()},
    "medians": X.median().to_dict(),
}, open("model_meta.json", "w"), indent=1)
print("Saved model.joblib and model_meta.json")