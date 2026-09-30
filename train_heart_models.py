import warnings, json, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, recall_score, precision_score,
                             accuracy_score, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 1. Load data
DATA_PATH = r"C:\Users\rasto\Downloads\heart_disease_uci.csv"
if not os.path.exists(DATA_PATH):
    # Fallback to local copy if available
    DATA_PATH = "heart_disease_uci.csv"

df = pd.read_csv(DATA_PATH)
print("Loaded shape:", df.shape)

# 2. Clean data
data = df.copy()
data.loc[data["chol"] == 0, "chol"] = np.nan
data.loc[data["trestbps"] == 0, "trestbps"] = np.nan
if "id" in data.columns:
    data = data.drop(columns=["id"])

data["target"] = (data["num"] > 0).astype(int)

for c in ["fbs", "exang"]:
    if c in data.columns:
        data[c] = data[c].map({True: "True", False: "False"}).astype(object)
        data[c] = data[c].where(data[c].notna(), np.nan)

HOME_FEATURES = ["age", "sex", "cp", "trestbps", "exang", "chol", "fbs"]
CLINIC_FEATURES = HOME_FEATURES + ["restecg", "thalch", "oldpeak", "slope"]
NUMERIC_POSSIBLE = ["age", "trestbps", "chol", "thalch", "oldpeak"]

def make_preprocessor(features):
    num = [c for c in features if c in NUMERIC_POSSIBLE]
    cat = [c for c in features if c not in NUMERIC_POSSIBLE]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])

y = data["target"]
X_home = data[HOME_FEATURES]
X_clinic = data[CLINIC_FEATURES]

idx_train, idx_test = train_test_split(data.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

Xh_train, Xh_test = X_home.loc[idx_train], X_home.loc[idx_test]
Xc_train, Xc_test = X_clinic.loc[idx_train], X_clinic.loc[idx_test]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

print(f"Train size: {len(idx_train)}, Test size: {len(idx_test)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Model comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

scoring = {"accuracy": "accuracy", "recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}

def run_compare(features, X_tr, y_tr):
    res_table = []
    for name, m in models.items():
        pipe = Pipeline([("prep", make_preprocessor(features)), ("model", m)])
        cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
        row = {"Model": name}
        for s in scoring:
            row[s + "_mean"] = float(round(cv_res["test_" + s].mean(), 3))
            row[s + "_std"] = float(round(cv_res["test_" + s].std(), 3))
        res_table.append(row)
    return res_table

home_compare_res = run_compare(HOME_FEATURES, Xh_train, y_train)
clinic_compare_res = run_compare(CLINIC_FEATURES, Xc_train, y_train)

# Hyperparameter Tuning
param_grids = {
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, None], "model__min_samples_leaf": [1, 3, 5]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]},
    ),
}

def tune_mode(features, X_tr, y_tr):
    best = None
    for name, (model, grid) in param_grids.items():
        pipe = Pipeline([("prep", make_preprocessor(features)), ("model", model)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        if best is None or gs.best_score_ > best[1]:
            best = (gs.best_estimator_, gs.best_score_, name, gs.best_params_)
    return best[0], best[2], best[3], best[1]

home_model, home_best_name, home_params, home_best_auc = tune_mode(HOME_FEATURES, Xh_train, y_train)
clinic_model, clinic_best_name, clinic_params, clinic_best_auc = tune_mode(CLINIC_FEATURES, Xc_train, y_train)

# Threshold picking for ~90% recall
TARGET_RECALL = 0.90
def pick_threshold(model, X_tr, y_tr):
    proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    best_t = 0.50
    for t in np.arange(0.05, 0.96, 0.01):
        if recall_score(y_tr, (proba >= t).astype(int)) >= TARGET_RECALL:
            best_t = t
    return float(round(best_t, 2))

thr_home = pick_threshold(home_model, Xh_train, y_train)
thr_clinic = pick_threshold(clinic_model, Xc_train, y_train)

# Fit on full training set
home_model.fit(Xh_train, y_train)
clinic_model.fit(Xc_train, y_train)

# Evaluate on Test Set
proba_home = home_model.predict_proba(Xh_test)[:, 1]
pred_home = (proba_home >= thr_home).astype(int)

proba_clinic = clinic_model.predict_proba(Xc_test)[:, 1]
pred_clinic = (proba_clinic >= thr_clinic).astype(int)

cm_home = confusion_matrix(y_test, pred_home).tolist()
cm_clinic = confusion_matrix(y_test, pred_clinic).tolist()

fpr_h, tpr_h, _ = roc_curve(y_test, proba_home)
fpr_c, tpr_c, _ = roc_curve(y_test, proba_clinic)

eval_summary = {
    "home": {
        "best_model_name": home_best_name,
        "best_params": {k.replace("model__", ""): v for k, v in home_params.items()},
        "threshold": thr_home,
        "accuracy": float(round(accuracy_score(y_test, pred_home), 3)),
        "recall": float(round(recall_score(y_test, pred_home), 3)),
        "precision": float(round(precision_score(y_test, pred_home), 3)),
        "f1": float(round(f1_score(y_test, pred_home), 3)),
        "roc_auc": float(round(roc_auc_score(y_test, proba_home), 3)),
        "confusion_matrix": cm_home,
        "roc_curve": {"fpr": fpr_h.round(3).tolist(), "tpr": tpr_h.round(3).tolist()}
    },
    "clinic": {
        "best_model_name": clinic_best_name,
        "best_params": {k.replace("model__", ""): v for k, v in clinic_params.items()},
        "threshold": thr_clinic,
        "accuracy": float(round(accuracy_score(y_test, pred_clinic), 3)),
        "recall": float(round(recall_score(y_test, pred_clinic), 3)),
        "precision": float(round(precision_score(y_test, pred_clinic), 3)),
        "f1": float(round(f1_score(y_test, pred_clinic), 3)),
        "roc_auc": float(round(roc_auc_score(y_test, proba_clinic), 3)),
        "confusion_matrix": cm_clinic,
        "roc_curve": {"fpr": fpr_c.round(3).tolist(), "tpr": tpr_c.round(3).tolist()}
    }
}

# Permutation Importances
imp_home_res = permutation_importance(home_model, Xh_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE)
imp_home = dict(zip(Xh_test.columns, imp_home_res.importances_mean.round(4)))

imp_clinic_res = permutation_importance(clinic_model, Xc_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE)
imp_clinic = dict(zip(Xc_test.columns, imp_clinic_res.importances_mean.round(4)))

# Fairness Checks
sex_test = data.loc[idx_test, "sex"]
hosp_test = data.loc[idx_test, "dataset"]

def calc_group_metrics(proba, thr, grp_series):
    res = []
    for g in grp_series.dropna().unique():
        mask = (grp_series == g).values
        yt = y_test.values[mask]
        pr = proba[mask]
        r = {
            "group": str(g),
            "patients": int(mask.sum()),
            "disease_share": float(round(yt.mean(), 3)) if len(yt) > 0 else 0,
            "recall": float(round(recall_score(yt, pr >= thr), 3)) if yt.sum() > 0 else None,
            "roc_auc": float(round(roc_auc_score(yt, pr), 3)) if len(np.unique(yt)) == 2 else None
        }
        res.append(r)
    return res

fairness = {
    "clinic_by_sex": calc_group_metrics(proba_clinic, thr_clinic, sex_test),
    "clinic_by_hosp": calc_group_metrics(proba_clinic, thr_clinic, hosp_test),
    "home_by_sex": calc_group_metrics(proba_home, thr_home, sex_test)
}

# Leave-one-hospital-out
loho_rows = []
for hosp in data["dataset"].dropna().unique():
    train_mask = data["dataset"] != hosp
    test_mask = data["dataset"] == hosp
    m = clone(clinic_model)
    m.fit(X_clinic[train_mask], y[train_mask])
    p = m.predict_proba(X_clinic[test_mask])[:, 1]
    yt = y[test_mask]
    loho_rows.append({
        "hospital": str(hosp),
        "patients": int(test_mask.sum()),
        "roc_auc": float(round(roc_auc_score(yt, p), 3)),
        "recall_at_clinic_thr": float(round(recall_score(yt, p >= thr_clinic), 3))
    })

# Save artifacts
joblib.dump(home_model, "models/home_model.joblib")
joblib.dump(clinic_model, "models/clinic_model.joblib")

config = {
    "home_features": HOME_FEATURES,
    "clinic_features": CLINIC_FEATURES,
    "home_threshold": thr_home,
    "clinic_threshold": thr_clinic,
    "categories": {
        "sex": sorted(data["sex"].dropna().unique().tolist()),
        "cp": sorted(data["cp"].dropna().unique().tolist()),
        "restecg": sorted(data["restecg"].dropna().unique().tolist()),
        "slope": sorted(data["slope"].dropna().unique().tolist()),
        "exang": ["False", "True"],
        "fbs": ["False", "True"],
    },
}

with open("models/config.json", "w") as f:
    json.dump(config, f, indent=2)

full_metadata = {
    "comparison": {
        "home": home_compare_res,
        "clinic": clinic_compare_res
    },
    "evaluation": eval_summary,
    "feature_importance": {
        "home": imp_home,
        "clinic": imp_clinic
    },
    "fairness": fairness,
    "loho": loho_rows,
    "dataset_stats": {
        "total_rows": len(data),
        "positive_count": int((data['target'] == 1).sum()),
        "negative_count": int((data['target'] == 0).sum()),
        "positive_rate": float(round(data['target'].mean(), 3)),
        "hospitals": data['dataset'].value_counts().to_dict(),
        "sex_dist": data['sex'].value_counts().to_dict(),
        "age_min": float(data['age'].min()),
        "age_max": float(data['age'].max()),
        "age_mean": float(round(data['age'].mean(), 1))
    }
}

with open("models/metrics_data.json", "w") as f:
    json.dump(full_metadata, f, indent=2)

print("SUCCESS: Models trained & metrics exported to models/")
