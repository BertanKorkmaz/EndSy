# train_mito_model.py
import os, glob, joblib, warnings
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------
# 1)  TÜM CSV’LERİ OKU ve birleştir
# -------------------------------------------------
DATA_DIR = "mito_sim_data"        # Output file of the mito_data.py
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"'{DATA_DIR}' içinde CSV bulunamadı!")

df = pd.concat(
    [pd.read_csv(fp, usecols=[
        "damage_rate", "prolif_rate", "energy_req_rate", "is_dys"
    ]) for fp in csv_files],
    ignore_index=True
)

X = df[["damage_rate", "prolif_rate", "energy_req_rate"]]
y = df["is_dys"].astype(int)           # bool → 0/1

# -------------------------------------------------
# 2)  EĞİTİM / TEST AYIR
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------
# 3)  PIPELINE + HİPERPARAMETRE TARAMASI (opsiyonel)
# -------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=500, class_weight="balanced"))
])

# Küçük bir grid – daha fazlasını ekleyebilirsiniz
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2", "elasticnet"],
    "clf__l1_ratio": [0, 0.5, 1]     # sadece elasticnet’te kullanılır
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=0
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"► En iyi hiperparametreler: {search.best_params_}")

# -------------------------------------------------
# 4)  DEĞERLENDİR
# -------------------------------------------------
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- TEST PERFORMANSI ---")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# -------------------------------------------------
# 5)  MODELİ KAYDET
# -------------------------------------------------
MODEL_PATH = "mito.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"\n✔ Model kaydedildi → {MODEL_PATH}")

# -------------------------------------------------
# 6)  HIZLI KULLANIM FONKSİYONU
# -------------------------------------------------
def predict_is_dys(damage, prolif, energy_req, model_path=MODEL_PATH):
    """Tek bir hücrenin parametreleri için 0/1 tahmini döndür."""
    mdl = joblib.load(model_path)
    proba = mdl.predict_proba([[damage, prolif, energy_req]])[0,1]
    return int(proba >= 0.5), proba   # (etiket, olasılık)

# Örnek:
if __name__ == "__main__":
    label, p = predict_is_dys(3.2, 5.8, 0.4)
    print(f"\nTahmin: is_dys={label}  (p={p:.3f})")
