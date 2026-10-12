import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

FEATURES = ["r1", "r2", "r3", "r4", "log_base", "semestre"]

def periodo_idx(anio, semestre):
    return int(anio) * 2 + (int(semestre) - 1)

def idx_to_periodo(i):
    return int(i) // 2, int(i) % 2 + 1

def features_from_history(hist, idx):
    h = list(hist)
    base = float(np.mean(h[-4:]))
    b = base if base > 0 else 1e-9
    _, sem = idx_to_periodo(idx)
    return {"r1": h[-1]/b, "r2": h[-2]/b, "r3": h[-3]/b, "r4": h[-4]/b,
            "log_base": float(np.log1p(base)), "semestre": sem, "base": base}

def build_training_frame(panel):
    rows = []
    for key, g in panel.groupby("key"):
        g = g.sort_values("idx")
        vals, idxs = g["y"].values, g["idx"].values
        for i in range(4, len(vals)):
            f = features_from_history(vals[:i], idxs[i])
            if f["base"] <= 0:
                continue
            f.update(key=key, idx=int(idxs[i]), y=float(vals[i]), target=float(vals[i])/f["base"])
            rows.append(f)
    return pd.DataFrame(rows)

class Log1pRatio(BaseEstimator, RegressorMixin):
    # Entrena sobre log(1 + razón) para reducir el sesgo hacia arriba de las razones muy altas
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator
    def fit(self, X, y):
        self.model_ = clone(self.base_estimator).fit(X, np.log1p(np.asarray(y, dtype=float)))
        return self
    def predict(self, X):
        return np.expm1(self.model_.predict(X))
    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

def recursive_forecast(model, hist, last_idx, n):
    h = list(hist); out = []
    for k in range(1, n + 1):
        idx = last_idx + k
        f = features_from_history(h, idx)
        x = pd.DataFrame([f])[FEATURES]
        pred = max(0.0, float(model.predict(x)[0]) * f["base"])
        anio, sem = idx_to_periodo(idx)
        out.append({"idx": idx, "anio": anio, "semestre": sem, "pronostico": round(pred, 1)})
        h.append(pred)
    return pd.DataFrame(out)
