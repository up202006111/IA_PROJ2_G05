"""
train.py
--------
Treina dois modelos para o franchise BurgerPT:

  Modelo 1 — Regressão: prever receita do próximo mês
  Modelo 2 — Classificação: prever prato mais vendido do próximo mês

Estratégia: para cada restaurante, usar os meses anteriores como features
(lag features) para prever o mês seguinte.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report
)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

# Carregar dados
df = pd.read_csv("data/franchise_monthly.csv")
df = df.sort_values(["restaurant_id", "year", "month"]).reset_index(drop=True)

print(f"Dataset: {len(df)} registos, {df['restaurant_id'].nunique()} restaurantes\n")

# Feature engineering: lag features
def build_features(df):
    """
    Para cada linha (restaurante × mês), cria:
      - revenue_lag1, revenue_lag2, revenue_lag3  (receitas meses anteriores)
      - units_lag1                                (unidades mês anterior)
      - revenue_rolling3                          (média móvel 3 meses)
      - month, zone, city, seats, open_year       (características fixas)
      - marketing_spend, avg_rating, delivery_pct (operacionais do mês actual)
    Target: revenue e top_dish do mês SEGUINTE
    """
    rows = []
    for rid, grp in df.groupby("restaurant_id"):
        grp = grp.sort_values(["year", "month"]).reset_index(drop=True)
        for i in range(3, len(grp) - 1):  # precisa de 3 lags + 1 futuro
            cur  = grp.iloc[i]
            nxt  = grp.iloc[i + 1]
            lag1 = grp.iloc[i - 1]
            lag2 = grp.iloc[i - 2]
            lag3 = grp.iloc[i - 3]

            rows.append({
                # Identidade
                "restaurant_id":   rid,
                "restaurant_name": cur["restaurant_name"],
                "date_predict":    nxt["date"],
                "year_predict":    nxt["year"],
                "month_predict":   nxt["month"],

                # Lag features
                "revenue_lag1":     lag1["revenue"],
                "revenue_lag2":     lag2["revenue"],
                "revenue_lag3":     lag3["revenue"],
                "units_lag1":       lag1["total_units_sold"],
                "revenue_rolling3": (lag1["revenue"] + lag2["revenue"] + lag3["revenue"]) / 3,

                # Features do mês corrente (disponíveis ao fazer previsão)
                "month":            cur["month"],
                "year":             cur["year"],
                "marketing_spend":  cur["marketing_spend"],
                "avg_rating":       cur["avg_rating"],
                "delivery_pct":     cur["delivery_pct"],
                "staff_cost_pct":   cur["staff_cost_pct"],
                "num_reviews":      cur["num_reviews"],

                # Características fixas do restaurante
                "zone":             cur["zone"],
                "city":             cur["city"],
                "seats":            cur["seats"],
                "open_year":        cur["open_year"],

                # Targets
                "target_revenue":   nxt["revenue"],
                "target_top_dish":  nxt["top_dish"],
            })
    return pd.DataFrame(rows)

feat_df = build_features(df)
print(f"Dataset com features: {len(feat_df)} linhas\n")

NUMERIC = [
    "revenue_lag1", "revenue_lag2", "revenue_lag3",
    "units_lag1", "revenue_rolling3",
    "month", "year", "marketing_spend", "avg_rating",
    "delivery_pct", "staff_cost_pct", "num_reviews",
    "seats", "open_year",
]
CATEGORICAL = ["zone", "city"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
], remainder="drop")

X = feat_df[NUMERIC + CATEGORICAL]
y_rev  = feat_df["target_revenue"]
y_dish = feat_df["target_top_dish"]

# Train/test split temporal: últimos 6 meses como teste
test_mask = (feat_df["year_predict"] == 2024) & (feat_df["month_predict"] >= 7)
X_train, X_test = X[~test_mask], X[test_mask]
y_rev_train,  y_rev_test  = y_rev[~test_mask],  y_rev[test_mask]
y_dish_train, y_dish_test = y_dish[~test_mask], y_dish[test_mask]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# MODELO 1 — Regressão de Receita

print("\n" + "="*55)
print("MODELO 1 — Previsão de Receita")
print("="*55)

reg_models = {
    "Ridge":         Pipeline([("pre", preprocessor), ("m", Ridge())]),
    "Random Forest": Pipeline([("pre", preprocessor), ("m", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
    "XGBoost":       Pipeline([("pre", preprocessor), ("m", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0))]),
}

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
reg_results = {}

for name, pipe in reg_models.items():
    scores = cross_val_score(pipe, X_train, y_rev_train, cv=cv_reg,
                             scoring="neg_mean_absolute_error")
    mae_cv = -scores.mean()
    pipe.fit(X_train, y_rev_train)
    preds = pipe.predict(X_test)
    mae_test = mean_absolute_error(y_rev_test, preds)
    r2_test  = r2_score(y_rev_test, preds)
    reg_results[name] = {"mae_cv": mae_cv, "mae_test": mae_test, "r2": r2_test, "preds": preds}
    print(f"[{name}]  CV MAE={mae_cv:,.0f}€  |  Test MAE={mae_test:,.0f}€  |  R²={r2_test:.4f}")

best_reg_name = min(reg_results, key=lambda n: reg_results[n]["mae_test"])
best_reg_pipe = reg_models[best_reg_name]
print(f"\n★ Melhor: {best_reg_name}  (MAE={reg_results[best_reg_name]['mae_test']:,.0f}€, R²={reg_results[best_reg_name]['r2']:.4f})")

# Plot: previsão vs real (melhor modelo) 

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

preds_best = reg_results[best_reg_name]["preds"]
axes[0].scatter(y_rev_test, preds_best, alpha=0.6, color="#2A9D8F", edgecolors="white", s=40)
mn, mx = y_rev_test.min(), y_rev_test.max()
axes[0].plot([mn, mx], [mn, mx], "k--", lw=1)
axes[0].set_xlabel("Receita Real (€)")
axes[0].set_ylabel("Receita Prevista (€)")
axes[0].set_title(f"{best_reg_name}\nPrevisto vs Real", fontweight="bold")

# MAE por modelo
colors = ["#4A90D9", "#6BBF7A", "#E05555"]
names  = list(reg_results.keys())
maes   = [reg_results[n]["mae_test"] for n in names]
bars   = axes[1].bar(names, maes, color=colors, edgecolor="white", width=0.5)
for bar, v in zip(bars, maes):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 50, f"{v:,.0f}€",
                 ha="center", fontsize=10)
axes[1].set_title("MAE no Test Set por Modelo", fontweight="bold")
axes[1].set_ylabel("Erro Absoluto Médio (€)")

plt.tight_layout()
plt.savefig("plots/01_revenue_model.png")
plt.close()
print("✓ plots/01_revenue_model.png")

# MODELO 2 — Classificação de Prato Mais Vendido

print("\n" + "="*55)
print("MODELO 2 — Previsão do Prato Mais Vendido")
print("="*55)
print(f"Classes: {sorted(y_dish.unique())}")
print(f"Distribuição test:\n{y_dish_test.value_counts().to_string()}\n")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_dish_train_enc = le.fit_transform(y_dish_train)
y_dish_test_enc  = le.transform(y_dish_test)

clf_models = {
    "Logistic Regression": Pipeline([("pre", preprocessor), ("m", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))]),
    "Random Forest":       Pipeline([("pre", preprocessor), ("m", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1))]),
    "XGBoost":             Pipeline([("pre", preprocessor), ("m", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0, eval_metric="mlogloss"))]),
}

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_results = {}

for name, pipe in clf_models.items():
    # XGBoost precisa de labels encoded, os outros aceitam strings
    if name == "XGBoost":
        y_tr, y_te = y_dish_train_enc, y_dish_test_enc
    else:
        y_tr, y_te = y_dish_train, y_dish_test

    scores = cross_val_score(pipe, X_train, y_tr, cv=cv_clf, scoring="accuracy")
    pipe.fit(X_train, y_tr)
    raw_preds = pipe.predict(X_test)
    # Converter de volta para nomes se XGBoost
    if name == "XGBoost":
        preds = le.inverse_transform(raw_preds)
    else:
        preds = raw_preds
    acc = accuracy_score(y_dish_test, preds)
    f1  = f1_score(y_dish_test, preds, average="weighted")
    clf_results[name] = {"acc_cv": scores.mean(), "acc_test": acc, "f1": f1, "preds": preds, "le": le if name == "XGBoost" else None}
    print(f"[{name}]  CV Acc={scores.mean():.4f}  |  Test Acc={acc:.4f}  |  F1={f1:.4f}")

best_clf_name = max(clf_results, key=lambda n: clf_results[n]["acc_test"])
best_clf_pipe = clf_models[best_clf_name]
print(f"\n★ Melhor: {best_clf_name}  (Acc={clf_results[best_clf_name]['acc_test']:.4f})")
print("\nClassification Report:")
print(classification_report(y_dish_test, clf_results[best_clf_name]["preds"]))

# Plot: accuracy por modelo + importâncias 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

accs  = [clf_results[n]["acc_test"] for n in clf_models]
bars  = axes[0].bar(list(clf_models.keys()), accs, color=colors, edgecolor="white", width=0.5)
for bar, v in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=10)
axes[0].set_title("Accuracy no Test Set por Modelo", fontweight="bold")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05)

# Feature importance do Random Forest classifier
rf_clf = clf_models["Random Forest"]
rf_model = rf_clf.named_steps["m"]
pre = rf_clf.named_steps["pre"]
cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
feat_names = NUMERIC + cat_names
imp_df = (
    pd.DataFrame({"feature": feat_names, "importance": rf_model.feature_importances_})
    .sort_values("importance", ascending=True).tail(12)
)
axes[1].barh(imp_df["feature"], imp_df["importance"], color="#F4A261", edgecolor="white")
axes[1].set_title("Random Forest — Feature Importance\n(Prato Mais Vendido)", fontweight="bold")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("plots/02_dish_model.png")
plt.close()
print("✓ plots/02_dish_model.png")

# Plot: receita mensal agregada por restaurante 
fig, ax = plt.subplots(figsize=(13, 5))
top5 = df.groupby("restaurant_id")["revenue"].sum().nlargest(5).index
colors5 = ["#2A9D8F", "#E76F51", "#264653", "#E9C46A", "#A8DADC"]
for rid, color in zip(top5, colors5):
    sub = df[df["restaurant_id"] == rid].copy()
    sub["dt"] = pd.to_datetime(sub["date"])
    name = sub["restaurant_name"].iloc[0].replace("BurgerPT ", "")
    ax.plot(sub["dt"], sub["revenue"], label=name, color=color, lw=2)
ax.set_title("Evolução de Receita — Top 5 Restaurantes", fontweight="bold")
ax.set_ylabel("Receita Mensal (€)")
ax.set_xlabel("")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/03_revenue_evolution.png")
plt.close()
print("✓ plots/03_revenue_evolution.png")

# Guardar modelos
with open("models/revenue_model.pkl", "wb") as f:
    pickle.dump({"name": best_reg_name, "pipeline": best_reg_pipe,
                 "numeric": NUMERIC, "categorical": CATEGORICAL}, f)

with open("models/dish_model.pkl", "wb") as f:
    best_clf_info = clf_results[best_clf_name]
    classes = list(le.classes_) if best_clf_name == "XGBoost" else list(best_clf_pipe.classes_)
    pickle.dump({"name": best_clf_name, "pipeline": best_clf_pipe,
                 "numeric": NUMERIC, "categorical": CATEGORICAL,
                 "classes": classes,
                 "label_encoder": best_clf_info["le"]}, f)

print(f"\n✓ models/revenue_model.pkl  ({best_reg_name})")
print(f"✓ models/dish_model.pkl     ({best_clf_name})")

# Tabela resumo final
print("\n── Resumo Regressão ──────────────────────────────────────────")
reg_summary = pd.DataFrame({n: {"MAE (€)": v["mae_test"], "R²": v["r2"]}
                             for n, v in reg_results.items()}).T.round(4)
print(reg_summary.to_string())

print("\n── Resumo Classificação ──────────────────────────────────────")
clf_summary = pd.DataFrame({n: {"Accuracy": v["acc_test"], "F1": v["f1"]}
                             for n, v in clf_results.items()}).T.round(4)
print(clf_summary.to_string())
