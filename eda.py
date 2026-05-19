"""
eda.py
------
Análise Exploratória de Dados — BurgerPT Franchise
Gera gráficos na pasta plots/ para análise do dataset mensal.

Para correr: python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

PALETTE = ["#E76F51", "#2A9D8F", "#264653", "#E9C46A", "#A8DADC",
           "#F4A261", "#457B9D", "#6A994E", "#BC4749", "#E9C46A"]

DISHES = ["Classic Burger", "Cheese Bacon Burger", "Veggie Burger",
          "Frango Crispy", "BurgerPT Especial", "Double Smash"]

DISH_COLS = [f"units_{d.replace(' ', '_')}" for d in DISHES]

MONTH_NAMES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
               "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# Carregar dados
df = pd.read_csv("data/franchise_monthly.csv")
df["date_dt"] = pd.to_datetime(df["date"])

print("=== VISÃO GERAL DO DATASET ===")
print(f"Total de registos:   {len(df)}")
print(f"Restaurantes:        {df['restaurant_id'].nunique()}")
print(f"Período:             {df['date'].min()} → {df['date'].max()}")
print(f"Receita total:       {df['revenue'].sum():,.0f}€")
print(f"Receita média/mês:   {df['revenue'].mean():,.0f}€")
print(f"Prato mais vendido:  {df['top_dish'].value_counts().idxmax()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n{df[['revenue','total_units_sold','avg_rating','delivery_pct','staff_cost_pct']].describe().T.to_string()}")


# 1. Receita total mensal do franchise
print("\n→ Gráfico 1: Receita mensal total...")
monthly = df.groupby("date_dt")["revenue"].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(monthly["date_dt"], monthly["revenue"], alpha=0.2, color="#E76F51")
ax.plot(monthly["date_dt"], monthly["revenue"], color="#E76F51", lw=2.5, marker="o", markersize=4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k€"))
ax.set_title("Receita Total Mensal do Franchise BurgerPT (2022–2024)", fontweight="bold", fontsize=13)
ax.set_ylabel("Receita (€)")
ax.set_xlabel("")

# Anotar máximo e mínimo
max_idx = monthly["revenue"].idxmax()
min_idx = monthly["revenue"].idxmin()
ax.annotate(f"Máx: {monthly['revenue'].max()/1000:.0f}k€",
            xy=(monthly["date_dt"][max_idx], monthly["revenue"][max_idx]),
            xytext=(10, 10), textcoords="offset points", fontsize=9,
            color="#E76F51", fontweight="bold")
ax.annotate(f"Mín: {monthly['revenue'].min()/1000:.0f}k€",
            xy=(monthly["date_dt"][min_idx], monthly["revenue"][min_idx]),
            xytext=(10, -18), textcoords="offset points", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig("plots/01_receita_mensal_total.png")
plt.close()
print("ok - plots/01_receita_mensal_total.png")


# 2. Ranking de restaurantes por receita total
print("→ Gráfico 2: Ranking de restaurantes...")
rank = df.groupby("restaurant_name")["revenue"].sum().sort_values(ascending=True)
rank.index = rank.index.str.replace("BurgerPT ", "")
mean_rev = rank.mean()

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#E76F51" if v >= mean_rev else "#2A9D8F" for v in rank.values]
bars = ax.barh(rank.index, rank.values / 1000, color=colors, edgecolor="white", height=0.7)
ax.axvline(mean_rev / 1000, color="gray", linestyle="--", lw=1.5, label=f"Média: {mean_rev/1000:.0f}k€")
ax.set_xlabel("Receita Total (k€)")
ax.set_title("Ranking de Restaurantes — Receita Total (2022–2024)", fontweight="bold", fontsize=13)
ax.legend()
for bar, v in zip(bars, rank.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{v/1000:.0f}k€", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("plots/02_ranking_restaurantes.png")
plt.close()
print("ok - plots/02_ranking_restaurantes.png")


# 3. Sazonalidade — receita média por mês 
print("→ Gráfico 3: Sazonalidade...")
seasonal = df.groupby("month")["revenue"].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

bar_colors = ["#E76F51" if v == seasonal.max() else
              "#264653" if v == seasonal.min() else "#A8DADC"
              for v in seasonal.values]
bars = axes[0].bar(range(1, 13), seasonal.values, color=bar_colors, edgecolor="white")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(MONTH_NAMES, fontsize=9)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k€"))
axes[0].set_title("Receita Média por Mês (todos os restaurantes)", fontweight="bold")
axes[0].set_ylabel("Receita Média (€)")
for bar, v in zip(bars, seasonal.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 100,
                 f"{v/1000:.1f}k", ha="center", fontsize=8)

# Por zona
zone_seasonal = df.groupby(["month", "zone"])["revenue"].mean().unstack()
for i, (zone, col) in enumerate(zip(zone_seasonal.columns, PALETTE)):
    axes[1].plot(range(1, 13), zone_seasonal[zone], label=zone, color=col, lw=2, marker="o", markersize=4)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(MONTH_NAMES, fontsize=9)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k€"))
axes[1].set_title("Sazonalidade por Zona", fontweight="bold")
axes[1].set_ylabel("Receita Média (€)")
axes[1].legend(fontsize=8)

plt.suptitle("Sazonalidade da Receita", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/03_sazonalidade.png", bbox_inches="tight")
plt.close()
print("ok - plots/03_sazonalidade.png")


# 4. Vendas por prato — total e por mês
print("→ Gráfico 4: Pratos mais vendidos...")
dish_totals = {d: df[col].sum() for d, col in zip(DISHES, DISH_COLS)}
dish_df = pd.DataFrame(dish_totals.items(), columns=["Prato", "Unidades"]).sort_values("Unidades", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Total geral
axes[0].barh(dish_df["Prato"], dish_df["Unidades"],
             color=PALETTE[:len(DISHES)], edgecolor="white", height=0.6)
axes[0].set_xlabel("Unidades Vendidas (total)")
axes[0].set_title("Total de Unidades por Prato (2022–2024)", fontweight="bold")
for i, (_, row) in enumerate(dish_df.iterrows()):
    axes[0].text(row["Unidades"] + 500, i, f"{row['Unidades']:,}", va="center", fontsize=9)

# Por mês (média mensal de unidades)
dish_monthly = df.groupby("month")[DISH_COLS].mean()
dish_monthly.columns = DISHES
for i, (dish, col) in enumerate(zip(DISHES, PALETTE)):
    axes[1].plot(range(1, 13), dish_monthly[dish], label=dish, color=col, lw=2, marker="o", markersize=4)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(MONTH_NAMES, fontsize=9)
axes[1].set_title("Média Mensal de Unidades por Prato", fontweight="bold")
axes[1].set_ylabel("Unidades (média)")
axes[1].legend(fontsize=7.5, ncol=2)

plt.suptitle("Análise de Vendas por Prato", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/04_pratos_vendidos.png", bbox_inches="tight")
plt.close()
print("ok - plots/04_pratos_vendidos.png")


# 5. Distribuição do prato mais vendido por mês 
print("→ Gráfico 5: Top dish por mês...")
top_dish_month = df.groupby(["month", "top_dish"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(12)
for dish, color in zip(top_dish_month.columns, PALETTE):
    if dish in top_dish_month.columns:
        vals = top_dish_month[dish].values
        ax.bar(range(1, 13), vals, bottom=bottom, label=dish, color=color, edgecolor="white")
        bottom += vals

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_NAMES, fontsize=10)
ax.set_title("Prato Mais Vendido por Mês — Contagem de Restaurantes", fontweight="bold", fontsize=13)
ax.set_ylabel("Nº de restaurantes com este top dish")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("plots/05_top_dish_por_mes.png")
plt.close()
print("ok - plots/05_top_dish_por_mes.png")


# 6. Receita por zona 
print("→ Gráfico 6: Receita por zona e cidade...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

zone_rev = df.groupby("zone")["revenue"].sum().sort_values(ascending=True)
axes[0].barh(zone_rev.index, zone_rev.values / 1000,
             color=PALETTE[:len(zone_rev)], edgecolor="white", height=0.5)
axes[0].set_xlabel("Receita Total (k€)")
axes[0].set_title("Receita Total por Zona", fontweight="bold")
for i, v in enumerate(zone_rev.values):
    axes[0].text(v/1000 + 10, i, f"{v/1000:.0f}k€", va="center", fontsize=9)

city_rev = df.groupby("city")["revenue"].sum().sort_values(ascending=True)
axes[1].barh(city_rev.index, city_rev.values / 1000,
             color=PALETTE[:len(city_rev)], edgecolor="white", height=0.5)
axes[1].set_xlabel("Receita Total (k€)")
axes[1].set_title("Receita Total por Cidade", fontweight="bold")
for i, v in enumerate(city_rev.values):
    axes[1].text(v/1000 + 10, i, f"{v/1000:.0f}k€", va="center", fontsize=9)

plt.suptitle("Receita por Zona e Cidade", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/06_receita_zona_cidade.png", bbox_inches="tight")
plt.close()
print("ok - plots/06_receita_zona_cidade.png")


# 7. Métricas operacionais 
print("→ Gráfico 7: Métricas operacionais...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

metrics = [
    ("avg_rating",      "Rating Médio",              "#2A9D8F"),
    ("delivery_pct",    "% Pedidos Delivery",         "#E76F51"),
    ("staff_cost_pct",  "% Custo Pessoal / Receita",  "#264653"),
    ("food_cost_pct",   "% Custo Ingredientes / Receita", "#E9C46A"),
]

for ax, (col, label, color) in zip(axes, metrics):
    monthly_metric = df.groupby("date_dt")[col].mean()
    ax.fill_between(monthly_metric.index, monthly_metric.values, alpha=0.2, color=color)
    ax.plot(monthly_metric.index, monthly_metric.values, color=color, lw=2)
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel(label)
    mean_val = monthly_metric.mean()
    ax.axhline(mean_val, color="gray", linestyle="--", lw=1,
               label=f"Média: {mean_val:.2f}")
    ax.legend(fontsize=8)

plt.suptitle("Evolução das Métricas Operacionais", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/07_metricas_operacionais.png")
plt.close()
print("ok - plots/07_metricas_operacionais.png")


# 8. Correlação entre variáveis numéricas 
print("→ Gráfico 8: Matriz de correlação...")
num_cols = [
    "revenue", "total_units_sold", "avg_ticket", "avg_rating",
    "delivery_pct", "staff_cost_pct", "food_cost_pct",
    "marketing_spend", "num_reviews", "seats",
]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Matriz de Correlação — Variáveis Numéricas", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("plots/08_correlacao.png")
plt.close()
print("ok - plots/08_correlacao.png")


# 9. Top 5 restaurantes — evolução da receita
print("→ Gráfico 9: Evolução top 5 restaurantes...")
top5_ids = df.groupby("restaurant_id")["revenue"].sum().nlargest(5).index
top5_df  = df[df["restaurant_id"].isin(top5_ids)]

fig, ax = plt.subplots(figsize=(12, 5))
for rid, color in zip(top5_ids, PALETTE):
    sub  = top5_df[top5_df["restaurant_id"] == rid].sort_values("date_dt")
    name = sub["restaurant_name"].iloc[0].replace("BurgerPT ", "")
    ax.plot(sub["date_dt"], sub["revenue"], label=name, color=color, lw=2)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
ax.set_title("Evolução da Receita — Top 5 Restaurantes", fontweight="bold", fontsize=13)
ax.set_ylabel("Receita Mensal (€)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/09_top5_evolucao.png")
plt.close()
print("ok - plots/09_top5_evolucao.png")


# 10. Crescimento anual por restaurante
print("→ Gráfico 10: Crescimento anual...")
annual = df.groupby(["restaurant_name", "year"])["revenue"].sum().unstack(fill_value=0)
annual.index = annual.index.str.replace("BurgerPT ", "")

# Crescimento 2023 vs 2022 (só restaurantes com os 2 anos completos)
if 2022 in annual.columns and 2023 in annual.columns:
    growth = ((annual[2023] - annual[2022]) / annual[2022] * 100).dropna()
    growth = growth[annual[2022] > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_g = ["#E76F51" if v < 0 else "#2A9D8F" for v in growth.values]
    ax.barh(growth.index, growth.values, color=colors_g, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Crescimento (%)")
    ax.set_title("Crescimento de Receita 2022 → 2023 por Restaurante", fontweight="bold", fontsize=13)
    for i, v in enumerate(growth.values):
        ax.text(v + 0.3 if v >= 0 else v - 0.3, i,
                f"{v:.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig("plots/10_crescimento_anual.png")
    plt.close()
    print("ok - plots/10_crescimento_anual.png")

print("\nok - EDA completa — 10 gráficos guardados em plots/")