"""
app.py
------
BurgerPT — Dashboard de Gestão do Franchise
Corre com: streamlit run app.py
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Config
st.set_page_config(
    page_title="BurgerPT Dashboard",
    page_icon="🍔",
    layout="wide",
)

PALETTE = ["#E76F51", "#2A9D8F", "#264653", "#E9C46A", "#A8DADC",
           "#F4A261", "#457B9D", "#1D3557", "#6A994E", "#BC4749"]

DISHES = ["Classic Burger", "Cheese Bacon Burger", "Veggie Burger",
          "Frango Crispy", "BurgerPT Especial", "Double Smash"]

MONTH_NAMES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
               "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# Carregar dados e modelos
@st.cache_data
def load_data():
    df = pd.read_csv("data/franchise_monthly.csv")
    df["date_dt"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_models():
    with open("models/revenue_model.pkl", "rb") as f:
        rev = pickle.load(f)
    with open("models/dish_model.pkl", "rb") as f:
        dish = pickle.load(f)
    return rev, dish

df = load_data()
rev_model, dish_model = load_models()

# Sidebar - navegação
with st.sidebar:
    st.title("BurguerPT")
    st.markdown("---")
    page = st.radio(
        "Navegação",
        ["Dashboard Geral", "Análise por Restaurante", "Previsões"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("BurgerPT Franchise · 18 restaurantes\nDados: Jan 2022 – Dez 2024")

# PÁGINA 1 — Dashboard Geral

if page == "Dashboard Geral":
    st.title("Dashboard Geral do Franchise")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_rev   = df["revenue"].sum()
    avg_monthly = df.groupby("date")["revenue"].sum().mean()
    best_rest   = df.groupby("restaurant_name")["revenue"].sum().idxmax()
    best_dish_overall = df[[f"units_{d.replace(' ','_')}" for d in DISHES]].sum().idxmax().replace("units_","").replace("_"," ")

    col1.metric("Receita Total (2022–2024)", f"{total_rev/1e6:.2f}M€")
    col2.metric("Receita Mensal Média",      f"{avg_monthly:,.0f}€")
    col3.metric("Melhor Restaurante",         best_rest.replace("BurgerPT ", ""))
    col4.metric("Prato Mais Vendido",          best_dish_overall)

    st.divider()

    # Receita total mensal do franchise
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("Receita Total Mensal do Franchise")
        monthly_total = df.groupby("date_dt")["revenue"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.fill_between(monthly_total["date_dt"], monthly_total["revenue"],
                        alpha=0.25, color="#E76F51")
        ax.plot(monthly_total["date_dt"], monthly_total["revenue"],
                color="#E76F51", lw=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k€"))
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close()

    with col_b:
        st.subheader("Receita por Zona")
        zone_rev = df.groupby("zone")["revenue"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        colors_z = ["#264653","#2A9D8F","#E9C46A","#F4A261","#E76F51"]
        ax.barh(zone_rev.index, zone_rev.values / 1000, color=colors_z, edgecolor="white")
        ax.set_xlabel("Receita (k€)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close()

    st.divider()

    # Top 5 restaurantes + Sazonalidade
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Ranking de Restaurantes (Receita Total)")
        rank = (
            df.groupby("restaurant_name")["revenue"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        rank["restaurant_name"] = rank["restaurant_name"].str.replace("BurgerPT ", "")
        rank["revenue_fmt"] = rank["revenue"].apply(lambda x: f"{x/1000:.1f}k€")
        rank.index = rank.index + 1
        rank.columns = ["Restaurante", "Receita Total (€)", "Receita"]
        st.dataframe(rank[["Restaurante", "Receita"]], width='stretch', height=380)

    with col_d:
        st.subheader("Sazonalidade — Receita por Mês")
        seasonal = df.groupby("month")["revenue"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        bar_colors = ["#E76F51" if v == seasonal.max() else "#2A9D8F" for v in seasonal.values]
        ax.bar(range(1, 13), seasonal.values, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k€"))
        ax.set_ylabel("Receita Média (€)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close()

    st.divider()

    # Pratos mais vendidos
    st.subheader("Vendas por Prato — Total Franchise")
    dish_cols = {d: f"units_{d.replace(' ','_')}" for d in DISHES}
    dish_totals = {d: df[col].sum() for d, col in dish_cols.items()}
    dish_df = pd.DataFrame(dish_totals.items(), columns=["Prato","Unidades"]).sort_values("Unidades", ascending=False)

    col_e, col_f = st.columns([1, 2])
    with col_e:
        dish_df["% Total"] = (dish_df["Unidades"] / dish_df["Unidades"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(dish_df[["Prato","Unidades","% Total"]].reset_index(drop=True),
                     width='stretch', hide_index=True)

    with col_f:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(dish_df["Prato"], dish_df["Unidades"],
                color=PALETTE[:len(DISHES)], edgecolor="white")
        ax.set_xlabel("Unidades Vendidas")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close()

# PÁGINA 2 — Análise por Restaurante

elif page == "Análise por Restaurante":
    st.title("Análise por Restaurante")

    restaurant_names = sorted(df["restaurant_name"].unique())
    selected = st.selectbox("Seleciona o restaurante:", restaurant_names)

    rdf = df[df["restaurant_name"] == selected].sort_values("date_dt")

    # KPIs do restaurante
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Receita Total",     f"{rdf['revenue'].sum():,.0f}€")
    col2.metric("Média Mensal",       f"{rdf['revenue'].mean():,.0f}€")
    col3.metric("Rating Médio",        f"{rdf['avg_rating'].mean():.1f}")
    col4.metric("Unidades Vendidas",  f"{rdf['total_units_sold'].sum():,}")

    st.divider()

    # Evolução da receita
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Evolução da Receita Mensal")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.fill_between(rdf["date_dt"], rdf["revenue"], alpha=0.2, color="#2A9D8F")
        ax.plot(rdf["date_dt"], rdf["revenue"], color="#2A9D8F", lw=2, marker="o", markersize=3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, width='stretch')
        plt.close()

    with col_b:
        st.subheader("Custos vs Receita")
        avg_staff = rdf["staff_cost_pct"].mean()
        avg_food  = rdf["food_cost_pct"].mean()
        avg_other = 1 - avg_staff - avg_food
        labels = ["Pessoal", "Ingredientes", "Outros/Lucro"]
        values = [avg_staff, avg_food, max(avg_other, 0)]
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.pie(values, labels=labels, autopct="%1.0f%%",
               colors=["#E76F51", "#E9C46A", "#2A9D8F"],
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        st.pyplot(fig, width='stretch')
        plt.close()

    st.divider()

    # Pratos por mês
    st.subheader("Vendas por Prato ao Longo do Tempo")
    dish_cols = {d: f"units_{d.replace(' ','_')}" for d in DISHES}
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, (dish, col) in enumerate(dish_cols.items()):
        ax.plot(rdf["date_dt"], rdf[col], label=f"{dish}",
                color=PALETTE[i], lw=1.8)
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, width='stretch')
    plt.close()

    st.divider()

    # Comparação com média do franchise
    st.subheader("Comparação com Média do Franchise")
    metrics = {
        "Receita Média (€)":     ("revenue",          "€"),
        "Rating Médio":           ("avg_rating",       ""),
        "% Delivery":             ("delivery_pct",     "%"),
        "% Custo Pessoal":        ("staff_cost_pct",   "%"),
        "% Custo Ingredientes":   ("food_cost_pct",    "%"),
    }
    comp_data = []
    for label, (col, unit) in metrics.items():
        rest_val     = rdf[col].mean()
        franchise_val = df[col].mean()
        delta = rest_val - franchise_val
        comp_data.append({
            "Métrica":     label,
            "Restaurante": f"{rest_val:.2f}{unit}",
            "Franchise":   f"{franchise_val:.2f}{unit}",
            "Diferença":   f"{'▲' if delta > 0 else '▼'} {abs(delta):.2f}{unit}",
        })
    st.dataframe(pd.DataFrame(comp_data), width='stretch', hide_index=True)

    # Onde investir
    st.subheader("Onde Investir")
    tips = []
    if rdf["avg_rating"].mean() < df["avg_rating"].mean():
        tips.append("Rating abaixo da média** — investir em qualidade de serviço e resposta a reviews.")
    if rdf["delivery_pct"].mean() < 0.25:
        tips.append("Baixa percentagem de delivery** — considerar parceria com plataformas de entrega.")
    if rdf["staff_cost_pct"].mean() > 0.35:
        tips.append("Custo de pessoal elevado** — rever escalonamento de turnos.")
    if rdf["food_cost_pct"].mean() > 0.33:
        tips.append("Custo de ingredientes alto** — renegociar fornecedores ou rever desperdício.")
    if rdf["revenue"].mean() < df["revenue"].mean() * 0.9:
        tips.append("Receita abaixo da média do franchise** — aumentar marketing local.")
    if not tips:
        tips.append("Este restaurante está acima da média em todas as métricas. Mantém o bom trabalho!")
    for t in tips:
        st.markdown(f"- {t}")


# PÁGINA 3 — Previsões

elif page == "Previsões":
    st.title("Previsões para o Próximo Mês")
    st.markdown("Seleciona um restaurante e o modelo prevê a **receita** e o **prato mais vendido** do mês seguinte.")

    restaurant_names = sorted(df["restaurant_name"].unique())
    selected = st.selectbox("Seleciona o restaurante:", restaurant_names)

    rdf = df[df["restaurant_name"] == selected].sort_values(["year","month"])
    last3 = rdf.tail(3)

    if len(last3) < 3:
        st.warning("Dados insuficientes para este restaurante.")
    else:
        cur  = rdf.iloc[-1]
        lag1 = rdf.iloc[-1]
        lag2 = rdf.iloc[-2]
        lag3 = rdf.iloc[-3]

        # Determinar mês a prever
        next_month = cur["month"] % 12 + 1
        next_year  = cur["year"] + (1 if cur["month"] == 12 else 0)

        st.info(f"A prever: **{MONTH_NAMES[next_month-1]} {next_year}** com base nos últimos 3 meses de dados.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Ajustar Parâmetros")
            st.caption("Podes ajustar o que esperas para o próximo mês.")
            marketing_spend = st.slider("Marketing Spend (€)", 100, 1500,
                                        int(cur["marketing_spend"]), step=50)
            avg_rating      = st.slider("Rating esperado", 2.5, 5.0,
                                        float(round(cur["avg_rating"], 1)), step=0.1)
            delivery_pct    = st.slider("% Pedidos Delivery", 0.10, 0.60,
                                        float(round(cur["delivery_pct"], 2)), step=0.01)

        with col_right:
            st.subheader("Contexto dos Últimos Meses")
            hist = rdf.tail(4)[["date","revenue","total_units_sold","avg_rating"]].copy()
            hist.columns = ["Mês", "Receita (€)", "Unidades", "Rating"]
            hist["Receita (€)"] = hist["Receita (€)"].apply(lambda x: f"{x:,.0f}€")
            st.dataframe(hist.reset_index(drop=True), width='stretch', hide_index=True)

        st.divider()

        if st.button("Gerar Previsão", type="primary", width='stretch'):

            input_row = {
                "revenue_lag1":     lag1["revenue"],
                "revenue_lag2":     lag2["revenue"],
                "revenue_lag3":     lag3["revenue"],
                "units_lag1":       lag1["total_units_sold"],
                "revenue_rolling3": (lag1["revenue"] + lag2["revenue"] + lag3["revenue"]) / 3,
                "month":            next_month,
                "year":             next_year,
                "marketing_spend":  marketing_spend,
                "avg_rating":       avg_rating,
                "delivery_pct":     delivery_pct,
                "staff_cost_pct":   cur["staff_cost_pct"],
                "num_reviews":      cur["num_reviews"],
                "seats":            cur["seats"],
                "open_year":        cur["open_year"],
                "zone":             cur["zone"],
                "city":             cur["city"],
            }
            input_df = pd.DataFrame([input_row])

            # Previsão de receita
            pred_rev = rev_model["pipeline"].predict(input_df)[0]

            # Previsão do prato
            le = dish_model.get("label_encoder")
            raw_dish = dish_model["pipeline"].predict(input_df)[0]
            if le is not None:
                pred_dish = le.inverse_transform([raw_dish])[0]
            else:
                pred_dish = raw_dish

            # Probabilidades do prato
            try:
                dish_probs = dish_model["pipeline"].predict_proba(input_df)[0]
                if le is not None:
                    classes = list(le.classes_)
                else:
                    classes = list(dish_model["pipeline"].classes_)
            except Exception:
                dish_probs = None
                classes    = []

            # Mostrar resultados
            st.subheader(f"Resultados para {MONTH_NAMES[next_month-1]} {next_year}")
            rc1, rc2 = st.columns(2)

            with rc1:
                delta_rev = pred_rev - cur["revenue"]
                st.metric(
                    "Receita Prevista",
                    f"{pred_rev:,.0f}€",
                    delta=f"{delta_rev:+,.0f}€ vs mês actual",
                    delta_color="normal",
                )

            with rc2:
                st.metric(
                    "Prato Mais Vendido Previsto",
                    f" {pred_dish}",
                )

            # Barra de probabilidades dos pratos
            if dish_probs is not None and len(classes) > 0:
                st.subheader("Probabilidade por Prato")
                prob_df = pd.DataFrame({"Prato": classes, "Probabilidade": dish_probs})
                prob_df = prob_df.sort_values("Probabilidade", ascending=True)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                bar_colors = ["#E76F51" if p == prob_df["Prato"].iloc[-1] else "#2A9D8F"
                              for p in prob_df["Prato"]]
                ax.barh(prob_df["Prato"], prob_df["Probabilidade"] * 100,
                        color=bar_colors, edgecolor="white")
                ax.set_xlabel("Probabilidade (%)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig, width='stretch')
                plt.close()

            # Recomendações baseadas na previsão
            st.subheader("Recomendações")
            recs = []
            if pred_rev < cur["revenue"] * 0.95:
                recs.append(f"Receita prevista em queda — considerar aumentar marketing ou promoções em {MONTH_NAMES[next_month-1]}.")
            if pred_dish == "Veggie Burger":
                recs.append("Mês forte para o Veggie Burger — garantir stock de ingredientes vegetarianos.")
            if pred_dish in ["Cheese Bacon Burger", "Double Smash"]:
                recs.append("Mês forte para hambúrgueres premium — reforçar stock de ingredientes de qualidade.")
            if next_month in [7, 8]:
                recs.append("Época de verão — esperar mais pedidos de delivery e maior fluxo turístico.")
            if next_month == 12:
                recs.append("Dezembro — planear capacidade extra e possível menu especial de Natal.")
            if not recs:
                recs.append("Mês estável previsto. Manter operações normais.")
            for r in recs:
                st.markdown(f"- {r}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("L.EIC IA G05 POC — Dados sintéticos")
