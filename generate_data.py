"""
generate_data.py
----------------
Gera dados sintéticos mensais para o franchise BurgerPT.
18 restaurantes × 36 meses (Jan 2022 – Dez 2024) = 648 linhas base.

Cada linha representa um mês num restaurante e contém:
  - receita total do mês
  - unidades vendidas por prato
  - características do restaurante (localização, tamanho, etc.)
  - contexto temporal (mês, época, etc.)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Restaurantes
RESTAURANTS = {
    1:  {"name": "BurgerPT Lisboa Centro",    "city": "Lisboa",   "zone": "Centro",       "seats": 80,  "open_year": 2018},
    2:  {"name": "BurgerPT Lisboa Belém",     "city": "Lisboa",   "zone": "Turístico",    "seats": 60,  "open_year": 2019},
    3:  {"name": "BurgerPT Lisboa Parque",    "city": "Lisboa",   "zone": "Suburbs",      "seats": 100, "open_year": 2020},
    4:  {"name": "BurgerPT Porto Aliados",    "city": "Porto",    "zone": "Centro",       "seats": 70,  "open_year": 2017},
    5:  {"name": "BurgerPT Porto Foz",        "city": "Porto",    "zone": "Turístico",    "seats": 55,  "open_year": 2020},
    6:  {"name": "BurgerPT Porto Campanhã",   "city": "Porto",    "zone": "Industrial",   "seats": 90,  "open_year": 2021},
    7:  {"name": "BurgerPT Braga Centro",     "city": "Braga",    "zone": "Centro",       "seats": 65,  "open_year": 2019},
    8:  {"name": "BurgerPT Braga Univ.",      "city": "Braga",    "zone": "Universitário","seats": 75,  "open_year": 2020},
    9:  {"name": "BurgerPT Coimbra Centro",   "city": "Coimbra",  "zone": "Centro",       "seats": 60,  "open_year": 2018},
    10: {"name": "BurgerPT Coimbra Univ.",    "city": "Coimbra",  "zone": "Universitário","seats": 80,  "open_year": 2019},
    11: {"name": "BurgerPT Aveiro",           "city": "Aveiro",   "zone": "Centro",       "seats": 55,  "open_year": 2021},
    12: {"name": "BurgerPT Faro",             "city": "Faro",     "zone": "Turístico",    "seats": 65,  "open_year": 2020},
    13: {"name": "BurgerPT Setúbal",          "city": "Setúbal",  "zone": "Suburbs",      "seats": 70,  "open_year": 2022},
    14: {"name": "BurgerPT Funchal",          "city": "Funchal",  "zone": "Turístico",    "seats": 50,  "open_year": 2021},
    15: {"name": "BurgerPT Guimarães",        "city": "Guimarães","zone": "Centro",       "seats": 60,  "open_year": 2022},
    16: {"name": "BurgerPT Viseu",            "city": "Viseu",    "zone": "Centro",       "seats": 55,  "open_year": 2023},
    17: {"name": "BurgerPT Leiria",           "city": "Leiria",   "zone": "Suburbs",      "seats": 65,  "open_year": 2022},
    18: {"name": "BurgerPT Évora",            "city": "Évora",    "zone": "Turístico",    "seats": 45,  "open_year": 2023},
}

# Pratos do menu
DISHES = [
    "Classic Burger",
    "Cheese Bacon Burger",
    "Veggie Burger",
    "Frango Crispy",
    "BurgerPT Especial",
    "Double Smash",
]

# Datas: Jan 2022 – Dez 2024
dates = pd.date_range("2022-01-01", "2024-12-01", freq="MS")

records = []

for rid, rinfo in RESTAURANTS.items():
    # Base de receita mensal depende do tamanho e zona
    zone_multiplier = {
        "Centro": 1.15, "Turístico": 1.25, "Universitário": 0.95,
        "Suburbs": 0.90, "Industrial": 0.80,
    }[rinfo["zone"]]

    base_revenue = rinfo["seats"] * 180 * zone_multiplier  # € por mês base

    # Restaurantes mais antigos têm base mais alta (clientela fidelizada)
    maturity_bonus = min((2022 - rinfo["open_year"]) * 0.04, 0.20)
    base_revenue *= (1 + maturity_bonus)

    for dt in dates:
        # Ignorar meses antes da abertura
        if dt.year < rinfo["open_year"]:
            continue
        if dt.year == rinfo["open_year"] and dt.month < 6:
            continue

        month = dt.month
        year  = dt.year

        # Sazonalidade mensal
        seasonal = {
            1: 0.78, 2: 0.82, 3: 0.90, 4: 0.95,
            5: 1.00, 6: 1.05, 7: 1.10, 8: 1.15,
            9: 1.05, 10: 0.98, 11: 0.88, 12: 1.20,
        }[month]

        # Boost turístico no verão
        if rinfo["zone"] == "Turístico" and month in [7, 8]:
            seasonal *= 1.20
        # Boost universitário em época escolar
        if rinfo["zone"] == "Universitário" and month in [10, 11, 2, 3]:
            seasonal *= 1.10
        if rinfo["zone"] == "Universitário" and month in [7, 8]:
            seasonal *= 0.70

        # Tendência de crescimento anual (+5% ao ano)
        trend = 1 + (year - 2022) * 0.05

        # Receita com ruído
        revenue = base_revenue * seasonal * trend
        revenue *= RNG.uniform(0.93, 1.07)
        revenue = round(revenue, 2)

        # Pratos vendidos
        # Total de unidades estimado pela receita / preço médio (~9.5€)
        total_units = int(revenue / 9.5 * RNG.uniform(0.95, 1.05))

        # Preferências base equilibradas para haver variação no top dish
        dish_weights = {
            "Classic Burger":      0.19,
            "Cheese Bacon Burger": 0.18,
            "Veggie Burger":       0.13,
            "Frango Crispy":       0.20,
            "BurgerPT Especial":   0.16,
            "Double Smash":        0.14,
        }

        # Ajustes sazonais
        if month in [6, 7, 8]:
            dish_weights["Veggie Burger"]  += 0.06
            dish_weights["Frango Crispy"]  += 0.04
            dish_weights["Double Smash"]   -= 0.05
            dish_weights["Classic Burger"] -= 0.03
        if month == 12:
            dish_weights["Cheese Bacon Burger"] += 0.06
            dish_weights["Double Smash"]         += 0.04
            dish_weights["Veggie Burger"]         -= 0.05
        if month in [1, 2]:
            dish_weights["Veggie Burger"]  += 0.05
            dish_weights["Frango Crispy"]  += 0.03
            dish_weights["Double Smash"]   -= 0.04

        if rinfo["zone"] == "Universitário":
            dish_weights["BurgerPT Especial"] -= 0.05
            dish_weights["Classic Burger"]    += 0.03
            dish_weights["Frango Crispy"]     += 0.02
        if rinfo["zone"] == "Turístico":
            dish_weights["BurgerPT Especial"] += 0.05
            dish_weights["Double Smash"]      += 0.03
            dish_weights["Classic Burger"]    -= 0.04

        # Ruído por restaurante/mês
        noise = RNG.uniform(-0.03, 0.03, len(dish_weights))
        for i, key in enumerate(dish_weights):
            dish_weights[key] += noise[i]

        weights = np.array(list(dish_weights.values()))
        weights = np.clip(weights, 0.02, 1)
        weights /= weights.sum()

        units_per_dish = RNG.multinomial(total_units, weights)
        dish_data = {f"units_{d.replace(' ', '_')}": u for d, u in zip(DISHES, units_per_dish)}

        top_dish = DISHES[int(np.argmax(units_per_dish))]

        # Outras métricas operacionais 
        avg_ticket      = round(revenue / max(total_units, 1), 2)
        num_reviews     = int(total_units * RNG.uniform(0.03, 0.08))
        avg_rating      = round(np.clip(RNG.normal(3.9, 0.3), 2.5, 5.0), 1)
        delivery_pct    = round(RNG.uniform(0.15, 0.45), 2)  # % de pedidos delivery
        staff_cost_pct  = round(RNG.uniform(0.28, 0.38), 2)  # % receita em pessoal
        food_cost_pct   = round(RNG.uniform(0.25, 0.35), 2)  # % receita em ingredientes
        marketing_spend = round(RNG.uniform(200, 800) * zone_multiplier, 2)

        records.append({
            "restaurant_id":    rid,
            "restaurant_name":  rinfo["name"],
            "city":             rinfo["city"],
            "zone":             rinfo["zone"],
            "seats":            rinfo["seats"],
            "open_year":        rinfo["open_year"],
            "year":             year,
            "month":            month,
            "date":             dt.strftime("%Y-%m"),
            "revenue":          revenue,
            "total_units_sold": total_units,
            "avg_ticket":       avg_ticket,
            "top_dish":         top_dish,
            "num_reviews":      num_reviews,
            "avg_rating":       avg_rating,
            "delivery_pct":     delivery_pct,
            "staff_cost_pct":   staff_cost_pct,
            "food_cost_pct":    food_cost_pct,
            "marketing_spend":  marketing_spend,
            **dish_data,
        })

df = pd.DataFrame(records)
df.to_csv("data/franchise_monthly.csv", index=False)

print(f"✓ {len(df)} registos gerados ({df['restaurant_id'].nunique()} restaurantes)")
print(f"  Período: {df['date'].min()} → {df['date'].max()}")
print(f"  Receita média mensal: {df['revenue'].mean():,.0f}€")
print(f"  Receita total: {df['revenue'].sum():,.0f}€")
print(f"\nTop dish distribuição:\n{df['top_dish'].value_counts()}")
print(f"\n{df.head(3).T.to_string()}")
