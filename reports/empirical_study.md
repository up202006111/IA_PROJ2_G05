# Estudo Empírico

Este estudo compara três algoritmos para cada tarefa usando validação cruzada no conjunto de treino e avaliação final num teste temporal composto pelos últimos 6 meses de 2024.

## Tarefa 1 - Previsão de Receita

Algoritmos avaliados: Ridge, Random Forest e XGBoost.

| Algoritmo | CV MAE (€) | MAE teste (€) | R² teste |
| --- | --- | --- | --- |
| Ridge | 1584.4318 | 2127.0157 | 0.4724 |
| Random Forest | 1252.8719 | 1729.1176 | 0.6426 |
| XGBoost | 901.6012 | 1176.4941 | 0.8013 |

Melhor algoritmo: **XGBoost**

Hiperparâmetros escolhidos: `{"m__subsample": 0.9, "m__reg_lambda": 5.0, "m__n_estimators": 500, "m__max_depth": 3, "m__learning_rate": 0.08, "m__colsample_bytree": 0.9}`

## Tarefa 2 - Previsão do Prato Mais Vendido

Algoritmos avaliados: Logistic Regression, Random Forest e XGBoost.

| Algoritmo | CV Accuracy | Accuracy teste | F1 teste |
| --- | --- | --- | --- |
| Logistic Regression | 0.5704 | 0.6667 | 0.6167 |
| Random Forest | 0.6490 | 0.6296 | 0.6043 |
| XGBoost | 0.6730 | 0.6481 | 0.6244 |

Melhor algoritmo: **Logistic Regression**

Hiperparâmetros escolhidos: `{"m__class_weight": null, "m__C": 0.1}`

## Conclusão

Para regressão, o melhor resultado foi obtido por **XGBoost**, com MAE de teste de 1,176€ e R² de 0.8013.
Para classificação, o melhor resultado foi obtido por **Logistic Regression**, com accuracy de teste de 0.6667 e F1 ponderado de 0.6167.
