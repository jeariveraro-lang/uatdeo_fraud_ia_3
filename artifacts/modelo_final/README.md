# Artifacts del modelo final

El archivo `xgboost_optuna_final.json` se genera automáticamente la **primera vez** que ejecutas:

```bash
python app/demo_gradio.py
```

o cuando corres el notebook `notebooks/03_modelo_final.ipynb`.

Es el modelo XGBoost entrenado con los hiperparámetros tuneados por Optuna:

| Hiperparámetro | Valor |
|---|---|
| max_depth | 7 |
| learning_rate | 0.082 |
| n_estimators | 220 |
| subsample | 0.85 |
| colsample_bytree | 0.78 |
| scale_pos_weight | 12.4 |
| reg_alpha | 0.034 |
| reg_lambda | 1.27 |

Se serializa con `model.save_model(...)` y se carga con `model.load_model(...)`. Formato JSON nativo de XGBoost (~500 KB).
