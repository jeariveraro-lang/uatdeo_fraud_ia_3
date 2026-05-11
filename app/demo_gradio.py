"""
Demo Gradio standalone — Pasarela de pagos con XGBoost.

Uso:
    pip install -r ../requirements.txt
    python demo_gradio.py

Se levanta una URL local + URL pública gradio.live (válida ~72h).

Autoras: Gabriela Cabrera · Jessica Rivera
Curso  : Inteligencia Artificial 2026-1S · UTadeo · Docente: Jorge Romero
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import gradio as gr
from pathlib import Path

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# =============================================================================
# CONFIG
# =============================================================================
FEATURES = [
    "TransactionAmt", "log_amt", "amt_round", "amt_decimals", "small_amount",
    "hour", "day_of_week", "card_tipo", "product_cd", "pais_riesgo",
    "email_risk", "device_type", "device_os_risk", "browser_risk",
    "billing_ship_match", "prev_attempts_24h", "ip_billing_distance",
]

UMBRAL_REVISION = 0.30
UMBRAL_BLOQUEO = 0.70

PAISES_RIESGO = {
    "Colombia": 1, "Estados Unidos": 1, "México": 2, "España": 1,
    "Brasil": 2, "Argentina": 3, "Otro": 5,
}
EMAIL_RISK = {
    "gmail.com": 1, "outlook.com": 1, "yahoo.com": 2, "hotmail.com": 2,
    "icloud.com": 1, "temp-mail.org": 5, "mailinator.com": 5, "otro": 3,
}
DEVICE_OS = {"Windows": 1, "macOS": 1, "iOS": 1, "Linux": 2, "Android": 2, "Otro": 4}
BROWSERS = {"Chrome": 1, "Firefox": 1, "Safari": 1, "Edge": 2, "Otro": 4}
CARDS = {"visa": 0, "mastercard": 1, "amex": 2, "discover": 3}
PRODUCTS = {"W": 0, "C": 1, "H": 2, "R": 3, "S": 4}

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "modelo_final" / "xgboost_optuna_final.json"


# =============================================================================
# CONSTRUCCIÓN DEL DATASET Y ENTRENAMIENTO (si no existe artifact)
# =============================================================================
def construir_dataset(N=200_000, seed=RANDOM_SEED):
    """Dataset sintético calibrado con momentos del IEEE-CIS."""
    rng = np.random.default_rng(seed)
    TransactionDT = np.sort(rng.uniform(0, 6 * 30 * 86400, N))
    isFraud = rng.binomial(1, 0.035, N)
    TransactionAmt = np.where(
        isFraud == 1,
        rng.lognormal(4.7, 1.2, N),
        rng.lognormal(4.0, 0.9, N),
    )
    return pd.DataFrame({
        "TransactionDT": TransactionDT,
        "TransactionAmt": TransactionAmt,
        "log_amt": np.log1p(TransactionAmt),
        "amt_round": (TransactionAmt == np.round(TransactionAmt)).astype(int),
        "amt_decimals": (np.round(TransactionAmt * 100) % 100 == 0).astype(int),
        "small_amount": (TransactionAmt < 10).astype(int),
        "hour": ((TransactionDT / 3600) % 24).astype(int),
        "day_of_week": ((TransactionDT / 86400) % 7).astype(int),
        "card_tipo": rng.integers(0, 4, N),
        "product_cd": rng.integers(0, 5, N),
        "pais_riesgo": np.where(isFraud == 1, rng.choice([3, 4, 5, 5], N),
                                  rng.choice([1, 1, 2, 2, 3], N)),
        "email_risk": np.where(isFraud == 1, rng.choice([2, 3, 4, 5, 5], N),
                                rng.choice([1, 1, 1, 2, 3], N)),
        "device_type": rng.integers(0, 2, N),
        "device_os_risk": np.where(isFraud == 1, rng.choice([2, 3, 3, 4], N),
                                     rng.choice([1, 1, 2, 2], N)),
        "browser_risk": rng.integers(1, 5, N),
        "billing_ship_match": np.where(isFraud == 1, rng.choice([0, 0, 0, 1], N),
                                         rng.choice([0, 1, 1, 1], N)),
        "prev_attempts_24h": np.where(isFraud == 1, rng.poisson(3, N),
                                        rng.poisson(0.4, N)),
        "ip_billing_distance": np.where(isFraud == 1, rng.exponential(4000, N),
                                          rng.exponential(200, N)),
        "isFraud": isFraud,
    }).sort_values("TransactionDT").reset_index(drop=True)


def cargar_o_entrenar_modelo():
    """Carga el modelo serializado si existe; si no, entrena uno rápido."""
    modelo = xgb.XGBClassifier(
        max_depth=7,
        learning_rate=0.082,
        n_estimators=220,
        subsample=0.85,
        colsample_bytree=0.78,
        scale_pos_weight=12.4,
        reg_alpha=0.034,
        reg_lambda=1.27,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    if ARTIFACT.exists():
        print(f"📦 Cargando modelo desde {ARTIFACT}")
        modelo.load_model(str(ARTIFACT))
        return modelo

    print("⚙️ No hay artifact serializado; entrenando modelo rápido...")
    df = construir_dataset(N=100_000)
    N = len(df); n_train = int(N * 0.85)
    X = df[FEATURES].iloc[:n_train]; y = df["isFraud"].iloc[:n_train]
    modelo.fit(X, y, verbose=False)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(str(ARTIFACT))
    print(f"✔ Modelo entrenado y guardado en {ARTIFACT}")
    return modelo


MODELO = cargar_o_entrenar_modelo()


# =============================================================================
# CALLBACK DE INFERENCIA
# =============================================================================
def evaluar_pago(monto, producto_cd, card_tipo, pais, ciudad, billing_match,
                  email_domain, prev_attempts, device_type_str, os_name,
                  browser_name, hora, dia_semana, ip_dist_str):
    feats = {
        "TransactionAmt": float(monto),
        "log_amt": float(np.log1p(monto)),
        "amt_round": int(monto == round(monto)),
        "amt_decimals": int(round(monto * 100) % 100 == 0),
        "small_amount": int(monto < 10),
        "hour": int(hora),
        "day_of_week": int(dia_semana),
        "card_tipo": CARDS.get(card_tipo, 0),
        "product_cd": PRODUCTS.get(producto_cd, 0),
        "pais_riesgo": PAISES_RIESGO.get(pais, 3),
        "email_risk": EMAIL_RISK.get(email_domain, 3),
        "device_type": 1 if device_type_str == "Mobile" else 0,
        "device_os_risk": DEVICE_OS.get(os_name, 2),
        "browser_risk": BROWSERS.get(browser_name, 2),
        "billing_ship_match": int(billing_match),
        "prev_attempts_24h": int(prev_attempts),
        "ip_billing_distance": {
            "Mismo país (billing = IP)": 50,
            "Mismo continente (alta distancia)": 800,
            "Continente distinto (alta distancia)": 5000,
        }.get(ip_dist_str, 100),
    }
    X = pd.DataFrame([feats])[FEATURES]
    proba = float(MODELO.predict_proba(X)[0, 1])

    if proba < UMBRAL_REVISION:
        decision = "✅ APROBADA"; color = "#06A77D"
        msg = "La transacción puede procesarse normalmente."
    elif proba < UMBRAL_BLOQUEO:
        decision = "⚠️ REVISIÓN MANUAL"; color = "#F77F00"
        msg = "Solicitar 3-D Secure / OTP antes de capturar fondos."
    else:
        decision = "🛑 BLOQUEADA"; color = "#E63946"
        msg = "Transacción rechazada. Notificar al titular y a su banco."

    html = f"""
    <div style="background:{color}; color:#FFF; padding:28px; border-radius:14px;
                text-align:center; font-family:system-ui;">
        <div style="font-size:12px; letter-spacing:3px; color:#FFFFFF; font-weight:600;">
            DECISIÓN DEL MODELO XGBOOST
        </div>
        <h1 style="margin:12px 0; font-size:42px; font-weight:800; color:#FFFFFF;">{decision}</h1>
        <div style="font-size:18px; color:#FFFFFF;">
            Probabilidad de fraude:
            <span style="font-size:32px; font-weight:800; padding-left:8px; color:#FFFFFF;">{proba*100:.2f}%</span>
        </div>
        <div style="font-size:14px; color:#FFFFFF; margin-top:14px; font-weight:500;">{msg}</div>
    </div>
    <p style="margin-top:14px; font-size:13px; color:#666;">
        <b>Limitaciones del sistema:</b> el modelo fue entrenado con datos sintéticos calibrados
        con la distribución del IEEE-CIS Fraud Detection (2019). En producción, las decisiones
        deben ser validadas por un analista antifraude humano y el modelo requiere monitoreo
        continuo de drift conceptual.
    </p>
    """
    return html


# =============================================================================
# INTERFAZ GRADIO
# =============================================================================
with gr.Blocks(title="Detección de Fraude — Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💳 Pasarela de Pagos — Detección de Fraude con XGBoost")
    gr.Markdown(
        "Llena el formulario y pulsa **Evaluar** para ver la decisión del modelo en tiempo real "
        "(<200 ms). Inspirado en el dataset IEEE-CIS Fraud Detection.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📦 Pedido")
            monto = gr.Number(label="Monto (USD)", value=125.50)
            producto_cd = gr.Dropdown(list(PRODUCTS.keys()), value="W", label="Tipo de producto")
            card_tipo = gr.Dropdown(list(CARDS.keys()), value="visa", label="Tarjeta")

            gr.Markdown("### 🌎 Localización")
            pais = gr.Dropdown(list(PAISES_RIESGO.keys()), value="Colombia", label="País")
            ciudad = gr.Textbox(value="Bogotá", label="Ciudad")
            ip_dist_str = gr.Dropdown([
                "Mismo país (billing = IP)",
                "Mismo continente (alta distancia)",
                "Continente distinto (alta distancia)",
            ], value="Mismo país (billing = IP)", label="Distancia IP - Billing")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 Cuenta")
            email_domain = gr.Dropdown(list(EMAIL_RISK.keys()), value="gmail.com",
                                        label="Dominio del email")
            billing_match = gr.Checkbox(value=True, label="Billing = Shipping address")
            prev_attempts = gr.Slider(0, 10, value=0, step=1, label="Intentos previos 24h")

            gr.Markdown("### 💻 Dispositivo")
            device_type_str = gr.Radio(["Desktop", "Mobile"], value="Desktop",
                                         label="Tipo de dispositivo")
            os_name = gr.Dropdown(list(DEVICE_OS.keys()), value="Windows",
                                    label="Sistema operativo")
            browser_name = gr.Dropdown(list(BROWSERS.keys()), value="Chrome", label="Navegador")
            hora = gr.Slider(0, 23, value=14, step=1, label="Hora del día")
            dia_semana = gr.Slider(0, 6, value=2, step=1, label="Día (0=lun, 6=dom)")

    btn = gr.Button("Evaluar transacción", variant="primary", size="lg")
    salida = gr.HTML()

    btn.click(evaluar_pago,
               inputs=[monto, producto_cd, card_tipo, pais, ciudad, billing_match,
                       email_domain, prev_attempts, device_type_str, os_name,
                       browser_name, hora, dia_semana, ip_dist_str],
               outputs=salida)

    gr.Markdown("""---
### 🧪 Casos de prueba (requisito Corte 3)

| Caso | Descripción | Resultado esperado |
|---|---|---|
| **1. Correcto** | $125, Colombia/Bogotá, gmail.com, billing=ship, Desktop+Windows+Chrome, 14:00 martes | ✅ APROBADA (~0.04%) |
| **2. Límite** | $450, México/Tijuana, outlook.com, billing≠ship, Mobile+Android, 23:00 sábado | ⚠️ REVISIÓN (~40%) |
| **3. Error/Fraude** | $2500, Otro/Bucharest, temp-mail.org, billing≠ship, Mobile+Android+otro, 03:00 dom, 4 intentos | 🛑 BLOQUEADA (~99%) |
""")

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
