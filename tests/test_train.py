import os
import joblib
import pandas as pd
import numpy as np
import importlib.util
import sys

# -----------------------------
# Charger explicitement TON app.py
# -----------------------------
APP_PATH = os.path.abspath("app.py")
spec = importlib.util.spec_from_file_location("my_flask_app", APP_PATH)
my_flask_app = importlib.util.module_from_spec(spec)
sys.modules["my_flask_app"] = my_flask_app
spec.loader.exec_module(my_flask_app)

app = my_flask_app.app


# -----------------------------
# Charger explicitement TON app.py
# -----------------------------
APP_PATH = os.path.abspath("app.py")
spec = importlib.util.spec_from_file_location("my_flask_app", APP_PATH)
my_flask_app = importlib.util.module_from_spec(spec)
sys.modules["my_flask_app"] = my_flask_app
spec.loader.exec_module(my_flask_app)

app = my_flask_app.app
# -----------------------------
# 1. Vérifier que le modèle existe
# -----------------------------
def test_model_file_exists():
    assert os.path.exists("data/churn_model_clean.pkl"), \
        "❌ Le fichier churn_model_clean.pkl est introuvable."

# test

# -----------------------------
# 2. Tester la prédiction sur un sous-ensemble du CSV
# -----------------------------
def test_model_can_predict_subset():
    model = joblib.load("data/churn_model_clean.pkl")

    # Charger le dataset CSV
    assert os.path.exists("data/train_data.csv"), \
        "❌ Le fichier CSV data/train_data.csv est introuvable."

    df = pd.read_csv("data/train_data.csv")

    # Vérifier que les colonnes nécessaires existent
    required_cols = ["Age", "Account_Manager", "Years", "Num_Sites"]
    for col in required_cols:
        assert col in df.columns, f"❌ Colonne manquante dans le CSV : {col}"

    # Prendre un sous-ensemble de 5 lignes
    subset = df[required_cols].head(5)

    # Convertir en numpy
    X = subset.values

    # Prédictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Vérifications
    assert len(predictions) == 5, "❌ Le modèle ne renvoie pas 5 prédictions."
    assert all(p in [0, 1] for p in predictions), "❌ Les prédictions doivent être 0 ou 1."
    assert all(0 <= p <= 1 for p in probabilities), "❌ Probabilités incohérentes."


# -----------------------------
# 3. Tester l'API Flask /predict
# -----------------------------
def test_api_predict_route():
    client = app.test_client()

    payload = {
        "Age": 50,
        "Account_Manager": 0,
        "Years": 2.5,
        "Num_Sites": 3
    }

    response = client.post("/predict", data=payload)
    assert response.status_code == 200, "❌ L'API /predict ne répond pas correctement."

    data = response.get_json()

    assert "prediction" in data, "❌ Le champ 'prediction' est manquant."
    assert "probability" in data, "❌ Le champ 'probability' est manquant."
    assert 0 <= data["probability"] <= 1, "❌ Probabilité incohérente."
