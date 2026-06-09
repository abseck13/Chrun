import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Chargement du dataset d'entraînement depuis un fichier CSV
data = pd.read_csv("data/train_data.csv")

# Sélection des variables explicatives (features)
# Ici : Age, Account_Manager, Years, Num_Sites
X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]

# Variable cible (label) : Churn (0 = reste, 1 = quitte)
y = data['Churn']

# Création d'un pipeline scikit-learn
# 1. StandardScaler : normalise les variables numériques
# 2. LogisticRegression : modèle de classification binaire
model = Pipeline([
    ('scaler', StandardScaler()),        # Étape 1 : normalisation des features
    ('logit', LogisticRegression())      # Étape 2 : régression logistique
])

# Entraînement du pipeline sur les données
model.fit(X, y)

# Sauvegarde du modèle entraîné dans un fichier .pkl
joblib.dump(model, 'data/churn_model_clean.pkl')

# Message de confirmation
print("Pipeline entraîné et sauvegardé.")
