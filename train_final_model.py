from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("data") / "asl_static.csv"
MODEL_PATH = Path("models") / "asl_classifier_final.joblib"


# 1. Cargar dataset completo
data = pd.read_csv(DATA_PATH)

print(f"Dataset total: {len(data)} muestras")
print()

print("Muestras por letra:")
print(data["label"].value_counts().sort_index())
print()


# 2. Separar metadata de features
metadata_columns = [
    "label",
    "participant",
    "session",
    "handedness"
]

X = data.drop(columns=metadata_columns)
y = data["label"]


# 3. Entrenar modelo final con TODO el dataset
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

print("Entrenando modelo final con S01 + S02...")
model.fit(X, y)

print("Entrenamiento terminado.")


# 4. Guardar modelo
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)

print()
print(f"Modelo final guardado en:")
print(MODEL_PATH)