from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = Path("data") / "asl_static.csv"
MODEL_PATH = Path("models") / "asl_classifier_s01.joblib"


# 1. Cargar dataset
data = pd.read_csv(DATA_PATH)

print("Dataset total:", len(data))
print()


# 2. Separar por sesión
train_data = data[data["session"] == "S01"].copy()
test_data = data[data["session"] == "S02"].copy()

print("Entrenamiento S01:", len(train_data))
print("Prueba S02:", len(test_data))
print()


# 3. Columnas que NO son características de la mano
metadata_columns = [
    "label",
    "participant",
    "session",
    "handedness"
]


X_train = train_data.drop(columns=metadata_columns)
y_train = train_data["label"]

X_test = test_data.drop(columns=metadata_columns)
y_test = test_data["label"]


# 4. Entrenar
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("Entrenando modelo con S01...")
model.fit(X_train, y_train)

print("Entrenamiento terminado.")
print()


# 5. Evaluar exclusivamente con S02
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy S01 -> S02: {accuracy:.2%}")
print()

print("Classification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=sorted(y_test.unique())
    )
)


# 6. Mostrar errores
errors = test_data[
    y_test.to_numpy() != predictions
].copy()

errors["prediction"] = predictions[
    y_test.to_numpy() != predictions
]

print()
print(f"Errores totales: {len(errors)}")

if len(errors) > 0:
    print()
    print("Confusiones:")
    print(
        errors.groupby(
            ["label", "prediction"]
        ).size().sort_values(
            ascending=False
        )
    )


# 7. Guardar modelo
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)

print()
print(f"Modelo guardado en: {MODEL_PATH}")