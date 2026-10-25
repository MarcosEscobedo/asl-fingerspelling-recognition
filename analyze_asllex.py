from pathlib import Path

import pandas as pd


DATA_PATH = Path("data") / "signdata.csv"
OUTPUT_PATH = Path("data") / "asllex_top15.csv"


# Cargar ASL-LEX 2.0
data = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print(f"ASL-LEX: {len(data)} signos")
print()


# Seleccionar variables relevantes
analysis = data[
    [
        "EntryID",
        "SignFrequency(M)",
        "LexicalClass",
        "DominantTranslation"
    ]
].copy()


# Eliminar signos sin información de frecuencia
analysis = analysis.dropna(
    subset=["SignFrequency(M)"]
)


# Ordenar por frecuencia
top15 = analysis.sort_values(
    by="SignFrequency(M)",
    ascending=False
).head(15)


print("Top 15 signs by subjective frequency:")
print()

print(
    top15[
        [
            "EntryID",
            "SignFrequency(M)",
            "LexicalClass"
        ]
    ].to_string(index=False)
)


# Guardar resultados
top15.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print()
print(f"Resultados guardados en: {OUTPUT_PATH}")