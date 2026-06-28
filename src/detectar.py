import cv2
import joblib
import pandas as pd
from pathlib import Path

from funcoes import segmentar, extrair_caracteristicas

BASE_DIR = Path(__file__).resolve().parent.parent

modelo = joblib.load(
    BASE_DIR / "modelos" / "modelo.pkl"
)

classes = [
    "moeda",
    "lapis"
]

acertos = 0
total = 0

for classe in classes:

    pasta = BASE_DIR / "dataset" / "teste" / classe

    imagens = sorted(pasta.glob("*.jpg"))
    imagens += sorted(pasta.glob("*.jpeg"))
    imagens += sorted(pasta.glob("*.png"))

    for arquivo in imagens:

        img = cv2.imread(str(arquivo))

        if img is None:
            continue

        mascara = segmentar(img)

        carac = extrair_caracteristicas(mascara)

        if carac is None:
            continue

        X = pd.DataFrame(
            [carac],
            columns=modelo.feature_names_in_
        )

        pred = modelo.predict(X)[0]

        ok = pred == classe

        if ok:
            acertos += 1

        total += 1

        print(
            f"{arquivo.name:<20} -> {pred:<8} "
            f"{'(OK)' if ok else '(ERRO)'}"
        )

print()

print("=" * 30)

print(f"Acurácia: {100 * acertos / total:.2f}%")

print("=" * 30)