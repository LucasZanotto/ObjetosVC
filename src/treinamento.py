import cv2
import pandas as pd
from pathlib import Path

from funcoes import segmentar, extrair_caracteristicas

BASE_DIR = Path(__file__).resolve().parent.parent

classes = [
    "moeda",
    "lapis"
]

dados = []

for classe in classes:

    pasta = BASE_DIR / "dataset" / "treino" / classe

    imagens = sorted(pasta.glob("*.jpg"))
    imagens += sorted(pasta.glob("*.jpeg"))
    imagens += sorted(pasta.glob("*.png"))

    print(f"{classe}: {len(imagens)} imagens")

    for arquivo in imagens:

        img = cv2.imread(str(arquivo))

        if img is None:
            continue

        mascara = segmentar(img)

        carac = extrair_caracteristicas(mascara)

        if carac is None:
            continue

        carac.append(classe)

        dados.append(carac)

print()
print("Total:", len(dados))

colunas = [

    "area",
    "perimetro",

    "largura",
    "altura",

    "aspect",

    "extent",

    "solidity",

    "circularidade",

    "hu1",
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7",

    "hull_area",

    "num_vertices",

    "defeitos",

    "classe"
]

df = pd.DataFrame(
    dados,
    columns=colunas
)

print(df.head())

saida = BASE_DIR / "modelos"

saida.mkdir(exist_ok=True)

arquivo_csv = saida / "caracteristicas.csv"

df.to_csv(
    arquivo_csv,
    index=False
)

print()
print(arquivo_csv)