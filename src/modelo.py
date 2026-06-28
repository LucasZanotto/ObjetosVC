import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

csv = BASE_DIR / "modelos" / "caracteristicas.csv"

df = pd.read_csv(csv)

X = df.drop(columns=["classe"])
y = df["classe"]

modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

modelo.fit(X, y)

joblib.dump(
    modelo,
    BASE_DIR / "modelos" / "modelo.pkl"
)

print("=" * 40)
print("Modelo treinado com sucesso!")
print(f"Imagens utilizadas: {len(df)}")
print("=" * 40)