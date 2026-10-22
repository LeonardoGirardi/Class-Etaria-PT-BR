"""
manutencao/gerar_manifesto.py

Gera o manifesto do Class-Etaria-PT-BR a partir do conjunto final com texto (uso
dos mantenedores; quem só quer reconstruir o conjunto não precisa dele).

Uso:
    python manutencao/gerar_manifesto.py caminho/conjunto_splits.csv

Lê: CSV com as colunas text, underage, source e split
Gera: manifesto/class_etaria_manifesto.csv (id, underage, source, split)
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

NOMES_PUBLICOS = {"Redações do 5º ano": "PROPOR 2024", "B2W-Reviews": "B2W-Reviews01"}


def main(origem: str) -> None:
    df = pd.read_csv(origem, usecols=["text", "underage", "source", "split"])
    df["id"] = df["text"].astype(str).map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    assert df["id"].is_unique, "textos repetidos no conjunto final"
    df["source"] = df["source"].replace(NOMES_PUBLICOS)
    destino = Path("manifesto/class_etaria_manifesto.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "underage", "source", "split"]].sort_values("id").to_csv(destino, index=False)
    print(f"💾 {destino}: {len(df):,} entradas")
    print(df.groupby(["split", "source"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
