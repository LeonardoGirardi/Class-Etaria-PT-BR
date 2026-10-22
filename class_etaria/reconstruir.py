"""
class_etaria/reconstruir.py

Reconstrói o Class-Etaria-PT-BR a partir das fontes originais.

O repositório não distribui nenhum texto. O manifesto
(manifesto/class_etaria_manifesto.csv) lista cada entrada do conjunto pelo
SHA-256 do seu texto, com rótulo, fonte e partição. Este script roda a
coleta de cada fonte sobre os arquivos originais, aplica a mesma limpeza de
ruído usada na construção do conjunto e mantém apenas os segmentos cujo
hash está no manifesto. Os tetos por fonte, a amostragem e a divisão não
precisam ser refeitos: a seleção por hash recupera exatamente as entradas
publicadas, já nas partições originais.

Uso (na raiz do repositório):
    python -m class_etaria.reconstruir            # coleta + reconstrução
    python -m class_etaria.reconstruir --sem-coleta   # só reconstrução, sobre data/processed/

Lê:   data/raw/ (fontes originais, ver README.md) e o manifesto
Gera: data/processed/*.csv (saída da coleta) e
      data/class_etaria/class_etaria_{train,val,test}.csv
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from class_etaria.limpeza import clean_text

MANIFESTO = Path("manifesto/class_etaria_manifesto.csv")
PROCESSADOS = Path("data/processed")
SAIDA = Path("data/class_etaria")

# Arquivo gerado pela coleta → nome da fonte no conjunto publicado
FONTES = {
    "childes_pt.csv": "CHILDES-PT",
    "essays_5ano.csv": "PROPOR 2024",
    "b2w_reviews.csv": "B2W-Reviews01",
    "coral_brasil.csv": "C-ORAL-BRASIL",
    "discurso_gramatica.csv": "Discurso & Gramática",
    "olimpiada.csv": "Olimpíada",
    "peul_censo1980.csv": "PEUL-1980",
    "sp2010.csv": "SP2010",
}


def sha256(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def coletar() -> None:
    """Roda a coleta de cada fonte. Os cortes por amostragem feitos na coleta
    original (B2W-Reviews01 e Discurso & Gramática) são desligados, para que
    todos os segmentos candidatos fiquem disponíveis para a seleção por hash."""
    from class_etaria.coleta import (collect_b2w, collect_childes_pt, collect_coraa_adultos, collect_coral,
                                        collect_dg, collect_essays_5ano, collect_olimpiada,
                                        collect_peul_censo1980)

    collect_b2w.MAX_OF_AGE_ROWS = sys.maxsize
    collect_dg.MAX_ROWS_PER_INFORMANT = sys.maxsize
    etapas = [
        ("CHILDES-PT", collect_childes_pt.collect_childes),
        ("PROPOR 2024", collect_essays_5ano.collect_essays_5ano),
        ("B2W-Reviews01", collect_b2w.collect_b2w),
        ("C-ORAL-BRASIL", collect_coral.collect_coral),
        ("Discurso & Gramática", lambda: collect_dg.collect_dg(granularity="medium")),
        ("Olimpíada", collect_olimpiada.collect_olimpiada),
        ("PEUL-1980", collect_peul_censo1980.collect_peul_censo1980),
        ("SP2010", collect_coraa_adultos.collect_coraa_adultos),
    ]
    for nome, funcao in etapas:
        print(f"\n{'#' * 60}\n# Coleta: {nome}\n{'#' * 60}")
        funcao()


def candidatos() -> pd.DataFrame:
    """Une a saída da coleta e aplica a limpeza de ruído, com o hash de cada texto limpo."""
    partes = []
    for arquivo in FONTES:
        f = PROCESSADOS / arquivo
        if not f.exists():
            print(f"  ✗ {f} não encontrado (fonte não coletada)")
            continue
        df = pd.read_csv(f, usecols=["text"], low_memory=False)
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    df = df[df["text"].notna()]
    df["text"] = df["text"].astype(str).str.strip().map(clean_text)
    df = df[df["text"].str.len() > 0]
    df["id"] = df["text"].map(sha256)
    return df.drop_duplicates(subset=["id"])


def reconstruir() -> pd.DataFrame:
    manifesto = pd.read_csv(MANIFESTO)
    df = manifesto.merge(candidatos(), on="id", how="left")

    print(f"\n{'=' * 60}\nEntradas recuperadas por fonte\n{'=' * 60}")
    resumo = df.groupby("source")["text"].agg(recuperadas="count", esperadas="size")
    resumo["%"] = (resumo["recuperadas"] / resumo["esperadas"] * 100).round(2)
    print(resumo.to_string())
    total = int(df["text"].notna().sum())
    print(f"\nTotal: {total:,} de {len(df):,} ({total / len(df) * 100:.2f}%)")

    SAIDA.mkdir(parents=True, exist_ok=True)
    for particao in ["train", "val", "test"]:
        parte = df[(df["split"] == particao) & df["text"].notna()][["text", "underage", "source"]]
        destino = SAIDA / f"class_etaria_{particao}.csv"
        parte.to_csv(destino, index=False)
        print(f"💾 {destino}  ({len(parte):,} entradas)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-coleta", action="store_true", help="usa a saída de coleta já existente em data/processed/")
    if not ap.parse_args().sem_coleta:
        coletar()
    reconstruir()
