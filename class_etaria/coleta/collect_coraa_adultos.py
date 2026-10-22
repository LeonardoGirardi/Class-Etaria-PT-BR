"""
src/collection/collect_coraa_adultos.py

Coleta fala espontânea ADULTA de dois subcorpora do CORAA v1.1 (Candido
Junior et al., 2021; licença CC BY-NC-ND 4.0), para servir de segunda (e
terceira) fonte de fala adulta além do C-ORAL-BRASIL:

  - SP2010 (Projeto SP2010, USP): 60 entrevistas sociolinguísticas
    gravadas em 2012–2013 em São Paulo; informantes em três faixas
    etárias, todas adultas (20–34, 35–59, 60+).
  - NURC-Recife (Projeto NURC): diálogos e entrevistas da norma culta de
    Recife, informantes com 25 anos ou mais (gravações das décadas de
    1970–1990). Só a fala espontânea entra; "Prepared Speech" (elocuções
    formais/aulas) é descartada.

Motivação (ver relatório, Seção 7.1): hoje a célula "fala × maior" do
dataset tem uma fonte só (C-ORAL-BRASIL), e o leave-one-source-out do
C-ORAL fica perto de zero. Uma segunda fonte adulta de fala permite testar
se o modelo reconhece adulto falando sem depender de uma fonte única.

Como todos os informantes das duas fontes são adultos, não é preciso a
idade de cada segmento: underage = False para todas as linhas, age = None
(só a faixa etária é conhecida, mesma convenção do C-ORAL-BRASIL). A fala
dos documentadores/entrevistadores, que o CORAA não separa, também é de
adultos.

group_id (agrupamento para o split sem vazamento):
  - NURC-Recife: código do inquérito, extraído do caminho do áudio
    (ex.: NURC_RE_DID_175).
  - SP2010: o CORAA não guarda o código do falante; os segmentos são
    agrupados em blocos de BLOCO_SP2010 ids consecutivos dentro de cada
    partição do CORAA, como aproximação de "mesma entrevista" (~770
    segmentos por entrevista em média). Limitação conhecida: um bloco pode
    juntar o fim de uma entrevista com o começo de outra.

ENGATILHADO, NÃO ATIVADO: este coletor gera data/processed/*.csv, mas as
fontes não estão em configs/config.yaml nem em build_and_explore.py. O
dataset atual não muda até isso ser decidido.

Licença: CC BY-NC-ND 4.0 — uso de pesquisa não comercial; a cláusula ND
impede redistribuir versões modificadas, o que precisa ser considerado
antes de publicar um dataset derivado que inclua essas fontes.

Uso:
    python src/collection/collect_coraa_adultos.py

Lê: data/raw/coraa/metadata_{train,dev,test}_final.csv
    (baixados de https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1)
Gera: data/processed/sp2010.csv, data/processed/nurc_recife.csv
      Schema (Tabela 1) + group_id: text, underage, age, source, language, group_id
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/coraa")
URL = "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/metadata_{}_final.csv"
PARTICOES = ["train", "dev", "test"]
MIN_WORDS = 3            # mesmo mínimo do resto do projeto (o filtro fino de 3–100 tokens é na limpeza)
BLOCO_SP2010 = 500
_ID = re.compile(r"/(\d+)_sp_\.wav$")
_INQUERITO = re.compile(r"(NURC_RE_[A-Z0-9]+_\d+)")


def carregar() -> pd.DataFrame:
    partes = []
    for p in PARTICOES:
        f = RAW_DIR / f"metadata_{p}_final.csv"
        if not f.exists():
            raise FileNotFoundError(f"'{f}' não encontrado. Baixe de {URL.format(p)}")
        df = pd.read_csv(f)
        df["particao_coraa"] = p
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def padronizar(df: pd.DataFrame, source: str, group_id: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({
        "text": df["text"].astype(str).str.strip(),
        "underage": False,
        "age": None,
        "source": source,
        "language": "pt-BR",
        "group_id": group_id.values,
    })
    out = out[out["text"].str.split().str.len() >= MIN_WORDS]
    return out.drop_duplicates(subset=["text"]).reset_index(drop=True)


def collect_coraa_adultos(out_dir: str = "data/processed") -> dict:
    meta = carregar()
    resultados = {}

    sp = meta[meta["dataset"] == "SP2010"].copy()
    ids = sp["file_path"].str.extract(_ID)[0].astype(int)
    grupo_sp = "sp2010::" + sp["particao_coraa"] + "::" + (ids // BLOCO_SP2010).astype(str)
    resultados["sp2010"] = padronizar(sp, "sp2010", grupo_sp)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for nome, df in resultados.items():
        caminho = Path(out_dir) / f"{nome}.csv"
        df.to_csv(caminho, index=False)
        print(f"💾 {caminho}: {len(df):,} segmentos, {df['group_id'].nunique():,} grupos, "
              f"mediana de {df['text'].str.split().str.len().median():.0f} palavras")
    return resultados


if __name__ == "__main__":
    collect_coraa_adultos()
