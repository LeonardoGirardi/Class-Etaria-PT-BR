"""
src/collection/collect_essays_5ano.py

Coleta redações de alunos do 5º ano do ensino fundamental.
Fonte: https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays

Conforme metodologia (seção 6.1), a faixa modal do 5º ano é 10-11 anos;
atribuímos age = 10 (limite inferior da faixa modal). Todos os autores
recebem underage = True por natureza da fonte (seção 6.4).

Schema de saída (Tabela 1): text, underage, age, source, language.
"""

import re
from pathlib import Path
import pandas as pd

# Idade modal para 5º ano EF (seção 6.1)
IDADE_5ANO = 10

# Limites por sentença após segmentação (mínimo conforme seção 6.3)
MIN_WORDS = 3
MAX_WORDS = 80

KAGGLE_CACHE = Path.home() / ".cache/kagglehub/datasets/moesiof/portuguese-narrative-essays"


def clean_essay_text(text: str) -> str:
    """Remove marcadores específicos do corpus e normaliza espaços."""
    text = str(text)
    # marcadores [T], [X], [F], etc.
    text = re.sub(r'\[[A-Z]\]', '', text)
    # marcadores de início de linha
    text = re.sub(r'^\s*—\s*', '', text, flags=re.MULTILINE)
    # hífens isolados no meio do texto
    text = re.sub(r'\s—\s', ' ', text)
    # quebras de linha viram espaço
    text = re.sub(r'\n+', ' ', text)
    # normaliza espaços
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_into_sentences(text: str) -> list[str]:
    """Segmenta texto em sentenças por pontuação terminal."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            result.append(s)
    return result


def find_data_files() -> list[Path]:
    if KAGGLE_CACHE.exists():
        files = list(KAGGLE_CACHE.rglob("*.csv")) + list(KAGGLE_CACHE.rglob("*.json"))
        if files:
            return files
    local = Path("data/raw/essays_5ano")
    if local.exists():
        return list(local.rglob("*.csv")) + list(local.rglob("*.json"))
    return []


def collect_essays_5ano(output_path: str = "data/processed/essays_5ano.csv") -> pd.DataFrame:
    files = find_data_files()

    if not files:
        print("⚠️  Nenhum arquivo encontrado.")
        print("    Execute: python -c \"import kagglehub; kagglehub.dataset_download('moesiof/portuguese-narrative-essays')\"")
        return pd.DataFrame()

    print(f"📂 {len(files)} arquivo(s) encontrado(s)")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip") if f.suffix == ".csv" else pd.read_json(f)
            print(f"  ✓ {f.name}: {len(df):,} linhas")
            dfs.append(df)
        except Exception:
            try:
                df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
                dfs.append(df)
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    text_col = _find_column(df, ["essay", "text", "texto", "redacao", "redação",
                                  "narrative", "content", "writing", "story"])
    if text_col is None:
        print(f"⚠️  Coluna de texto não encontrada. Colunas: {list(df.columns)}")
        return pd.DataFrame()

    print(f"\n  Coluna de texto: '{text_col}'")
    print("  Limpando marcadores e segmentando em sentenças...")

    records = []
    for essay_idx, raw_text in enumerate(df[text_col]):
        clean = clean_essay_text(raw_text)
        # group_id identifica a redação de origem (mesmo racional do
        # collect_essay_br.py): evita vazamento entre treino/val/teste.
        group_id = f"essays_5ano::{essay_idx}"
        for sent in segment_into_sentences(clean):
            records.append({
                "text":     sent,
                "underage": True,
                "age":      IDADE_5ANO,
                "source":   "essays_5ano",
                "language": "pt-BR",
                "group_id": group_id,
            })

    # Schema da Tabela 1 + group_id (metadado interno para o split)
    df_final = pd.DataFrame(records, columns=["text", "underage", "age", "source", "language", "group_id"])

    # Remoção de duplicatas conforme seção 6.3
    df_final = df_final.drop_duplicates(subset=["text"]).reset_index(drop=True)

    n_words = df_final["text"].str.split().str.len()

    print(f"\n{'='*50}")
    print(f"Redações originais : {len(df):,}")
    print(f"Sentenças geradas  : {len(df_final):,}")
    if len(df) > 0:
        print(f"Média por redação  : {len(df_final)/len(df):.1f} sentenças")
    print(f"Palavras/sentença  : {n_words.mean():.1f} média | {n_words.min()}-{n_words.max()} range")
    print(f"Idade atribuída    : {IDADE_5ANO} anos (5º ano EF)")
    print(f"{'='*50}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out, index=False)
    print(f"\n✅ Salvo em '{out}'")

    print("\nAmostra:")
    print(df_final[["text", "underage", "age"]].head(5).to_string(index=False))

    return df_final


def _find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if any(k in col.lower() for k in candidates):
            return col
    return None


if __name__ == "__main__":
    import time
    _t0 = time.perf_counter()
    collect_essays_5ano()
    print(f"\n⏱️  Tempo total de execução: {time.perf_counter() - _t0:.1f}s")