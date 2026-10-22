"""
src/collection/collect_childes_pt.py

Coleta transcrições de fala infantil em português do banco CHILDES,
utilizando a biblioteca pylangacq conforme metodologia (seção 6.2).

Estratégia:
- Idade extraída do cabeçalho @ID via regex (estável entre versões pylangacq).
- Utterances extraídas e limpas via pylangacq (onde ele agrega valor real).
- A API de utterances varia entre versões; funções auxiliares lidam com
  os formatos modernos (objetos Utterance/Token) e antigos (strings/tuplas).

Referência: https://childes.talkbank.org/access/Romance/Portuguese/
Biblioteca:  https://pylangacq.org/

Schema de saída (Tabela 1): text, underage, age, source, language.
"""

import re
from pathlib import Path
import pandas as pd

try:
    import pylangacq
except ImportError:
    raise ImportError(
        "pylangacq é necessária. Instale com: pip install pylangacq"
    )


# CHILDES-PT cobre crianças até ~12 anos (seção 6.1)
MAX_AGE_YEARS = 12

# Só português brasileiro (19/10/2026, v0.3.0): o CHILDES Portuguese mistura
# corpora do Brasil (AlegreLong, AlegreX, Florianopolis) e de Portugal
# (Santos, PereiraFreitas, CCF). O corpus vem do 2º campo do @ID. Os
# corpora brasileiros entram inteiros: o corte de 27 mil da v0.2.0 (que
# amostrava os dois países juntos) saiu. Teste da decisão em
# scripts/testar_sem_essay_ptpt.py.
CORPORA_PTPT = {"Santos", "PereiraFreitas", "CCF"}

# Regex para o cabeçalho @ID do CHAT: ex. "@ID: pt|corpus|CHI|2;06.11|female|..."
# Captura "years;months" antes do opcional ".days"
_AGE_PATTERN = re.compile(r"\|CHI\|(\d+);(\d*)(?:\.\d+)?")


def _extract_age_months_from_cha(filepath: Path) -> int | None:
    """
    Lê a idade da criança a partir do cabeçalho @ID do arquivo CHA.
    Retorna o total em meses, ou None se ausente.

    Mais confiável que pylangacq.ages() entre versões: depende apenas
    do formato do CHAT, não da API da biblioteca.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("@ID:") and "|CHI|" in line:
                    match = _AGE_PATTERN.search(line)
                    if match:
                        years = int(match.group(1))
                        months_str = match.group(2)
                        months = int(months_str) if months_str else 0
                        # Sanitiza erros tipográficos do corpus CHILDES-PT,
                        # como "20;06" que na verdade é "2;06".
                        if years > MAX_AGE_YEARS:
                            years = years // 10
                        return years * 12 + months
    except Exception:
        pass
    return None


def _corpus_from_cha(filepath: Path) -> str | None:
    """Nome do corpus CHILDES (2º campo do @ID), ex.: "@ID: por|Santos|CHI|..." → "Santos"."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("@ID"):
                return line.split("|")[1]
    return None


def _utterance_to_text(utt) -> str:
    """
    Extrai texto limpo de uma utterance, tolerante a APIs antigas e novas
    do pylangacq.

    Modernas (>= 0.17): objeto Utterance com .tokens, cada Token com .word
    Antigas: utterance pode ser string ou tupla
    """
    # API moderna
    if hasattr(utt, "tokens") and utt.tokens:
        words = [
            getattr(t, "word", None) or "" for t in utt.tokens
        ]
        return " ".join(w for w in words if w).strip()
    # API antiga: string direta
    if isinstance(utt, str):
        return utt.strip()
    # API antiga: tupla (texto, ...)
    if isinstance(utt, (list, tuple)):
        for item in utt:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return str(utt).strip()


def _get_chi_utterances(reader):
    """
    Obtém utterances do falante CHI lidando com diferenças entre versões.
    Tenta as assinaturas conhecidas em ordem de preferência.
    """
    for kw in ("participants", "participant", "speakers", "speaker"):
        try:
            return reader.utterances(**{kw: "CHI"})
        except TypeError:
            continue
        except Exception:
            break
    # Último recurso: pega todas e filtra manualmente
    try:
        all_utts = reader.utterances()
    except Exception:
        return []
    chi_utts = []
    for u in all_utts:
        speaker = getattr(u, "participant", None) or getattr(u, "speaker", None)
        if speaker == "CHI":
            chi_utts.append(u)
    return chi_utts


def collect_childes(raw_dir: str = "data/raw/childes_pt",
                    output_path: str = "data/processed/childes_pt.csv") -> pd.DataFrame:
    """
    Lê todos os arquivos .cha no diretório, extrai utterances do falante CHI
    e gera o CSV intermediário no schema da Tabela 1.

    A idade vem em meses no formato CHAT e é convertida para anos por
    divisão inteira conforme seção 6.4 (e.g., 30 meses → 2 anos).
    """
    raw_path = Path(raw_dir)
    cha_files = sorted(raw_path.rglob("*.cha"))

    if not cha_files:
        print(f"⚠️  Nenhum arquivo .cha encontrado em '{raw_dir}'")
        return pd.DataFrame()

    print(f"📂 {len(cha_files)} arquivos .cha encontrados em '{raw_dir}'")

    records = []
    sem_idade = 0
    descartados_idade = 0
    descartados_ptpt = 0

    for cha_file in cha_files:
        if _corpus_from_cha(cha_file) in CORPORA_PTPT:
            descartados_ptpt += 1
            continue

        # Idade extraída direto do @ID (independente da versão do pylangacq)
        age_months = _extract_age_months_from_cha(cha_file)
        if age_months is None or age_months <= 0:
            age_years = None
            sem_idade += 1
        else:
            # Conversão meses → anos por divisão inteira (seção 6.4)
            age_years = age_months // 12

        # Descarta idades implausíveis para CHILDES-PT (seção 6.3)
        if age_years is not None and age_years > MAX_AGE_YEARS:
            descartados_idade += 1
            continue

        # Pylangacq faz o parsing/limpeza das utterances
        try:
            reader = pylangacq.read_chat(str(cha_file))
        except Exception as e:
            print(f"  ✗ {cha_file.name}: erro de parse ({e})")
            continue

        utterances = _get_chi_utterances(reader)

        # group_id identifica a sessão/arquivo de origem: usado no split
        # treino/val/teste para garantir que todas as utterances de uma
        # mesma sessão fiquem no mesmo conjunto (evita vazamento de dados).
        group_id = f"childes_pt::{cha_file.relative_to(raw_path)}"

        n_extracted = 0
        for utt in utterances:
            text = _utterance_to_text(utt)
            if not text:
                continue
            records.append({
                "text":      text,
                "underage":  True,
                "age":       age_years,
                "source":    "childes_pt",
                "language":  "pt-BR",
                "group_id":  group_id,
            })
            n_extracted += 1

        print(f"  ✓ {cha_file.name}: {n_extracted} utterances (idade ~{age_years} anos)")

    if not records:
        print("⚠️  Nenhuma utterance CHI extraída.")
        return pd.DataFrame()

    # Schema da Tabela 1 + group_id (metadado interno para o split)
    df = pd.DataFrame(records, columns=["text", "underage", "age", "source", "language", "group_id"])

    # Filtro de comprimento conforme seção 6.3: mínimo 3 tokens
    df = df[df["text"].str.split().str.len() >= 3]

    # Remoção de duplicatas conforme seção 6.3
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"Total de entradas      : {len(df):,}")
    print(f"Arquivos sem idade     : {sem_idade:,}")
    print(f"Arquivos descartados   : {descartados_idade:,} (idade > {MAX_AGE_YEARS} anos)")
    print(f"Arquivos PT-PT fora    : {descartados_ptpt:,} ({', '.join(sorted(CORPORA_PTPT))})")
    if df["age"].notna().any():
        print(f"Faixa etária           : {int(df['age'].min())}-{int(df['age'].max())} anos")
    print(f"{'='*50}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Salvo em '{out}'")

    print("\nAmostra:")
    print(df[["text", "underage", "age"]].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    import time
    _t0 = time.perf_counter()
    collect_childes()
    print(f"\n⏱️  Tempo total de execução: {time.perf_counter() - _t0:.1f}s")