"""
src/collection/collect_coral.py

Coleta o corpus textual do C-ORAL-BRASIL I (fala espontânea informal em
português brasileiro), transcrito em formato CHAT anotado com marcas
prosódicas.
Fonte: https://www.c-oral-brasil.org/ (licença CC BY-NC-SA 4.0)

Diferente das outras fontes, não há idade exata disponível — cada falante
tem uma FAIXA etária codificada no header de metadados da sessão (campo 2
de @Participants: A=18-25, B=26-40, C=41-60, D=60+, M=menor de idade,
X=desconhecida — ver "3. Appendix/Specifications/Specifications.pdf").
underage é derivado dessa faixa por falante (M -> True, A/B/C/D -> False);
falantes com faixa X ou sem header são descartados, pois não há como
rotular a classe com confiança. age fica None (só a faixa é conhecida,
não o valor exato) — mesma convenção das fontes escolares (seção 6.4).

Nota: só ~2% dos falantes do corpus têm faixa M (menor) — a fonte serve
principalmente como fala espontânea ADULTA informal, não para engordar a
classe underage (ver auditoria em outputs/results/, se existir).

Os headers de metadados vêm em encoding misto (UTF-16 ou Latin-1,
dependendo do arquivo) — _smart_decode() lida com isso via heurística de
densidade de bytes nulos. Os .txt do corpus textual em si são UTF-8.

Schema de saída (Tabela 1): text, underage, age, source, language.
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/c_oral_brasil")
TEXT_SUBDIR = "2. Textual Corpus"
METADATA_SUBDIR = Path("3. Appendix") / "Metadata"

# Faixa etária por falante -> underage (seção 6.4/6.5: idade ausente é
# aceitável; aqui só a faixa é conhecida, não o valor exato).
AGE_BAND_TO_UNDERAGE = {"M": True, "A": False, "B": False, "C": False, "D": False}

_SPEAKER_LINE = re.compile(r"^\*([A-Z]{2,5}):\s*(.*)$")
_UTT_MARKER = re.compile(r"\[\d+\]")            # numeração de utterance: [12] — ruído, não delimitador
_RETRACE_MARKER = re.compile(r"\[/+\d*\]")      # retracing/repair: [/1] [//] [///]
_FRAGMENT = re.compile(r"&\S+")                  # hesitação/falso início: &he &pr
# Tokens que sobram isolados após a limpeza acima e não são palavras reais:
# quebra prosódica não-terminal (/), marca de interrupção (+), fala ininteligível.
_NOISE_TOKENS = {"/", "+", "xxx", "xxxx", "yyy", "yyyy"}


def _smart_decode(raw: bytes) -> str:
    """Headers do C-ORAL-BRASIL vêm em UTF-16 ou Latin-1, sem padrão fixo."""
    if raw.count(b"\x00") > len(raw) * 0.2:
        return raw.decode("utf-16-le", errors="ignore").replace("\x00", "")
    return raw.decode("latin-1", errors="ignore")


def _parse_header(header_path: Path) -> dict:
    """Extrai {codigo_falante: faixa_etaria} do bloco @Participants do header."""
    text = _smart_decode(header_path.read_bytes())
    m = re.search(r"@Participants:(.*?)@Date", text, re.S)
    if not m:
        return {}
    speakers = {}
    for code, fields in re.findall(r"([A-Z]{2,5}),\s*[^(]+\(([^)]*)\)", m.group(1)):
        parts = [p.strip() for p in fields.split(",")]
        if len(parts) > 1:
            speakers[code] = parts[1].upper()
    return speakers


def _clean_utterance(text: str) -> str:
    """Remove marcações CHAT/prosódicas, mantendo o texto falado."""
    text = _UTT_MARKER.sub(" ", text)
    text = _RETRACE_MARKER.sub(" ", text)
    text = _FRAGMENT.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    tokens = [t for t in text.split() if t.lower() not in _NOISE_TOKENS]
    return " ".join(tokens).strip()


def _extract_utterances(txt_path: Path) -> list[tuple[str, str]]:
    """Retorna [(codigo_falante, texto_limpo), ...] para todas as utterances do arquivo.

    A numeração [n] de utterance só existe em ~15% dos arquivos do corpus
    (os demais têm um único turno por linha, sem numeração) — por isso o
    delimitador usado aqui é a quebra prosódica TERMINAL ("//"), marcação
    universal presente em todos os arquivos, e não o marcador [n].
    """
    results = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        speaker_match = _SPEAKER_LINE.match(line.strip())
        if not speaker_match:
            continue
        code, rest = speaker_match.group(1), speaker_match.group(2)
        for chunk in rest.split("//"):
            clean = _clean_utterance(chunk)
            if clean:
                results.append((code, clean))
    return results


def collect_coral(raw_dir: str = "data/raw/c_oral_brasil",
                   output_path: str = "data/processed/coral_brasil.csv") -> pd.DataFrame:
    """
    Lê todos os .txt de "2. Textual Corpus", casa cada um com seu header em
    "3. Appendix/Metadata", extrai utterances por falante e rotula underage
    pela faixa etária do falante. Gera o CSV intermediário no schema da
    Tabela 1.
    """
    base = Path(raw_dir)
    text_dir = base / TEXT_SUBDIR
    meta_dir = base / METADATA_SUBDIR

    txt_files = sorted(text_dir.glob("*.txt"))
    if not txt_files:
        print(f"⚠️  Nenhum arquivo .txt encontrado em '{text_dir}'")
        return pd.DataFrame()

    print(f"📂 {len(txt_files)} transcrições encontradas em '{text_dir}'")

    records = []
    sem_header = 0
    descartados_faixa_desconhecida = 0

    for txt_file in txt_files:
        header_file = meta_dir / f"{txt_file.stem} header.txt"
        speaker_bands = _parse_header(header_file) if header_file.exists() else {}
        if not speaker_bands:
            sem_header += 1

        for code, text in _extract_utterances(txt_file):
            band = speaker_bands.get(code)
            if band not in AGE_BAND_TO_UNDERAGE:
                # Falante sem metadado de faixa etária ou faixa X
                # (desconhecida) — não dá pra rotular com confiança.
                descartados_faixa_desconhecida += 1
                continue
            records.append({
                "text":      text,
                "underage":  AGE_BAND_TO_UNDERAGE[band],
                "age":       None,
                "source":    "coral_brasil",
                "language":  "pt-BR",
                # group_id por FALANTE dentro da sessão, não por sessão
                # inteira: uma mesma conversa pode reunir falante menor e
                # maior de idade (achado no split de 19/09/2026 — 8 sessões
                # com underage misto causavam vazamento entre partições,
                # já que o split trata cada group_id como uma única
                # classe). Agrupar por (sessão, falante) mantém utterances
                # do mesmo autor juntas sem misturar classes no grupo.
                "group_id":  f"coral_brasil::{txt_file.stem}::{code}",
            })

    if not records:
        print("⚠️  Nenhuma utterance rotulável extraída.")
        return pd.DataFrame()

    # Schema da Tabela 1 + group_id (metadado interno para o split)
    df = pd.DataFrame(records, columns=["text", "underage", "age", "source", "language", "group_id"])

    # Filtro de comprimento conforme seção 6.3: mínimo 3 tokens
    df = df[df["text"].str.split().str.len() >= 3]

    # Remoção de duplicatas conforme seção 6.3
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    n_menores = int(df["underage"].sum())
    print(f"\n{'='*50}")
    print(f"Total de entradas                    : {len(df):,}")
    print(f"Sessões sem header de metadados       : {sem_header:,}")
    print(f"Utterances descartadas (faixa X/sem dado): {descartados_faixa_desconhecida:,}")
    print(f"Menores de idade (faixa M)            : {n_menores:,} ({n_menores/len(df)*100:.1f}%)")
    print(f"{'='*50}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Salvo em '{out}'")

    print("\nAmostra:")
    print(df[["text", "underage"]].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    import time
    _t0 = time.perf_counter()
    collect_coral()
    print(f"\n⏱️  Tempo total de execução: {time.perf_counter() - _t0:.1f}s")
