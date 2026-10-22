"""
src/collection/collect_dg.py

Coleta o corpus Discurso & Gramática (D&G-UFF), que reúne entrevistas
sociolinguísticas (fala E escrita do mesmo informante) em 6 arquivos PDF
públicos, um por cidade: Juiz de Fora, Natal, Niterói, Rio de Janeiro
(dois arquivos, A e B) e Rio Grande.
Fonte: https://deg.uff.br/corpus-dg/ (download direto, sem cadastro)

Diferente das outras fontes escolares do projeto (Essay-BR, 5º ano), aqui
a idade vem INDIVIDUAL e EXATA por informante (campo "Idade: N anos" no
cabeçalho de cada entrevista) — não é inferida pela seção/série. Isso
importa porque um informante da seção "8ª série" já apareceu com 18 anos
no material (aluno mais velho na mesma turma/série) — confiar só no
cabeçalho da seção rotularia esse caso errado. Por isso este coletor
EXIGE o campo "Idade:" explícito e descarta o informante se ele não
existir (não infere idade pela seção/nível de ensino).

Cada informante contribui até 2 documentos: um de fala (PARTE ORAL,
mantendo só os turnos do entrevistado "I:", descartando as perguntas do
entrevistador "E:") e um de escrita (PARTE ESCRITA, texto direto do
informante, sem marcação de turno). Os textos são segmentados em frases
menores: por pontuação terminal na escrita, e pelas pausas "..." (marca
de transcrição do corpus, não existe pontuação de frase na fala) no
áudio — mesmos limites de 3-80 palavras usados no resto do projeto.

group_id é por INFORMANTE (não por cidade nem por linha): garante que
fala e escrita da mesma pessoa fiquem no mesmo split treino/val/teste.

Schema de saída (Tabela 1): text, underage, age, source, language.
"""

import re
from pathlib import Path

import pandas as pd
import pypdf

RAW_DIR = Path("data/raw/discurso_gramatica")

# Nome do arquivo -> URL de origem (deg.uff.br, download direto sem login)
CITY_FILES = {
    "juiz_de_fora":     "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/juiz_de_fora.pdf",
    "natal":            "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/natal.pdf",
    "niteroi":          "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/niteroi.pdf",
    "rio_de_janeiro_a": "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/rio_de_janeiro_a.pdf",
    "rio_de_janeiro_b": "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/rio_de_janeiro_b.pdf",
    "rio_grande":       "https://deg.uff.br/wp-content/uploads/sites/330/2020/11/rio_grande.pdf",
}

MIN_WORDS = 3
MAX_WORDS = 80

# Teto de linhas por informante — ver nota em collect_dg(). Mantém
# ~75% dos informantes intactos (75º percentil de linhas/informante na
# granularidade "medium" é ~119) sem deixar nenhum dominar o corpus.
MAX_ROWS_PER_INFORMANT = 100
RANDOM_SEED = 42

_INFORMANTE_RE = re.compile(r"Informante\s+\d+\s*:\s*([^\n]+)")
_SEXO_RE = re.compile(r"Sexo:\s*(masculino|feminino)", re.IGNORECASE)
_IDADE_RE = re.compile(r"Idade:\s*0*(\d{1,3})\s*anos", re.IGNORECASE)

# Cabeçalhos de tarefa/seção que aparecem como linhas soltas no meio da
# transcrição (variam ligeiramente entre cidades: "Narrativa experiencial"
# vs "Narrativa de experiência pessoal", "Descrição" vs "Descrição de
# local") — removidos por não fazerem parte da fala/escrita do informante.
_HEADING_PATTERNS = [
    re.compile(r"^narrativa\s+(de\s+experi[êe]ncia\s+pessoal|experiencial)\s*\.?$", re.IGNORECASE),
    re.compile(r"^narrativa\s+recontada\s*\.?$", re.IGNORECASE),
    re.compile(r"^descri[çc][ãa]o(\s+de\s+local)?\s*\.?$", re.IGNORECASE),
    re.compile(r"^relato\s+de\s+procedimento\s*\.?$", re.IGNORECASE),
    re.compile(r"^relato\s+de\s+opini[ãa]o\s*\.?$", re.IGNORECASE),
    re.compile(r"^parte\s+(oral|escrita)\s*\.?$", re.IGNORECASE),
]


def _is_heading(line: str) -> bool:
    line = line.strip()
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _download_pdfs() -> None:
    import urllib.request
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in CITY_FILES.items():
        out = RAW_DIR / f"{name}.pdf"
        if out.exists():
            continue
        print(f"⬇️  Baixando {name}.pdf...")
        urllib.request.urlretrieve(url, out)


def _pdf_to_text(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() for page in reader.pages)


# Marcador de turno no início da linha: "I:" (informante), "E:"/"E1:"/
# "E2:" (um ou mais entrevistadores — Natal usa dois) ou "OBS:" (nota do
# transcritor, também não é fala do informante).
_TURN_RE = re.compile(r"^(I|E\d?|OBS)\s*:\s*(.*)$")


def _split_by_task(text: str) -> list[str]:
    """
    Divide o texto (oral ou escrito) em blocos por TAREFA (narrativa
    pessoal, narrativa recontada, descrição, relato de procedimento,
    relato de opinião), usando os cabeçalhos de tarefa como fronteira —
    granularidade mais grossa que dividir por pausa "...", preserva o
    contexto completo de cada tarefa numa linha só do dataset (19/09/2026,
    comparação pedida após a fala em D&G sair com acurácia ~50% — teste
    de hipótese: fragmento curto demais vs. confundimento genuíno).
    """
    blocks = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading(line) and not re.match(r"^parte\s+(oral|escrita)", line, re.IGNORECASE):
            if current:
                blocks.append("\n".join(current))
            current = []
            continue
        if re.match(r"^parte\s+(oral|escrita)", line, re.IGNORECASE):
            continue
        current.append(raw_line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _extract_informant_turns(oral_text: str) -> str:
    """
    Reconstrói só a fala do informante (turnos "I:") a partir de um
    trecho da PARTE ORAL, descartando as perguntas do(s) entrevistador(es)
    ("E:"/"E1:"/"E2:") e notas do transcritor ("OBS:"). Linhas sem
    marcador de turno são continuação do último falante identificado
    (convenção do corpus: um turno longo só marca "I:"/"E:" uma vez, o
    resto flui em linhas seguintes).
    """
    buffer = []
    current_speaker = None
    for raw_line in oral_text.splitlines():
        line = raw_line.strip()
        if not line or _is_heading(line):
            continue
        turn_match = _TURN_RE.match(line)
        if turn_match:
            current_speaker = turn_match.group(1)
            line = turn_match.group(2).strip()
        if current_speaker == "I":
            buffer.append(line)
    return " ".join(buffer)


def _strip_leading_tag(text: str) -> str:
    """Rede de segurança: remove prefixo de turno ('I:'/'E:'/'OBS:') que
    ocasionalmente sobra no início de uma frase segmentada (~0,07% dos
    casos — a marcação nem sempre cai exatamente na fronteira usada pela
    segmentação por pausas/pontuação)."""
    match = _TURN_RE.match(text.strip())
    return match.group(2).strip() if match else text


def _clean_dg_text(text: str) -> str:
    """Remove marcações de transcrição do corpus D&G (não são texto real)."""
    text = re.sub(r"::+", "", text)              # alongamento vocálico "da::"
    text = re.sub(r"\(\([^)]*\)\)", " ", text)    # comentário do transcritor "((riso))"
    text = re.sub(r"\([^)]*\)", " ", text)        # ininteligível "(   )" / reconstrução "(uma)"
    text = re.sub(r"\{[^}]*\}", " ", text)        # reconstrução alternativa "{ficou com o dinheiro}"
    text = text.replace("/", " ")                  # autocorreção "que sai/ que foi sair"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_heading_lines(text: str) -> str:
    """Remove linhas de cabeçalho de tarefa (ex.: 'Relato de procedimento')
    que aparecem soltas no meio do texto escrito — sem isso, a primeira
    frase de cada tarefa fica colada ao título da tarefa anterior."""
    kept = [line for line in text.splitlines() if not _is_heading(line.strip())]
    return "\n".join(kept)


def _segment_oral(text: str) -> list[str]:
    """Fala não tem pontuação de frase — usa as pausas '...' do corpus como fronteira."""
    chunks = re.split(r"\.{2,}", text)
    result = []
    for c in chunks:
        c = _strip_leading_tag(_clean_dg_text(c))
        words = c.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            result.append(c)
    return result


def _segment_written(text: str) -> list[str]:
    """Escrita segmentada por pontuação terminal, igual Essay-BR/5º ano."""
    text = _strip_heading_lines(text)
    text = _clean_dg_text(re.sub(r"\.{2,}", ". ", text))
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for s in sentences:
        s = _strip_leading_tag(s.strip())
        words = s.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            result.append(s)
    return result


# Quantos fragmentos de pausa/frase juntar por linha na granularidade
# "medium" (meio-termo entre "fine", fragmento curto demais — ver
# comparação de 19/09/2026 — e "task", contexto completo mas volume
# inviável: só 969 linhas).
PAUSE_GROUP_SIZE = 3


def _segment_oral_medium(text: str) -> list[str]:
    """Agrupa PAUSE_GROUP_SIZE fragmentos de pausa por linha, em vez de um por um."""
    raw_chunks = [_strip_leading_tag(_clean_dg_text(c)) for c in re.split(r"\.{2,}", text)]
    raw_chunks = [c for c in raw_chunks if c]
    result = []
    for i in range(0, len(raw_chunks), PAUSE_GROUP_SIZE):
        merged = " ".join(raw_chunks[i:i + PAUSE_GROUP_SIZE])
        words = merged.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            result.append(merged)
    return result


def _segment_written_medium(text: str) -> list[str]:
    """Agrupa PAUSE_GROUP_SIZE frases por linha, em vez de uma por uma."""
    text = _strip_heading_lines(text)
    text = _clean_dg_text(re.sub(r"\.{2,}", ". ", text))
    sentences = [_strip_leading_tag(s.strip()) for s in re.split(r"(?<=[.!?])\s+", text.strip())]
    sentences = [s for s in sentences if s]
    result = []
    for i in range(0, len(sentences), PAUSE_GROUP_SIZE):
        merged = " ".join(sentences[i:i + PAUSE_GROUP_SIZE])
        words = merged.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            result.append(merged)
    return result


# Teto de palavras pra granularidade "por tarefa" (uma tarefa inteira
# recontada pode passar longe do limite de 80 palavras usado por frase).
MAX_WORDS_TASK = 300


def _segment_oral_by_task(text: str) -> list[str]:
    """Uma linha por TAREFA (não por pausa) — mantém o contexto completo."""
    result = []
    for chunk in _split_by_task(text):
        informante_fala = _extract_informant_turns(chunk)
        cleaned = _strip_leading_tag(_clean_dg_text(re.sub(r"\.{2,}", " ", informante_fala)))
        words = cleaned.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS_TASK:
            result.append(cleaned)
    return result


def _segment_written_by_task(text: str) -> list[str]:
    """Uma linha por TAREFA (não por frase) — mantém o contexto completo."""
    result = []
    for chunk in _split_by_task(text):
        cleaned = _strip_leading_tag(_clean_dg_text(re.sub(r"\.{2,}", " ", chunk)))
        words = cleaned.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS_TASK:
            result.append(cleaned)
    return result


def _parse_city_file(path: Path, city: str, granularity: str = "fine") -> list[dict]:
    text = _pdf_to_text(path)
    markers = list(_INFORMANTE_RE.finditer(text))

    records = []
    sem_idade = 0
    for i, marker in enumerate(markers):
        start = marker.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        block = text[start:end]

        idade_match = _IDADE_RE.search(block)
        if not idade_match:
            # Sem idade individual explícita — não dá pra confiar no
            # cabeçalho da seção (ver docstring do módulo), descarta.
            sem_idade += 1
            continue
        age = int(idade_match.group(1))
        if not (0 < age <= 110):
            continue
        underage = age < 18

        group_id = f"discurso_gramatica::{city}::informante_{i}"

        oral_match = re.search(r"PARTE ORAL(.*?)(?:PARTE ESCRITA|\Z)", block, re.S)
        escrita_match = re.search(r"PARTE ESCRITA(.*)", block, re.S)

        if oral_match:
            if granularity == "task":
                oral_sentences = _segment_oral_by_task(oral_match.group(1))
            elif granularity == "medium":
                informante_fala = _extract_informant_turns(oral_match.group(1))
                oral_sentences = _segment_oral_medium(informante_fala)
            else:
                informante_fala = _extract_informant_turns(oral_match.group(1))
                oral_sentences = _segment_oral(informante_fala)
            for sentence in oral_sentences:
                records.append({
                    "text": sentence, "underage": underage, "age": age,
                    "source": "discurso_gramatica", "language": "pt-BR",
                    "group_id": group_id,
                })

        if escrita_match:
            if granularity == "task":
                written_sentences = _segment_written_by_task(escrita_match.group(1))
            elif granularity == "medium":
                written_sentences = _segment_written_medium(escrita_match.group(1))
            else:
                written_sentences = _segment_written(escrita_match.group(1))
            for sentence in written_sentences:
                records.append({
                    "text": sentence, "underage": underage, "age": age,
                    "source": "discurso_gramatica", "language": "pt-BR",
                    "group_id": group_id,
                })

    print(f"  ✓ {city:16s}: {len(markers):>3} informantes | {sem_idade:>2} sem idade (descartados)")
    return records


def collect_dg(output_path: str = "data/processed/discurso_gramatica.csv",
                granularity: str = "fine") -> pd.DataFrame:
    """
    Baixa (se necessário) e processa os 6 arquivos do corpus D&G-UFF.

    granularity:
      "fine" (padrão) — fala dividida por pausa "..." do transcritor,
        escrita por frase; fragmentos curtos (~7 palavras em média).
      "task" — uma linha por TAREFA inteira (narrativa, descrição, relato
        de procedimento/opinião), fala e escrita; fragmentos bem maiores,
        preserva contexto completo. Ver nota de 19/09/2026 no docstring
        de _split_by_task: comparação pedida depois que "fine" deu
        acurácia ~50% (pior que aleatório) especificamente nesta fonte no
        baseline geral — testar se é sinal fraco genuíno (evidência de
        confundimento gênero×classe) ou fragmento curto demais
        (artefato da minha segmentação).
    """
    _download_pdfs()

    print(f"📂 Processando corpus Discurso & Gramática ({len(CITY_FILES)} cidades, granularidade={granularity}):")
    all_records = []
    for city in CITY_FILES:
        pdf_path = RAW_DIR / f"{city}.pdf"
        all_records.extend(_parse_city_file(pdf_path, city, granularity=granularity))

    if not all_records:
        print("⚠️  Nenhum registro extraído.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records, columns=["text", "underage", "age", "source", "language", "group_id"])

    # Remoção de duplicatas conforme seção 6.3 (o filtro de comprimento
    # 3-80 palavras já foi aplicado na segmentação, por tarefa/frase)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Teto de linhas por informante (19/09/2026, achado durante comparação
    # de granularidade): tamanho de grupo MUITO desigual entre informantes
    # (de 1 a 1.388 linhas — um único informante de Natal chegava a 9% do
    # corpus inteiro) quebra a proporção 70/15/15 do split específica
    # dessa fonte (algoritmo de alocação é por grupo, e um grupo gigante
    # dominava sozinho o teste ou a validação dependendo de quando caía no
    # sorteio). Cap também reduz overrepresentação do idioleto de um único
    # informante mais falante no vocabulário da fonte.
    group_sizes = df.groupby("group_id").size()
    grupos_grandes = int((group_sizes > MAX_ROWS_PER_INFORMANT).sum())
    if grupos_grandes:
        # NÃO usar groupby(...).apply(...) aqui: em pandas recentes, quando
        # a função aplicada retorna o grupo com as mesmas colunas, a coluna
        # de agrupamento ("group_id") acaba excluída do resultado — bug
        # descoberto em 19/09/2026 (sumia o group_id inteiro, quebrando o
        # split treino/val/teste pra fonte inteira). Loop manual + concat
        # evita esse comportamento.
        parts = []
        for _, g in df.groupby("group_id"):
            parts.append(g if len(g) <= MAX_ROWS_PER_INFORMANT
                         else g.sample(n=MAX_ROWS_PER_INFORMANT, random_state=RANDOM_SEED))
        capped = pd.concat(parts, ignore_index=True)
        print(f"\n✂️  Teto de {MAX_ROWS_PER_INFORMANT} linhas/informante: "
              f"{grupos_grandes} grupo(s) reduzidos, {len(df):,} → {len(capped):,} linhas")
        df = capped

    n_menores = int(df["underage"].sum())
    print(f"\n{'='*50}")
    print(f"Total de entradas     : {len(df):,}")
    print(f"Menores de idade      : {n_menores:,} ({n_menores/len(df)*100:.1f}%)")
    print(f"Faixa etária          : {df['age'].min()}-{df['age'].max()} anos")
    print(f"{'='*50}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Salvo em '{out}'")

    print("\nAmostra:")
    print(df[["text", "underage", "age"]].sample(min(10, len(df)), random_state=42).to_string(index=False))

    return df


if __name__ == "__main__":
    import time
    _t0 = time.perf_counter()
    collect_dg()
    print(f"\n⏱️  Tempo total de execução: {time.perf_counter() - _t0:.1f}s")
