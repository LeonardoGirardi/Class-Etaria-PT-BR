"""
src/cleaning/clean_text.py

Limpeza centralizada do dataset consolidado, conforme a seção 6.3 do artigo
("Critérios de Filtragem e Limpeza"):

  1. Remoção de ruído textual: marcações residuais do formato CHAT, emojis,
     URLs, pontuação repetida e caracteres não-alfabéticos irrelevantes.
  2. Filtragem por comprimento: contagem REAL de tokens via tokenizador do
     spaCy (exclui espaço e pontuação), reaplicada após a limpeza de ruído
     pois remover ruído pode encurtar uma entrada. Critério único usado do
     início ao fim do pipeline (aqui e em src/modeling/tokenize_for_model.py)
     — antes a limpeza usava contagem ingênua por str.split() e só a etapa
     de modelagem usava contagem real, o que descasava o dataset público
     do que de fato entrava no modelo. Unificado a pedido do orientador
     (18/09/2026): ver count_real_tokens() abaixo.
  3. Filtragem por idioma: descarta entradas detectadas como não-português.
  4. Validação de idade: idade ausente é aceitável (ver seção 6.4/6.5);
     idade negativa ou > 110 anos é descartada.
  5. Remoção de duplicatas (rede de segurança pós-limpeza; os scripts de
     coleta já deduplicam individualmente, mas a limpeza pode fazer dois
     textos originalmente distintos convergirem para o mesmo texto limpo).

Este módulo é aplicado sobre o dataset já consolidado (todas as fontes
juntas), antes da divisão em treino/validação/teste — ver
src/exploration/build_and_explore.py e src/modeling/split_data.py.

Uso:
    from src.cleaning.clean_text import clean_dataframe, print_report
    df_clean, report = clean_dataframe(df, config)
    print_report(report)
"""

from __future__ import annotations

import re
import time

import pandas as pd
import spacy

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42  # torna a detecção determinística
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

_nlp_tokenizer = None  # cache do modelo spaCy — carregado uma única vez sob demanda


def count_real_tokens(texts: pd.Series) -> pd.Series:
    """
    Conta tokens "de verdade" via tokenizador do spaCy (exclui espaço e
    pontuação) — critério único de comprimento usado do início ao fim do
    pipeline: aqui (clean_text.py, sobre "text") e em
    src/modeling/tokenize_for_model.py (sobre "text_model", já sem
    pontuação/emoji/risada — nesse caso excluir pontuação não muda nada,
    já que ela já foi removida antes).

    Usa nlp.tokenizer.pipe (só o tokenizador, sem POS/parser/etc.) —
    rápido, já que aqui só precisamos contar.
    """
    global _nlp_tokenizer
    if _nlp_tokenizer is None:
        _nlp_tokenizer = spacy.load("pt_core_news_sm")

    counts = [
        sum(1 for t in doc if not t.is_space and not t.is_punct)
        for doc in _nlp_tokenizer.tokenizer.pipe(texts.astype(str), batch_size=256)
    ]
    return pd.Series(counts, index=texts.index)


# ---------------------------------------------------------------------------
# Regex de ruído (seção 6.3)
# ---------------------------------------------------------------------------

# Resíduos do formato CHAT que sobrevivem à extração via pylangacq/regex
# (marcadores de repetição/reformulação, sobreposição de fala, pausas e
# placeholders para trechos ininteligíveis).
_CHAT_MARKUP = re.compile(
    r"&[-=]\S*"            # &-uh, &=laughs
    r"|\[/{1,3}\]"          # [/] [//] [///]
    r"|\[<\]|\[>\]"         # sobreposição de fala
    r"|<[^>]{1,40}>"        # <...> marcações de escopo (limite evita greedy em HTML/texto real)
    r"|\(\.{1,3}\)"         # (.) (..) (...) pausas
    r"|\bwww\b|\bxxx\b|\byyy\b"  # placeholders CHAT para ininteligível
)

_URL = re.compile(r"https?://\S+|www\.\S+")

# Faixas Unicode de emojis/pictogramas mais comuns
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+"
)

# Pontuação/símbolos repetidos 3+ vezes → colapsa para uma ocorrência
_REPEATED_PUNCT = re.compile(r"([!?.,;:\-_*#~])\1{2,}")

# Caracteres fora do alfabeto português + pontuação básica são ruído
_NON_ALPHA_NOISE = re.compile(r"[^\w\sÀ-ÿ.,!?;:()\"'\-]", flags=re.UNICODE)

_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove ruído textual de uma entrada conforme os critérios da seção 6.3."""
    text = str(text)
    text = _CHAT_MARKUP.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _NON_ALPHA_NOISE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def is_portuguese(text: str) -> bool | None:
    """
    Detecta se o texto está em português via langdetect.

    Retorna None quando a detecção falha (texto curto/ambíguo demais para
    o detector) — nesse caso o chamador deve manter a entrada por padrão,
    já que a falha do detector não é evidência de que o texto não é
    português (e descartar por padrão introduziria viés contra entradas
    curtas, que já são maioria em algumas fontes, como o CHILDES-PT).
    """
    if not HAS_LANGDETECT:
        return True
    try:
        return detect(text) == "pt"
    except LangDetectException:
        return None
    except Exception:
        # Blindagem extra: qualquer outra falha inesperada do detector não
        # deve derrubar uma execução de dezenas de minutos por causa de uma
        # entrada atípica — trata como ambíguo (mantém por padrão).
        return None


def clean_dataframe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Aplica o pipeline de limpeza (seção 6.3) sobre o dataset consolidado.

    Retorna (dataframe_limpo, relatorio, descartadas_por_idioma):
      - relatorio documenta quantas entradas restam após cada etapa —
        necessário para a seção de resultados do artigo e reprodutibilidade;
      - descartadas_por_idioma preserva as linhas removidas pelo filtro de
        idioma, para auditoria (ex.: checar se o langdetect não está
        descartando português legítimo por engano em textos curtos).
    """
    cleaning_cfg = config.get("cleaning", {})
    min_len = cleaning_cfg.get("min_sentence_length", 3)
    max_len = cleaning_cfg.get("max_sentence_length", 100)
    lang_filter_on = cleaning_cfg.get("language_filter", True)

    report: dict = {"inicial": len(df)}
    tempos: dict = {}
    df = df.copy()
    discarded_lang = df.iloc[0:0].copy()

    # 1. Remoção de ruído textual
    t0 = time.perf_counter()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    report["apos_limpeza_ruido"] = len(df)
    tempos["limpeza_ruido"] = time.perf_counter() - t0

    # 2. Filtro de comprimento — contagem REAL de tokens via spaCy (exclui
    # pontuação), min e max, reaplicado após a limpeza de ruído (que pode
    # encurtar uma entrada). Antes só o mínimo era checado aqui via
    # str.split() ingênuo, e o máximo só era aplicado bem depois, na etapa
    # de modelagem (tokenize_for_model.py) — ver count_real_tokens() acima.
    t0 = time.perf_counter()
    n_tokens = count_real_tokens(df["text"])
    df = df[(n_tokens >= min_len) & (n_tokens <= max_len)]
    report["apos_filtro_comprimento"] = len(df)
    tempos["filtro_comprimento"] = time.perf_counter() - t0

    # 3. Filtragem por idioma (etapa mais lenta: langdetect roda linha a linha)
    #
    # langdetect é pouco confiável em textos muito curtos (poucos tokens
    # não dão sinal estatístico suficiente para a detecção — é um problema
    # documentado da biblioteca, não específico deste dataset). Como as
    # fontes já são curadas como português por construção (CHILDES-PT,
    # Essay-BR, Redações do 5º ano, B2W-Reviews todas coletadas de corpora
    # brasileiros/lusófonos), entradas abaixo do limiar são mantidas sem
    # checagem em vez de arriscar falsos positivos — descartar por engano
    # introduziria viés sistemático contra fontes com texto curto (no caso,
    # sobretudo o CHILDES-PT, cujas utterances infantis são naturalmente
    # curtas: ver seção 6.3, mínimo de 3 tokens).
    t0 = time.perf_counter()
    if lang_filter_on:
        min_tokens_lang = cleaning_cfg.get("language_filter_min_tokens", 6)
        n_tok_now = df["text"].str.split().str.len()
        eligible = n_tok_now >= min_tokens_lang
        report["idioma_nao_verificado_texto_curto"] = int((~eligible).sum())

        print(
            f"  🌐 Detectando idioma de {int(eligible.sum()):,} entradas com "
            f">= {min_tokens_lang} tokens ({int((~eligible).sum()):,} entradas mais "
            f"curtas mantidas sem checagem — langdetect não é confiável abaixo desse limiar)..."
        )
        # dtype=object (não bool): is_portuguese() pode retornar None para
        # casos ambíguos, e uma Series bool estrita não comporta esse valor.
        is_pt = pd.Series(True, index=df.index, dtype=object)
        is_pt.loc[eligible] = df.loc[eligible, "text"].map(is_portuguese)

        # mantém português confirmado (True) e casos ambíguos (None);
        # descarta apenas idiomas confirmadamente diferentes (False)
        mask = is_pt != False  # noqa: E712
        discarded_lang = df[~mask].copy()
        report["descartado_idioma"] = int((~mask).sum())
        report["descartado_idioma_por_fonte"] = (
            discarded_lang.groupby("source").size().to_dict()
        )
        df = df[mask]
    report["apos_filtro_idioma"] = len(df)
    tempos["filtro_idioma"] = time.perf_counter() - t0

    # 4. Validação de idade — ausente é aceitável; negativa/implausível não
    t0 = time.perf_counter()
    age_num = pd.to_numeric(df["age"], errors="coerce")
    invalid_age = age_num.notna() & ((age_num < 0) | (age_num > 110))
    report["descartado_idade_invalida"] = int(invalid_age.sum())
    df = df[~invalid_age]
    report["apos_validacao_idade"] = len(df)
    tempos["validacao_idade"] = time.perf_counter() - t0

    # 5. Deduplicação global (rede de segurança pós-limpeza)
    t0 = time.perf_counter()
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    report["duplicatas_removidas"] = before - len(df)
    report["final"] = len(df)
    tempos["dedup"] = time.perf_counter() - t0

    report["tempos_s"] = tempos
    report["tempo_total_s"] = sum(tempos.values())

    return df, report, discarded_lang


def _format_duration(seconds: float) -> str:
    """Formata segundos como '1m 23s' ou '4.2s', para logs legíveis."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        rest = seconds - minutes * 60
        return f"{minutes}m {rest:04.1f}s"
    return f"{seconds:.1f}s"


def print_report(report: dict) -> None:
    print("\n🧹 RELATÓRIO DE LIMPEZA (seção 6.3)")
    print(f"  Entradas iniciais            : {report['inicial']:,}")
    print(f"  Após remoção de ruído        : {report['apos_limpeza_ruido']:,}")
    print(f"  Após filtro de comprimento   : {report['apos_filtro_comprimento']:,}")
    if "descartado_idioma" in report:
        if "idioma_nao_verificado_texto_curto" in report:
            print(f"  Mantidas sem checar (texto curto): {report['idioma_nao_verificado_texto_curto']:,}")
        print(f"  Descartadas (não-português)  : {report['descartado_idioma']:,}")
        por_fonte = report.get("descartado_idioma_por_fonte", {})
        if por_fonte:
            print("    ↳ por fonte:")
            for fonte, n in sorted(por_fonte.items(), key=lambda x: -x[1]):
                print(f"        {fonte:20s}: {n:,}")
    print(f"  Descartadas (idade inválida) : {report['descartado_idade_invalida']:,}")
    print(f"  Duplicatas removidas         : {report['duplicatas_removidas']:,}")
    print(f"  Total final                  : {report['final']:,}")

    if "tempos_s" in report:
        print("\n⏱️  TEMPO POR ETAPA")
        nomes = {
            "limpeza_ruido": "Remoção de ruído",
            "filtro_comprimento": "Filtro de comprimento",
            "filtro_idioma": "Filtro de idioma",
            "validacao_idade": "Validação de idade",
            "dedup": "Deduplicação",
        }
        for chave, nome in nomes.items():
            s = report["tempos_s"].get(chave, 0)
            print(f"  {nome:25s}: {_format_duration(s)}")
        print(f"  {'Total (limpeza)':25s}: {_format_duration(report['tempo_total_s'])}")

    if not HAS_LANGDETECT:
        print(
            "\n  ⚠️  langdetect não instalado — filtro de idioma foi pulado "
            "(pip install langdetect)."
        )
