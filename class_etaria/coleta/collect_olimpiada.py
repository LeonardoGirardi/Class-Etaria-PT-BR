"""
src/collection/collect_olimpiada.py

Coleta textos de estudantes finalistas da Olimpíada de Língua Portuguesa
(Programa Escrevendo o Futuro / Cenpec, Itaú Social e MEC), publicados em
coletâneas PDF por edição sob licença Creative Commons Atribuição 4.0.
Fonte: https://www.cenpec.org.br/pesquisa/textos-finalistas-olimpiada-de-lingua-portuguesa/

Por que interessa: é escrita de estudantes de escola pública com a SÉRIE
conhecida pela categoria do concurso, do 5º ano do ensino fundamental ao
3º ano do ensino médio. É a única fonte aberta de escrita de adolescentes
com faixa escolar verificada (o Essay-BR tem idade presumida).

Idade: os PDFs não trazem a idade dos alunos (as idades que aparecem no
texto são dos entrevistados das Memórias literárias, que são adultos). A
faixa vem da categoria, definida pelo regulamento de cada edição:
  2019: Poema 5º ano EF; Memórias literárias 6º–7º EF; Crônica 8º–9º EF;
        Documentário 1º–2º EM; Artigo de opinião 3º EM.
  2010, 2012, 2014, 2016: Poema 5º–6º EF; Memórias literárias 7º–8º EF;
        Crônica 9º EF–1º EM; Artigo de opinião 2º–3º EM ("dois últimos
        anos do Ensino Médio", segundo as apresentações das coletâneas).
  2008: a coletânea junta três livros (Memórias, pp. 3–100; Artigo de
        opinião, pp. 103–140; Poesia, pp. 143 em diante, índice 0 do PDF).
        A apresentação diz que a 1ª edição trabalhou com 4ª–5ª séries (já
        atendidas pelo programa desde 2002), 7ª–8ª séries do EF "ou séries
        equivalentes do ciclo de nove anos" e 2º–3º anos do EM. Como nas
        edições seguintes, Poema = 4ª–5ª série (5º–6º ano), Memórias =
        7ª–8ª série (8º–9º ano), Artigo = 2º–3º EM.
Poema, Memórias e Crônica (~10 a 16 anos) são rotulados como menor.
Artigo de opinião (~16 a 18 anos) também, mas a coluna "categoria" fica no
arquivo processado para permitir tratá-lo à parte: alunos do 3º ano podem
ter 18 anos. Documentário fica de fora (roteiro audiovisual, não texto
corrido).

Edições: as seis coletâneas (2008, 2010, 2012, 2014, 2016, 2019). Cada
uma marca a categoria de um jeito:
  - 2014, 2016, 2019: nome da categoria no cabeçalho da página (nas 3
    primeiras linhas);
  - 2010, 2012: o sumário dá a página inicial de cada categoria e cada
    página começa com seu número impresso (faixas em FAIXAS_PAGINA);
  - 2008: faixas de página do PDF em FAIXAS_PAGINA_2008.
Páginas de apresentação (textos dos organizadores, adultos), listas de
nomes e sumários (linhas "número + título", inclusive quando o PDF escreve
"IÍndice"/"Indice") são descartados.

Limpeza: remove cabeçalhos, números de página, marcas de impressão,
títulos, linhas "Aluno:/Aluna:" e o nome do autor (menores de idade), e as
notas "Texto baseado na entrevista...". Junta hifenização de fim de linha
e segmenta em sentenças (ou versos agrupados por estrofe, nos poemas),
mantendo unidades de 3 a 80 palavras, como nas outras fontes escritas.

group_id: edição + categoria + bloco de páginas (aproximação de "mesmo
texto"; um texto costuma ocupar 1 a 3 páginas).

ENGATILHADO, NÃO ATIVADO: gera data/processed/olimpiada.csv, mas a fonte
não está em configs/config.yaml nem em build_and_explore.py.

Uso:
    python src/collection/collect_olimpiada.py

Lê: data/raw/olimpiada/olimpiada_{2008,2010,2012,2014,2016,2019}.pdf
Gera: data/processed/olimpiada.csv
      (schema da Tabela 1 + group_id + categoria + edicao)
"""

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

RAW_DIR = Path("data/raw/olimpiada")
EDICOES = [2008, 2010, 2012, 2014, 2016, 2019]
CATEGORIAS = {  # chaves sem espaço (em 2014 o PDF cola "Memóriasliterárias")
    "POEMA": "poema",
    "MEMÓRIASLITERÁRIAS": "memorias",
    "MEMORIASLITERARIAS": "memorias",
    "CRÔNICA": "cronica",
    "CRONICA": "cronica",
    "ARTIGODEOPINIÃO": "artigo",
    "ARTIGODEOPINIAO": "artigo",
    "DOCUMENTÁRIO": "documentario",
}
_FAIXA_PADRAO = {"poema": "5º–6º EF", "memorias": "7º–8º EF", "cronica": "9º EF–1º EM", "artigo": "2º–3º EM"}
FAIXA = {(ano, c): f for ano in (2010, 2012, 2014, 2016) for c, f in _FAIXA_PADRAO.items()}
FAIXA.update({(2019, "poema"): "5º EF", (2019, "memorias"): "6º–7º EF", (2019, "cronica"): "8º–9º EF",
              (2019, "artigo"): "3º EM", (2008, "poema"): "5º–6º EF (4ª–5ª série)",
              (2008, "memorias"): "8º–9º EF (7ª–8ª série)", (2008, "artigo"): "2º–3º EM"})
# 2008: índice (0-based) da página do PDF em que cada livro começa
FAIXAS_PAGINA_2008 = [(0, "memorias"), (102, "artigo"), (142, "poema")]
# 2010 e 2012: página impressa em que cada categoria começa (sumário)
FAIXAS_PAGINA = {
    2010: [(6, "poema"), (56, "memorias"), (134, "cronica"), (202, "artigo")],
    2012: [(6, "poema"), (60, "memorias"), (138, "cronica"), (208, "artigo")],
}
_APRESENTACAO = re.compile(r"orientados por|O que se encontra aqui|Os professores dos dois últimos|As atividades para "
                           r"a construção|foram escritos por|Escrevendo o Futuro|Olimpíada de Língua", re.I)
_SUMARIO = re.compile(r"^\s*I?([ÍIíi]ndice|Sumário)\b|^\s*(POEMA|CRÔNICA|ARTIGO DE OPINIÃO|MEM[ÓO]RIAS)\s+[ÍI]ndice", re.I)
_ITEM_SUMARIO = re.compile(r"\b\d{1,3}\s+[A-ZÀ-Ý]")  # "24 O terror da guerra 26 Outros tempos"

MIN_WORDS, MAX_WORDS = 3, 80
PAGINAS_POR_GRUPO = 3

_CAB = re.compile(r"^(POEMA|MEM[ÓO]RIAS\s*LITER[ÁA]RIAS|CR[ÔO]NICA|ARTIGO\s*DE\s*OPINI[ÃA]O|DOCUMENT[ÁA]RIO)", re.I)
_LIXO = [
    re.compile(r"^\s*\d+\s*$"),                                  # número de página
    re.compile(r"—\s*\d+\s*\d*\s*—"),                            # cabeçalho "POEMA — 23 22 — POEMA"
    re.compile(r"\.indd|\d{2}/\d{2}/\d{2}\s+\d{1,2}:\d{2}"),     # marca de impressão
    re.compile(r"^\s*alun[oa]s?\s*:", re.I),                      # "Aluno: nome"
    re.compile(r"^\s*(professor[a]?|escola|cidade)\s*:", re.I),     # "Professora: ... Escola: ... Cidade: ..."
    re.compile(r"^\s*\d+\s+\d+\s*$"),                             # par de números de página "60 61"
    re.compile(r"texto baseado n[ao]", re.I),                     # nota das Memórias
    re.compile(r"^\s*\*\s*$"),
]
_SENTENCA = re.compile(r"(?<=[.!?…])\s+")


def _categoria(texto: str, ano: int, num: int):
    linhas = [l.strip() for l in texto.strip().split("\n") if l.strip()]
    if ano == 2008:
        return [c for inicio, c in FAIXAS_PAGINA_2008 if num >= inicio][-1]
    if ano in FAIXAS_PAGINA:
        if not linhas or not linhas[0].isdigit():
            return None
        pag, cat = int(linhas[0]), None
        for inicio, c in FAIXAS_PAGINA[ano]:
            if pag >= inicio:
                cat = c
        return cat
    for l in linhas[:3]:
        m = _CAB.match(l)
        if m:
            return CATEGORIAS.get(re.sub(r"\s", "", m.group(1)).upper())
    return None


def _pagina_de_texto(texto: str) -> bool:
    """Descarta apresentações dos organizadores e páginas que são listas de nomes."""
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    if any(_SUMARIO.search(l) for l in linhas[:4]):
        return False  # sumário da seção
    if sum(bool(re.match(r"\d{1,3}\s+[A-ZÀ-Ý]", l)) or len(_ITEM_SUMARIO.findall(l)) >= 2 for l in linhas) >= 5:
        return False  # continuação do sumário (várias linhas "pág. título pág. título")
    tem_aluno = any(re.match(r"^alun[oa]\s*:", l, re.I) for l in linhas)
    if _APRESENTACAO.search(texto) and not tem_aluno:
        return False  # apresentação dos organizadores (adultos)
    if len(linhas) < 4:
        return False
    nomes = sum(1 for l in linhas if 2 <= len(l.split()) <= 6 and not re.search(r"[.,;:!?]", l)
                and all(w[:1].isupper() or w in {"de", "da", "do", "dos", "das", "e"} for w in l.split()))
    return nomes / len(linhas) < 0.6


def _limpar_pagina(texto: str) -> list[str]:
    linhas = [l.rstrip() for l in texto.split("\n")]
    saida = []
    for i, l in enumerate(linhas):
        s = l.strip()
        if not s or any(p.search(s) for p in _LIXO) or _CAB.match(s):
            if not s:
                saida.append("")  # preserva quebra de estrofe/parágrafo
            continue
        letras = re.sub(r"[^A-Za-zÀ-ÿ]", "", s)
        if letras and letras.isupper() and len(s.split()) <= 12:
            continue  # título em caixa alta (2019)
        prox = linhas[i + 1].strip() if i + 1 < len(linhas) else ""
        ant = linhas[i - 1].strip() if i > 0 else ""
        if re.match(r"^\s*alun[oa]\s*:", prox, re.I):
            continue  # título da linha anterior a "Aluno:" (2016)
        ant_letras = re.sub(r"[^A-Za-zÀ-ÿ]", "", ant)
        if ant_letras and ant_letras.isupper() and len(ant.split()) <= 12 and len(s.split()) <= 7 \
                and all(w[:1].isupper() or w in {"de", "da", "do", "dos", "das", "e"} for w in s.split()):
            continue  # nome do autor logo depois do título (2019)
        saida.append(s)
    return saida


def _unidades(linhas: list[str], categoria: str) -> list[str]:
    texto = "\n".join(linhas)
    texto = re.sub(r"-\n(\w)", r"\1", texto)                   # hifenização de fim de linha
    texto = re.sub(r"\b([TVWY]) (?=[a-zà-ÿ])", r"\1", texto)   # kerning do PDF: "T odas" -> "Todas"
    blocos = [b for b in re.split(r"\n\s*\n", texto) if b.strip()]
    unidades = []
    for b in blocos:
        corrido = re.sub(r"\s*\n\s*", " ", b).strip()
        partes = [corrido] if categoria == "poema" else _SENTENCA.split(corrido)
        for u in partes:
            n = len(u.split())
            if MIN_WORDS <= n <= MAX_WORDS:
                unidades.append(u)
            elif n > MAX_WORDS and categoria == "poema":
                unidades += [" ".join(u.split()[k:k + 40]) for k in range(0, n, 40) if len(u.split()[k:k + 40]) >= MIN_WORDS]
    return unidades


def collect_olimpiada(output_path: str = "data/processed/olimpiada.csv") -> pd.DataFrame:
    linhas_saida = []
    for ano in EDICOES:
        leitor = PdfReader(RAW_DIR / f"olimpiada_{ano}.pdf")
        for num, pagina in enumerate(leitor.pages):
            texto = pagina.extract_text() or ""
            if not texto.strip():
                continue
            cat = _categoria(texto, ano, num)
            if cat is None or cat == "documentario" or not _pagina_de_texto(texto):
                continue
            for u in _unidades(_limpar_pagina(texto), cat):
                linhas_saida.append({
                    "text": u, "underage": True, "age": None, "source": "olimpiada", "language": "pt-BR",
                    "group_id": f"olimpiada::{ano}::{cat}::{num // PAGINAS_POR_GRUPO}",
                    "categoria": cat, "faixa_escolar": FAIXA[(ano, cat)], "edicao": ano,
                })
    df = pd.DataFrame(linhas_saida).drop_duplicates(subset=["text"]).reset_index(drop=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"💾 {out}: {len(df):,} unidades de texto")
    print(df.groupby(["edicao", "categoria"]).size().to_string())
    return df


if __name__ == "__main__":
    collect_olimpiada()
