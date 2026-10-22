"""
src/collection/collect_peul_censo1980.py

Coleta a fala dos INFORMANTES da Amostra Censo 1980 do PEUL/UFRJ (Programa
de Estudos sobre o Uso da Língua): entrevistas sociolinguísticas gravadas
no Rio de Janeiro entre 1980 e 1982, transcritas em modo semi-ortográfico.
Fonte: https://peul.letras.ufrj.br/amostras/censo-1980 (transcrições .txt
publicadas em links do Google Drive, uma por falante).

Por que interessa: é uma das poucas fontes abertas de FALA ESPONTÂNEA de
crianças em idade escolar com idade anotada. A amostra junta dois projetos:
  - Amostra Censo: falantes de 15 a 71 anos (5 deles com 15 a 17 anos);
  - "Estruturas da Fala e Aquisição da Língua Padrão" (1981): 16 crianças
    de 7 a 14 anos, entrevistadas em escolas municipais do Rio.
Assim, cobre a faixa de 7 a 14 anos, que falta no CHILDES-PT (77% de 1 a
3 anos), e traz menores e adultos no mesmo gênero (entrevista).

Idade: exata por falante, na tabela da página do PEUL (coluna "Idade").
underage = age < 18. Só as falas do informante ("F-") entram; as do
entrevistador ("E-") são descartadas, porque não têm idade.

Transcrição: marcas do PEUL removidas: comentários entre parênteses
("(ruídos)", "(est)", "(f)"), palavras truncadas entre <...>; os colchetes
de sobreposição/repetição perdem os símbolos e mantêm as palavras; o hífen
de autocorreção ("quando- quando") vira espaço. Os turnos são segmentados
em sentenças por pontuação terminal, com 3 a 80 palavras, como nas outras
fontes.

group_id: código do falante (C01 a C64), para o split sem vazamento.

Limitações: gravações de 1980–1982 (o CHILDES-PT também tem corpora
antigos); amostra pequena (61 falantes com transcrição disponível: C13 não
existe na listagem e C38 não tem arquivos); só a variedade carioca.

ENGATILHADO, NÃO ATIVADO: gera data/processed/peul_censo1980.csv, mas a
fonte não está em configs/config.yaml nem em build_and_explore.py.

Uso:
    python src/collection/collect_peul_censo1980.py

Lê/baixa: data/raw/peul_censo1980/EntrevistaNN.txt (baixa o que faltar)
Gera: data/raw/peul_censo1980/falantes.csv (falante, idade, id do arquivo)
      data/processed/peul_censo1980.csv (schema da Tabela 1 + group_id + amostra)
"""

import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

PAGINAS = ["https://peul.letras.ufrj.br/amostras/censo-1980",
           "https://peul.letras.ufrj.br/amostras/censo-1980/c%C3%B3pia-de-censo-1980-2"]
RAW_DIR = Path("data/raw/peul_censo1980")
DRIVE = "https://drive.google.com/uc?export=download&id={}"
MIN_WORDS, MAX_WORDS = 3, 80

_FALANTE = re.compile(r">\s*(C\d{2})\s+(\d+)\s*anos\s*<")
_LINK = re.compile(r'href="https://drive\.google\.com/file/d/([^/]+)/')
_SENTENCA = re.compile(r"(?<=[.!?…])\s+")


def listar_falantes() -> pd.DataFrame:
    """Lê as duas páginas da amostra; o 1º link depois de cada falante é a transcrição."""
    linhas = []
    for url in PAGINAS:
        h = requests.get(url, timeout=60).text
        eventos = sorted([(m.start(), "F", m.groups()) for m in _FALANTE.finditer(h)]
                         + [(m.start(), "L", m.group(1)) for m in _LINK.finditer(h)])
        atual = None
        for _, tipo, valor in eventos:
            if tipo == "F":
                atual = {"falante": valor[0], "idade": int(valor[1]), "drive_id": None}
                linhas.append(atual)
            elif atual is not None and atual["drive_id"] is None:
                atual["drive_id"] = valor
    return pd.DataFrame(linhas).dropna(subset=["drive_id"])


def baixar() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / "falantes.csv"
    falantes = pd.read_csv(meta_path) if meta_path.exists() else listar_falantes()
    arquivos = []
    for f in falantes.itertuples():
        destino = RAW_DIR / f"Entrevista{f.falante[1:]}.txt"
        if not destino.exists():
            r = requests.get(DRIVE.format(f.drive_id), timeout=120)
            nome = re.search(r'filename="([^"]+)"', r.headers.get("content-disposition", ""))
            if not nome or nome.group(1) != destino.name:
                print(f"  ⚠ {f.falante}: o 1º link não é a transcrição esperada ({nome.group(1) if nome else 'sem nome'})")
                arquivos.append(None)
                continue
            destino.write_bytes(r.content)
            time.sleep(0.5)
        arquivos.append(destino.name)
    falantes["arquivo"] = arquivos
    falantes.to_csv(meta_path, index=False)
    return falantes.dropna(subset=["arquivo"])


def _limpar(t: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", t)          # comentários: (ruídos), (est), (f)
    t = re.sub(r"<[^>]*>", " ", t)            # palavras truncadas: <depo...>
    t = t.replace("[", " ").replace("]", " ")  # sobreposição/repetição: mantém as palavras
    t = re.sub(r"(?<=\w)-(?=\s|$)", " ", t)   # autocorreção: "quando- quando"
    t = re.sub(r"\s-\s", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def turnos_do_informante(texto: str) -> list[str]:
    """Junta linhas de continuação ao turno anterior e devolve só os turnos "F-"."""
    turnos, atual = [], None
    for linha in re.split(r"\r\n|\r|\n", texto):
        s = linha.strip()
        if not s:
            continue
        m = re.match(r"^([EF])\s*-\s*(.*)$", s)
        if m:
            atual = [m.group(1), m.group(2)]
            turnos.append(atual)
        elif atual is not None:
            atual[1] += " " + s
    return [t for quem, t in turnos if quem == "F"]


def collect_peul_censo1980(output_path: str = "data/processed/peul_censo1980.csv") -> pd.DataFrame:
    falantes = baixar()
    linhas = []
    for f in falantes.itertuples():
        texto = (RAW_DIR / f.arquivo).read_bytes().decode("cp1252", errors="replace")
        for turno in turnos_do_informante(html.unescape(texto)):
            for u in _SENTENCA.split(_limpar(turno)):
                if MIN_WORDS <= len(u.split()) <= MAX_WORDS:
                    linhas.append({"text": u, "underage": f.idade < 18, "age": f.idade, "source": "peul_censo1980",
                                   "language": "pt-BR", "group_id": f"peul::{f.falante}",
                                   "amostra": "criancas_1981" if f.idade <= 14 else "censo"})
    df = pd.DataFrame(linhas)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    por_falante = df.groupby("group_id").agg(idade=("age", "first"), n=("text", "size"))
    print(f"💾 {out}: {len(df):,} sentenças de {len(por_falante)} falantes "
          f"({int((por_falante['idade'] < 18).sum())} menores)")
    print(df.groupby("underage").size().rename({True: "menor", False: "maior"}).to_string())
    return df


if __name__ == "__main__":
    collect_peul_censo1980()
