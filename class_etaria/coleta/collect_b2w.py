"""
src/collection/collect_b2w.py

Coleta reviews do B2W-Reviews01 (Americanas.com) com idade calculada.
Fonte: https://github.com/americanas-tech/b2w-reviews01

O B2W-Reviews01 foi coletado entre janeiro e maio de 2018; a idade do
reviewer é, portanto, calculada como (2018 - ano_de_nascimento),
conforme metodologia (seção 6.4).

underage é calculado POR LINHA a partir da idade real (age < 18), igual
à regra geral do projeto (configs/config.yaml: labels.underage/of_age) —
até 18/09/2026 essa fonte rotulava TODAS as linhas como underage=False
por definição da fonte, e o filtro de idade descartava quem desse <18
anos, mesmo quando o dado de nascimento indicava um avaliador menor de
idade de verdade. Auditoria (18/09/2026) encontrou 374 avaliadores com
idade real plausível abaixo de 18 anos (majoritariamente 15-17 anos) no
dataset bruto — dado real que estava sendo descartado, não ruído. Isso
foi corrigido porque é a única fonte do projeto com idade individual
real tanto para menores quanto para maiores dentro do MESMO gênero
textual (review de produto) — as outras três fontes têm gênero
confundido com classe por construção (ver leave-one-source-out em
outputs/results/tfidf_naive_bayes/). Manter esses casos permite, pela primeira vez, testar
sinal de idade sem esse confundimento (ver
scripts/experimento_b2w_idade_real.py).

Corte de balanceamento (19/09/2026, pedido do orientador): com a entrada
do C-ORAL-BRASIL, o B2W sozinho já era 48% do dataset e quase todo maior
de idade — isso empurrava o dataset geral pra ainda mais desbalanceado
(43,6% menor). MAX_OF_AGE_ROWS reduz por amostragem aleatória (seed fixa,
reprodutível) só a maioria (age >= 18); os 354 casos reais de <18 anos
são mantidos INTEGRALMENTE (são raros e são a única evidência do projeto
livre do confundimento fonte×classe — não faz sentido descartar nenhum).
Ver scripts/testar_peso_b2w_menor.py para o experimento equivalente por
peso amostral em vez de corte, se for comparar as duas abordagens.

Corte de balanceamento (22/09/2026, pedido do orientador): teto reduzido
de 70k para 37k. B2W ainda era 74,7% da classe Adulto (69.026 de 92.383)
mesmo com C-ORAL-BRASIL e Discurso&Gramática somados — fonte dominante
dentro da própria classe, não só no dataset geral. 37k + C-ORAL
(20.296) + DG-adulto (3.061) ≈ 60k, no mesmo alvo combinado usado para a
classe Menor (Essay-BR 30k + CHILDES-PT 20k + fontes pequenas de idade
real ≈ 9,8k — ver collect_essay_br.py e collect_childes_pt.py).

Schema de saída (Tabela 1): text, underage, age, source, language.
"""

from pathlib import Path
import pandas as pd

# Ano de coleta do dataset (B2W-Reviews01: jan-mai 2018)
ANO_COLETA = 2018

B2W_URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"

# Idade mínima plausível para um avaliador (abaixo disso, o ano de
# nascimento informado é quase certamente erro de digitação/dado ruim,
# não uma criança revisando produto de verdade — auditoria de 18/09/2026
# encontrou só 4 casos abaixo desse piso, contra 370 entre 5-17 anos).
IDADE_MIN_PLAUSIVEL = 5

# Teto pra maioria (age >= 18) após limpeza/dedupe — ver nota de
# balanceamento no topo do módulo. Reduzido pra 37k (22/09/2026, pedido do
# orientador) pra tirar o B2W da dominância dentro da própria classe
# Adulto — os <18 reais NÃO entram nesse teto (mantidos 100%).
MAX_OF_AGE_ROWS = 37_000
RANDOM_SEED = 42


def collect_b2w(output_path: str = "data/processed/b2w_reviews.csv") -> pd.DataFrame:
    """
    Baixa e processa o B2W-Reviews01.
    Gera o CSV intermediário no schema da Tabela 1 do TCC.
    """
    print("⬇️  Baixando B2W-Reviews01 do GitHub...")
    try:
        df = pd.read_csv(B2W_URL, sep=",", encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"⚠️  Erro ao baixar: {e}")
        local = Path("data/raw/b2w/B2W-Reviews01.csv")
        if local.exists():
            print(f"    Carregando arquivo local: {local}")
            df = pd.read_csv(local, sep=",", encoding="utf-8", on_bad_lines="skip")
        else:
            return pd.DataFrame()

    print(f"✓ {len(df):,} reviews carregadas")

    birth_col = _find_column(df, ["reviewer_birth_year", "birth_year", "ano_nascimento", "birth", "born"])
    text_col = _find_column(df, ["review_text", "text", "texto", "body", "review_title", "title", "comment"])

    if birth_col is None or text_col is None:
        print(f"⚠️  Colunas obrigatórias não encontradas. Colunas: {list(df.columns)}")
        return pd.DataFrame()

    print(f"  Coluna de nascimento : '{birth_col}'")
    print(f"  Coluna de texto      : '{text_col}'")

    # Idade = ano_coleta - ano_nascimento (seção 6.4)
    df["age"] = ANO_COLETA - pd.to_numeric(df[birth_col], errors="coerce")

    # Validação de idade (seção 6.3): descarta ausente e implausível.
    # NÃO descarta mais quem dá <18 — ver nota no topo do módulo: são
    # avaliadores menores de idade reais, não ruído, e são o único dado
    # do projeto com idade real variando dentro do mesmo gênero textual.
    df = df[df["age"].notna()]
    df = df[(df["age"] >= IDADE_MIN_PLAUSIVEL) & (df["age"] <= 110)].copy()
    print(f"  Após validação de idade ({IDADE_MIN_PLAUSIVEL}-110 anos): {len(df):,} reviews")
    n_menores = int((df["age"] < 18).sum())
    print(f"    ↳ dos quais menores de 18 anos: {n_menores:,}")

    # group_id identifica cada review individualmente: como não há
    # segmentação (uma linha = um documento), cada review é seu próprio
    # grupo, mas a coluna é mantida para uniformidade do pipeline de split.
    group_ids = [f"b2w::{i}" for i in df.index]

    # Schema da Tabela 1 + group_id (metadado interno para o split) —
    # underage calculado por linha a partir da idade real (< 18), igual
    # à regra geral do projeto, não mais fixo por fonte.
    df_final = pd.DataFrame({
        "text":     df[text_col].astype(str).str.strip(),
        "underage": (df["age"] < 18).to_numpy(),
        "age":      df["age"].astype(int),
        "source":   "b2w_reviews",
        "language": "pt-BR",
        "group_id": group_ids,
    })

    # Filtro de comprimento conforme seção 6.3: mínimo 3 tokens
    df_final = df_final[df_final["text"].str.split().str.len() >= 3]

    # Remoção de duplicatas conforme seção 6.3
    df_final = df_final.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Corte de balanceamento (ver nota no topo do módulo): reduz só a
    # maioria (age >= 18) por amostragem aleatória até MAX_OF_AGE_ROWS;
    # os <18 reais ficam intactos, não entram na amostragem.
    menores_mask = df_final["underage"]
    n_maiores_antes = int((~menores_mask).sum())
    if n_maiores_antes > MAX_OF_AGE_ROWS:
        maiores_amostrados = df_final[~menores_mask].sample(
            n=MAX_OF_AGE_ROWS, random_state=RANDOM_SEED
        )
        df_final = pd.concat(
            [df_final[menores_mask], maiores_amostrados], ignore_index=True
        )
        print(f"\n✂️  Corte de balanceamento: maioria reduzida de {n_maiores_antes:,} para {MAX_OF_AGE_ROWS:,} "
              f"(amostragem aleatória, seed={RANDOM_SEED}); {int(menores_mask.sum()):,} menores mantidos intactos")

    print(f"\n{'='*50}")
    print(f"Total de entradas : {len(df_final):,}")
    print(f"Faixa etária      : {df_final['age'].min()}-{df_final['age'].max()} anos")
    print(f"Média de idade    : {df_final['age'].mean():.1f} anos")
    print(f"{'='*50}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out, index=False)
    print(f"\n✅ Salvo em '{out}'")

    print("\nAmostra:")
    print(df_final[["text", "underage", "age"]].head(5).to_string(index=False))

    return df_final


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Busca coluna por lista de candidatos ou palavra-chave."""
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
    collect_b2w()
    print(f"\n⏱️  Tempo total de execução: {time.perf_counter() - _t0:.1f}s")