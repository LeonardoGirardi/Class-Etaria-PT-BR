# Class-Etaria-PT-BR — receita de reconstrução

O Class-Etaria-PT-BR é um conjunto de dados textual em português brasileiro para o
estudo da classificação de textos segundo a menoridade do autor (corte em
18 anos). Ele reúne 82.922 entradas de oito fontes públicas de texto
escrito e de transcrições de fala, divididas em treino, validação e teste.

Este repositório **não distribui nenhum texto**. Ele contém:

- `manifesto/class_etaria_manifesto.csv`: uma linha por entrada do conjunto,
  com o SHA-256 do texto (`id`), o rótulo de menoridade (`underage`), a
  fonte (`source`) e a partição (`split`);
- `class_etaria/`: a coleta de cada fonte, a limpeza de ruído e o script
  de reconstrução.

A reconstrução roda a coleta sobre as fontes originais, aplica a mesma
limpeza usada na construção do conjunto e mantém apenas os segmentos cujo
hash está no manifesto. O resultado são as mesmas entradas, nas mesmas
partições.

## Fontes originais

Cada fonte tem os próprios termos de uso, e cabe a quem reconstrói ou
redistribui o conjunto respeitá-los. As fontes marcadas como automáticas
são baixadas pela própria coleta; as demais devem ser colocadas em
`data/raw/` antes de rodar.

| Fonte | Obtenção | Onde colocar |
|---|---|---|
| CHILDES-PT | Corpora em português do [CHILDES](https://childes.talkbank.org/access/Romance/Portuguese/); a coleta mantém só os corpora brasileiros | `data/raw/childes_pt/` (arquivos `.cha`, em qualquer subpasta) |
| PROPOR 2024 | [Kaggle: portuguese-narrative-essays](https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays) | `data/raw/essays_5ano/` |
| B2W-Reviews01 | Automática, do [repositório oficial](https://github.com/americanas-tech/b2w-reviews01) | — |
| C-ORAL-BRASIL | Conteúdo do DVD do [C-ORAL-BRASIL](https://www.c-oral-brasil.org/) | `data/raw/c_oral_brasil/` (pastas `2. Textual Corpus` e `3. Appendix`) |
| Discurso & Gramática | Automática, do [site do corpus](https://deg.uff.br/corpus-dg/) | — |
| Olimpíada de Língua Portuguesa | Coletâneas de textos finalistas no [site do Cenpec](https://www.cenpec.org.br/pesquisa/textos-finalistas-olimpiada-de-lingua-portuguesa/) | `data/raw/olimpiada/olimpiada_{2008,2010,2012,2014,2016,2019}.pdf` |
| PEUL-1980 | Automática, da [Amostra Censo 1980](https://peul.letras.ufrj.br/amostras/censo-1980) | — |
| SP2010 | Arquivos `metadata_{train,dev,test}_final.csv` do [CORAA v1.1](https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1) | `data/raw/coraa/` |

## Reconstrução

Com Docker, que fixa as versões de Python e das bibliotecas:

```bash
docker build -t class-etaria-pt-br .
docker run --rm -v "$PWD/data:/receita/data" class-etaria-pt-br
```

Sem Docker, com Python 3.12:

```bash
pip install -r requirements.txt
python -m class_etaria.reconstruir
```

O conjunto reconstruído fica em `data/class_etaria/class_etaria_{train,val,test}.csv`,
com as colunas `text`, `underage` e `source`. Ao final, o script informa
quantas entradas de cada fonte foram recuperadas. Se uma fonte original
mudar de versão ou a extração de texto variar, as entradas afetadas não são
recuperadas, e o relatório mostra quantas faltaram.
