FROM python:3.12.3-slim

WORKDIR /receita
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY class_etaria/ class_etaria/
COPY manifesto/ manifesto/

# data/ é montado de fora: data/raw/ com as fontes originais (ver README.md)
# e data/class_etaria/ recebe o conjunto reconstruído.
VOLUME ["/receita/data"]
ENTRYPOINT ["python", "-m", "class_etaria.reconstruir"]
