FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/ ./service/
COPY data/ ./data/
COPY tools/ ./tools/

ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "service/app.py"]

