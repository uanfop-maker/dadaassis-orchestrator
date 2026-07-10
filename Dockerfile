FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/_jobs/pending /data/_jobs/running /data/_jobs/done /data/_jobs/dead /data/_state

ENV DATA_DIR=/data
ENV PORT=8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
