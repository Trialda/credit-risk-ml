FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -r requirements.txt

COPY . ./ml/

ENV PYTHONPATH=/app

CMD ["python", "-m", "ml.train"]