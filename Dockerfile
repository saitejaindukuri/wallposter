FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
build-essential \
g++ \
libopenblas-dev \
libgomp1 \
libsqlite3-dev \
curl \
wget && apt-get clean \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app

RUN pip install  --no-cache-dir --target=/python-deps -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /python-deps /usr/local/lib/python3.10/site-packages/

COPY . /app

EXPOSE 9000

CMD ["python3", "app.py"]




