FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends z3 cvc5 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/benchmark
ENV PYTHONPATH=/source PYTHONDONTWRITEBYTECODE=1
COPY run.py test_run.py ./
ENTRYPOINT ["python", "/opt/benchmark/run.py"]
