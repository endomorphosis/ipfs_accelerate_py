FROM python@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed
RUN pip install numpy==2.3.0
RUN if ! command -v gcc >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*; fi
WORKDIR /app
