FROM ipfs-supervisor-numerical-dev:20261011-r1
USER root
COPY wheels /tmp/supervisor-dev-wheels
COPY requirements.lock /tmp/supervisor-dev-requirements.lock
RUN /opt/ipfs-supervisor/venv/bin/python -m pip --isolated install --no-index --no-deps --require-hashes --find-links=/tmp/supervisor-dev-wheels --report=/opt/ipfs-supervisor/supervisor-dev-install.json -r /tmp/supervisor-dev-requirements.lock && /opt/ipfs-supervisor/venv/bin/python -m pip check && /opt/ipfs-supervisor/venv/bin/python -m pip list --format=json > /opt/ipfs-supervisor/supervisor-dev-packages.json && /opt/ipfs-supervisor/venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cpu'" && rm -rf /tmp/supervisor-dev-wheels /tmp/supervisor-dev-requirements.lock
USER supervisor
