FROM ipfs-supervisor-preview-worker:20261005-r3
USER root
COPY wheels /tmp/numerical-dev-wheels
COPY requirements.lock /tmp/numerical-dev-requirements.lock
RUN /opt/ipfs-supervisor/venv/bin/python -m pip --isolated install --no-index --no-deps --require-hashes --find-links=/tmp/numerical-dev-wheels --report=/opt/ipfs-supervisor/numerical-dev-install.json -r /tmp/numerical-dev-requirements.lock && /opt/ipfs-supervisor/venv/bin/python -m pip check && /opt/ipfs-supervisor/venv/bin/python -m pip list --format=json > /opt/ipfs-supervisor/numerical-dev-packages.json && /opt/ipfs-supervisor/venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cpu'" && rm -rf /tmp/numerical-dev-wheels /tmp/numerical-dev-requirements.lock
USER supervisor
