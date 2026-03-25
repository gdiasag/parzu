UV_RUN := .venv-uv/bin/uv

# Virtual environment

install-uv:
	python3 -m venv .venv-uv
	.venv-uv/bin/pip install uv

upgrade-uv:
	.venv-uv/bin/pip install --upgrade pip
	.venv-uv/bin/pip install --upgrade uv

venv-min:
	$(UV_RUN) sync 

venv-clean:
	rm -rf .venv

# Requirements

requirements-min:
	$(UV_RUN) export --no-hashes --no-emit-workspace --no-group dev --output-file requirements.txt

requirements-dev:
	$(UV_RUN) export --no-hashes --only-dev --output-file dev-requirements.txt

requirements: requirements-min requirements-dev

requirements-upgrade: venv-upgrade requirements

# Run

run:
	$(UV_RUN) run python parzu/parzu_server.py
