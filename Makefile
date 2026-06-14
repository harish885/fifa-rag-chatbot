# Convenience targets. Windows users without `make` can run the underlying
# Python commands shown in the README directly.
PYTHON ?= python3

.PHONY: install test evaluate answers notebook run audit reproduce

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

evaluate:
	$(PYTHON) eval/evaluate.py

answers:
	$(PYTHON) eval/evaluate_answers.py

notebook:
	jupyter nbconvert --to notebook --execute --output /tmp/walkthrough.executed.ipynb walkthrough.ipynb

run:
	$(PYTHON) local_server.py

audit:
	$(PYTHON) -m pip_audit -r requirements.txt

# Offline reproduction: tests + evaluation regression check (no Groq key needed).
reproduce: test evaluate
	$(PYTHON) eval/check_regression.py
