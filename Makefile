CEBRA_LENS_VERSION := 0.1.0.dev0

dist:
	python3 -m pip install virtualenv
	python3 -m pip install --upgrade build twine
	python3 -m build --wheel --sdist

build: dist

test:
	python -m pytest --ff tests

interrogate:
	interrogate \
		--ignore-property-decorators \
		--ignore-init-method \
		--verbose \
		--ignore-semiprivate \
		--ignore-private \
		--ignore-magic \
		--omit-covered-files \
		-f 80 \
		cebra_lens

docs:
	export PYTHONPATH=$(pwd)
	jupyter-book build docs

docs-touch:
	find docs/docs -iname '*.md' -exec touch {} \;
	jupyter-book build docs/docs

docs-strict:
	jupyter-book build docs --keep-going --strict

# Serve the docs
serve_docs:
	python -m http.server 8080 --bind 127.0.0.1 -d docs/_build/html

# Serve the entire page
serve_page:
	python -m http.server 8080 --bind 127.0.0.1 -d docs/_build/html

# Format code in the main package and docs
format:
	yapf -i -p -r cebra_lens
	yapf -i -p -r tests
	isort cebra_lens/
	isort tests/

codespell:
	codespell cebra_lens/ tests/ docs/docs/*.md -L "nce, nd"


.PHONY: docs docs-touch docs-strict serve_docs serve_page