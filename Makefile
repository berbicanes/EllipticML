# Rebuild every number, table and figure from the raw CSV files.
#   make all       full pipeline (roughly 2-3 hours on an Apple M4; GNNs use the GPU)
#   make test      unit and leakage tests
#   make e04       a single experiment (re-runs it, plus anything it needs that is missing)
#   make report    compile the written report (needs tectonic: brew install tectonic)
#
# Each experiment writes results/<name>.json; later experiments depend on
# those files, so an experiment only re-runs when its script is newer than
# its result.

PY := uv run python -u
R := results

.PHONY: all data test lint report slides clean e00 e01 e02 e03 e04 e05 e05b e06 e07 e08

all: test $(R)/e00_eda.json $(R)/e01_reproduce.json $(R)/e02_models.json \
     $(R)/e03_imbalance.json $(R)/e04_graph.json $(R)/e05_hybrid.json $(R)/e05b_memorisation.json $(R)/e06_unsupervised.json \
     $(R)/e07_drift.json $(R)/e08_explain.json

data: data/processed/nodes.parquet
data/processed/nodes.parquet:
	$(PY) -c "from elliptic import data; d = data.load(); print(d.nodes.shape, d.edges.shape)"

test: data
	uv run pytest

lint:
	uv run ruff check src experiments tests
	uv run ruff format --check src experiments tests

$(R)/e00_eda.json: experiments/e00_eda.py data/processed/nodes.parquet
	$(PY) $<
$(R)/e01_reproduce.json: experiments/e01_reproduce.py data/processed/nodes.parquet
	$(PY) $<
$(R)/e02_models.json: experiments/e02_models.py data/processed/nodes.parquet
	$(PY) $<
$(R)/e03_imbalance.json: experiments/e03_imbalance.py $(R)/e02_models.json
	$(PY) $<
$(R)/e04_graph.json: experiments/e04_graph.py $(R)/e02_models.json
	$(PY) $<
$(R)/e05_hybrid.json: experiments/e05_hybrid.py $(R)/e04_graph.json
	$(PY) $<
$(R)/e05b_memorisation.json: experiments/e05b_memorisation.py $(R)/e04_graph.json
	$(PY) $<
$(R)/e06_unsupervised.json: experiments/e06_unsupervised.py $(R)/e02_models.json
	$(PY) $<
$(R)/e07_drift.json: experiments/e07_drift.py $(R)/e02_models.json $(R)/e04_graph.json
	$(PY) $<
$(R)/e08_explain.json: experiments/e08_explain.py $(R)/e02_models.json
	$(PY) $<

# Force a single experiment to re-run: make e04
e00 e01 e02 e03 e04 e05 e05b e06 e07 e08:
	$(PY) experiments/$@_*.py

report:
	$(PY) report/build_numbers.py
	cd report && tectonic main.tex

slides:
	cd slides && tectonic slides.tex

clean:
	rm -rf data/processed/*.parquet artifacts
