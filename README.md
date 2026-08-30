# AIMS_basement

Medical VQA experiment scaffold with CNN/ViT, uncertainty routing, and RAG components.

## Quick Start (Windows, Python 3.12)

```bat
cd c:\Users\Minseong_Kim\AIMS_basement
py -3.12 -m venv aims-env-312
aims-env-312\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Run Examples

```bat
python -m aims.check
python -m aims.rag.embed
python -m aims.rag.retriever
python -m aims.experiments.run_comparison
```

## GitHub-safe Notes

- Virtual environments are ignored via `.gitignore`.
- Cache files (`__pycache__`) and local FAISS artifacts are ignored.
- Share only source + config files, then recreate environment with `requirements.txt`.

## Project Structure

```
aims/
├── data/
│   └── dataset.py
├── models/
│   └── medcnn.py
├── rag/
│   ├── embed.py
│   ├── retriever.py
│   └── pipeline.py
├── uncertainty/
│   └── metrics.py
├── experiments/
│   └── run_comparison.py
└── __init__.py
```