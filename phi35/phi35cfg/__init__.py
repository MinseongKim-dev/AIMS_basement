"""Phi-3.5-Vision (4.2B) baseline, RAG and Expert-CFG on transformers 4.44.2.

Mirrors the layout of phi3/phi3cfg; model-independent logic lives in the repository's common/ package.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:      # `common` is imported from the repository root
    sys.path.insert(0, str(REPO_ROOT))
