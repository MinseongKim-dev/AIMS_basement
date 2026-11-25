"""Retrieval files (built by common/build_rag_testjson.py) and the compliant highlight rule.

Nothing here takes the ground-truth answer: it is used only to verify that a retrieval file
belongs to the split being run.
"""
import json
from pathlib import Path

from . import REPO_ROOT  # noqa: F401  (puts `common` on sys.path)
from .data import align
from common.highlights import closed_sentence_highlights


def load_retrieval(path, rows):
    """Per row: {'reference': top-1 caption, 'file_highlights': highlights stored by the builder}."""
    name = Path(path).name
    records = align(rows, json.loads(Path(path).read_text(encoding='utf-8')), name)
    out = []
    for row, rec in zip(rows, records):
        if rec['img_id'] != row['image']:
            raise ValueError(f'{name}: idx {row["idx"]} is for image {rec["img_id"]}, the split has {row["image"]}')
        out.append({'reference': rec['top_k_captions'][0], 'file_highlights': list(rec['highlights'])})
    return out


def compliant_highlights(question, answer_type, reference, file_highlights):
    """closed: reference sentences sharing most content words with the question (at most two);
    open: the highlights stored in the retrieval file."""
    if answer_type == 'CLOSED':
        return closed_sentence_highlights(question, reference)
    return list(file_highlights)
