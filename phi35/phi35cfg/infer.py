"""Run Phi-3.5V-Med on a split in one mode (resumable JSONL output).

  baseline    question only; records the eq. 4 entropy
  rag         top-1 retrieved caption in the prompt (all questions, or the top-x% with --top-percent)
  expert_cfg  reference + highlights + classifier-free guidance (eq. 7-10) on the top-x% questions
  entropy     eq. 4 entropy of existing predictions by teacher forcing (docs/UNCERTAINTY_SPEC.md)
  yesno       val only: first-token "yes" / "no" probabilities of the yes/no questions (a check, not eq. 4)

No mode puts the ground-truth answer into the model input.

  python -m phi35cfg.infer --mode expert_cfg --dataset vqarad --image-root IMAGES --base-dir BASE \\
      --adapter ADAPTER --retrieval rag.json --entropy entropy.json --top-percent 5 --output cfg5.jsonl
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from . import generation
from . import model as model_lib
from . import yesno
from .data import DATASETS, SPLITS, align, append_record, file_sha256, load_split, open_run, read_predictions
from .retrieval import compliant_highlights, load_retrieval
from common.gating import select_top_percent

ENTROPY_DEFINITION = 'eq4: answer-token vocabulary entropy, special tokens excluded, mean over answer tokens'
ENTROPY_MODE = 'teacher_forcing_on_baseline_predictions'
SAVE_EVERY = 100


def select_ids(entropy, n, percent=5.0):
    """idx of the top-percent highest-entropy questions, in ascending idx order."""
    return sorted(select_top_percent([entropy.get(i) for i in range(n)], percent))


def load_entropy(path, dataset, n, predictions=None):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    name = Path(path).name
    if data.get('dataset') != dataset:
        raise ValueError(f'{name}: entropy of dataset {data.get("dataset")!r}, expected {dataset!r}')
    if predictions is not None and data.get('predictions') != predictions:
        raise ValueError(f'{name}: computed on predictions {data.get("predictions")!r}, expected {predictions!r}')
    values = data['entropy']
    if set(values) != {str(i) for i in range(n)}:
        raise ValueError(f'{name}: {len(values)} entries; the keys must be exactly 0..{n - 1}')
    return {int(k): v for k, v in values.items()}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=('baseline', 'rag', 'expert_cfg', 'entropy', 'yesno'))
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--split', default='test', choices=SPLITS)
    p.add_argument('--image-root', help='VQA-RAD image folder')
    p.add_argument('--data-root', help='SLAKE download folder')
    p.add_argument('--base-dir', required=True, help='local folder of the base model (downloaded if empty)')
    p.add_argument('--adapter', required=True, help='downstream LoRA folder')
    p.add_argument('--output', required=True)
    p.add_argument('--retrieval', help='rag_*.json from common/build_rag_testjson.py')
    p.add_argument('--entropy', help='entropy.json used to select the top-x%% questions')
    p.add_argument('--top-percent', type=float)
    p.add_argument('--predictions', help='entropy mode: baseline predictions to score')
    p.add_argument('--predictions-name', help='release name of the baseline predictions')
    p.add_argument('--adapter-name', help='entropy mode: release name of the adapter')
    p.add_argument('--gamma', type=float, default=1.5)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=3.0)
    p.add_argument('--delta', type=float, help='default log(beta)+2')
    p.add_argument('--limit', type=int)
    return p


def _check(parser, a):
    if a.mode in ('rag', 'expert_cfg') and not a.retrieval:
        parser.error('--retrieval is required for rag and expert_cfg')
    if a.mode == 'expert_cfg' and a.top_percent is None:
        parser.error('expert_cfg runs on the selected questions only: give --top-percent')
    if a.top_percent is not None and not a.entropy:
        parser.error('--top-percent needs --entropy')
    if a.mode == 'entropy' and not (a.predictions and a.predictions_name and a.adapter_name):
        parser.error('entropy mode needs --predictions, --predictions-name and --adapter-name')
    if a.mode == 'yesno' and a.split != 'val':
        parser.error('yesno compares entropy definitions: it runs on --split val only')


def _image(images, row):
    with Image.open(Path(images) / row['image']) as im:
        return im.convert('RGB')


def _run_entropy(a, rows, images):
    preds = align(rows, read_predictions(a.predictions).values(), Path(a.predictions).name)
    head = {'model': 'phi35', 'dataset': a.dataset, 'definition': ENTROPY_DEFINITION, 'mode': ENTROPY_MODE,
            'adapter': a.adapter_name, 'predictions': a.predictions_name, 'n': len(rows)}
    out, values = Path(a.output), {}
    if out.exists():
        old = json.loads(out.read_text(encoding='utf-8'))
        changed = [k for k in head if old.get(k) != head[k]]
        if changed:
            raise ValueError(f'{out} holds another entropy run (differs: {changed}); use a new --output')
        values = old['entropy']
    model, processor, meta = model_lib.load(a.base_dir, a.adapter)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps({**head, 'env': meta, 'entropy': values}, ensure_ascii=False), encoding='utf-8')

    todo = [r for r in rows if str(r['idx']) not in values]
    if a.limit is not None:
        todo = todo[:a.limit]
    for n, r in enumerate(todo, 1):
        values[str(r['idx'])] = generation.answer_entropy(
            model, processor, _image(images, r), r['question'], preds[r['idx']]['prediction'])
        if n % SAVE_EVERY == 0:
            save()
            print(f'[{n}/{len(todo)}]', flush=True)
    save()
    print(f'Entropy: {out} ({len(values)}/{len(rows)})')


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    _check(parser, a)
    rows, images = load_split(a.dataset, a.split, a.image_root, a.data_root)
    if a.mode == 'entropy':
        return _run_entropy(a, rows, images)

    retrieval = load_retrieval(a.retrieval, rows) if a.mode in ('rag', 'expert_cfg') else None
    targets = rows
    if a.mode == 'yesno':
        targets = [r for r in rows if yesno.label_of(r['answer'])]
    elif a.mode != 'baseline' and a.top_percent is not None:
        entropy = load_entropy(a.entropy, a.dataset, len(rows), a.predictions_name)
        selected = set(select_ids(entropy, len(rows), a.top_percent))
        targets = [r for r in rows if r['idx'] in selected]
        print(f'Selected {len(targets)} high-entropy questions', flush=True)
    cfg = generation.CFGConfig(a.gamma, a.alpha, a.beta, a.delta) if a.mode == 'expert_cfg' else None

    model, processor, meta = model_lib.load(a.base_dir, a.adapter)
    config = {k: v for k, v in vars(a).items() if k not in ('limit', 'output')}
    config['runtime'] = meta
    if cfg is not None:
        config['cfg'] = vars(cfg)
    for key, path in (('retrieval_sha256', a.retrieval), ('entropy_sha256', a.entropy),
                      ('adapter_sha256', Path(a.adapter) / 'adapter_model.safetensors')):
        if path and Path(path).is_file():
            config[key] = file_sha256(path)
    done = open_run(a.output, config)
    todo = [r for r in targets if r['idx'] not in done]
    if a.limit is not None:
        todo = [r for r in targets[:a.limit] if r['idx'] not in done]
    ids = yesno.label_token_ids(processor.tokenizer) if a.mode == 'yesno' else None

    for n, r in enumerate(todo, 1):
        image = _image(images, r)
        if a.mode == 'yesno':
            logp = generation.first_token_logp(model, processor, image, r['question'])
            append_record(a.output, {
                'idx': r['idx'], 'question': r['question'], 'answer': r['answer'], 'answer_type': r['answer_type'],
                'mode': a.mode, **yesno.probe_record(logp, ids)})
            print(f'[{n}/{len(todo)}] Q: {r["question"][:60]}', flush=True)
            continue
        entropy = None
        if a.mode == 'baseline':
            res = generation.generate(model, processor, image, r['question'])
            entropy = generation.answer_entropy(model, processor, image, r['question'], res['prediction'])
        else:
            ref = retrieval[r['idx']]['reference']
            highlights = None
            if cfg is not None:
                highlights = compliant_highlights(r['question'], r['answer_type'], ref,
                                                  retrieval[r['idx']]['file_highlights'])
            res = generation.generate(model, processor, image, r['question'], ref, highlights, cfg)
        append_record(a.output, {
            'idx': r['idx'], 'question': r['question'], 'answer': r['answer'], 'answer_type': r['answer_type'],
            'prediction': res['prediction'], 'entropy': entropy, 'mode': a.mode,
            'cfg_applied': res['cfg_applied'], 'highlight_tokens': res['highlight_tokens']})
        print(f'[{n}/{len(todo)}] Q: {r["question"][:60]} | P: {res["prediction"][:80]}', flush=True)
    print('Predictions:', a.output)


if __name__ == '__main__':
    main()
