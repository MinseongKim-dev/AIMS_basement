"""Knowledge base + BiomedCLIP retrieval (Sec. 3.3, 4.1, Table 2 'Union') and simulated expert
annotations (Sec. 5.2 'Balancing Expert Workload').

KB: all training QA pairs of the three MedVQA datasets merged per image, negative ('no')
answers excluded (Appendix B). Optional extra image-caption pairs (ROCO/PEIR) via --extra.

Expert annotation simulation (as in the paper, Sec. 5.2):
  open-ended  -> ground truth is the expert annotation; highlighted in the reference, appended
                 to the reference when the retrieved text does not contain it ("expert adds text").
  closed      -> retrieved caption; key terms matching the question are highlighted.
                 Default = deterministic keyword matching; --llm uses the GPT prompts of Fig. 8/9.

Usage:
  python -m phi3cfg.retrieval build  --data-root DATA --output kb/        [--extra roco.jsonl ...]
  python -m phi3cfg.retrieval search --kb kb/ --dataset-dir DATA/vqa_rad --output ann/vqa_rad.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from .data import DATASETS, load_data, read_jsonl

BIOMEDCLIP = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
STOP = set('''a an the is are was were be been of in on at to for from by with and or not no yes
this that these those there here it its their what which where when how why who whom does do did
can could would should shown show shows seen see image images picture photo scan visible present
any some into than as about type kind part side'''.split())

FIG7 = ('Combine the following pairs of questions and answers into concise sentences while '
        'maintaining clarity and relevance. Use proper grammar and punctuation.\n')
FIG8 = ('As a medical expert, extract and highlight key medical terms or phrases from the given '
        'question and caption. Focus on identifying specific diseases, locations, and relevant '
        'contextual information. Separate multiple terms with a semicolon (;).\n')
FIG9 = ('Given a query, select the most relevant keywords from the provided list and highlight '
        'them. If multiple keywords are relevant, separate them with a semicolon (;).\n')


def words(text):
    return re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', str(text).lower())


def content(text):
    return {w for w in words(text) if w not in STOP and len(w) > 1}


def qa_sentence(q, a):
    return f"{q.strip().rstrip('?').strip()}? {a.strip()}."


class LLM:
    """Optional OpenAI client (needs OPENAI_API_KEY). Sends dataset text to OpenAI."""
    def __init__(self, model):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def __call__(self, prompt):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, messages=[{'role': 'user', 'content': prompt}])
        return r.choices[0].message.content.strip()


# ----------------------------------------------------------------------------- encoder
class Encoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        import open_clip
        self.model, self.preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def images(self, paths, bs=64):
        feats = []
        for i in range(0, len(paths), bs):
            x = torch.stack([self.preprocess(Image.open(p).convert('RGB')) for p in paths[i:i + bs]])
            feats.append(torch.nn.functional.normalize(
                self.model.encode_image(x.to(self.device)).float(), dim=-1).cpu())
            print(f'  images {min(i + bs, len(paths))}/{len(paths)}', flush=True)
        return torch.cat(feats)

    @torch.no_grad()
    def texts(self, texts, bs=256):
        feats = []
        for i in range(0, len(texts), bs):
            x = self.tokenizer(texts[i:i + bs], context_length=256).to(self.device)
            feats.append(torch.nn.functional.normalize(
                self.model.encode_text(x).float(), dim=-1).cpu())
        return torch.cat(feats)


# ----------------------------------------------------------------------------- build
def build(args):
    root, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.llm) if args.llm else None
    entries = []
    for ds in args.datasets:
        groups = defaultdict(list)
        for r in load_data(root / ds / 'train.jsonl', 'train'):
            if r['answer'].strip().lower() != 'no':   # Appendix B: drop negative answers
                groups[r['image']].append(r)
        for image, rows in groups.items():
            sents = [qa_sentence(r['question'], r['answer']) for r in rows]
            caption = ' '.join(sents)
            if llm:
                caption = llm(FIG7 + 'Input:\n' + '\n'.join(
                    f"{r['question']} {r['answer']}" for r in rows) + '\nOutput:')
                sents = re.split(r'(?<=[.!?])\s+', caption)
            entries.append({'source': ds, 'image': str((root / ds / 'images' / image).resolve()),
                            'caption': caption, 'sentences': sents})
        print(f'{ds}: {len(groups)} images in KB', flush=True)
    for path in args.extra or []:  # {"image": abs path, "caption": text}
        for r in read_jsonl(path):
            entries.append({'source': Path(path).stem, 'image': r['image'], 'caption': r['caption'],
                            'sentences': re.split(r'(?<=[.!?])\s+', r['caption'])})
    enc = Encoder()
    img = enc.images([e['image'] for e in entries])
    txt = enc.texts([e['caption'] for e in entries])
    with (out / 'kb.jsonl').open('w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    torch.save({'image': img, 'text': txt}, out / 'kb_features.pt')
    print(f'KB: {len(entries)} entries -> {out}')


# ----------------------------------------------------------------------------- annotate
def keyword_highlights(question, caption_sents):
    """Fig. 8/9 offline stand-in: KB sentences sharing the most content words with the question."""
    q = content(question)
    scored = [(len(q & content(s)), s) for s in caption_sents]
    best = max((n for n, _ in scored), default=0)
    return [s.strip() for n, s in scored if n == best and n > 0][:2]


def llm_highlights(llm, question, caption):
    kws = llm(FIG8 + f'Input:\n{caption}\nOutput:')
    sel = llm(FIG9 + f'Query: {question}\nKeywords: {kws}\nOutput:')
    return [k.strip() for k in sel.split(';') if k.strip() and k.strip().lower() in caption.lower()]


def annotate(row, hits, llm=None):
    top = hits[0]['caption']
    rec = {'id': row['id'], 'rag_reference': top}
    if row['answer_type'] == 'open':
        gt = row['answer'].strip()
        rec['expert_reference'] = top if gt.lower() in top.lower() else f'{top} {gt}.'
        rec['highlights'] = [gt]
        rec['expert_source'] = 'ground_truth' + ('' if gt.lower() in top.lower() else '_appended')
    else:
        rec['expert_reference'] = top
        rec['highlights'] = (llm_highlights(llm, row['question'], top) if llm
                             else keyword_highlights(row['question'], hits[0]['sentences']))
        rec['expert_source'] = 'llm_keywords' if llm else 'keyword_match'
    return rec


def search(args):
    kb = read_jsonl(Path(args.kb) / 'kb.jsonl')
    feats = torch.load(Path(args.kb) / 'kb_features.pt')
    ds_dir = Path(args.dataset_dir)
    rows = load_data(ds_dir / f'{args.split}.jsonl', args.split)
    enc = Encoder()
    img_paths = sorted({r['image'] for r in rows})
    qimg = dict(zip(img_paths, enc.images([str(ds_dir / 'images' / p) for p in img_paths])))
    qtxt = enc.texts([r['question'] for r in rows])
    llm = LLM(args.llm) if args.llm else None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for r, t in zip(rows, qtxt):
            s_img = feats['image'] @ qimg[r['image']]
            s_txt = feats['text'] @ t
            if args.feature == 'image':
                score = s_img
            elif args.feature == 'text':
                score = s_txt
            elif args.feature == 'sum':
                score = s_img + s_txt
            else:
                score = torch.maximum(s_img, s_txt)  # 'union' (Table 2 best)
            vals, idx = score.topk(args.top_k)
            hits = [{'kb_index': int(i), 'score': float(v), **{k: kb[i][k] for k in
                     ('source', 'caption', 'sentences')}} for v, i in zip(vals, idx)]
            rec = annotate(r, hits, llm)
            rec['retrieved'] = [{k: h[k] for k in ('kb_index', 'score', 'source', 'caption')}
                                for h in hits]
            rec['answer_in_top1'] = r['answer'].lower() in hits[0]['caption'].lower()
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    recs = read_jsonl(out)
    open_recs = [x for x, r in zip(recs, rows) if r['answer_type'] == 'open']
    if open_recs:
        print(f'Top-1 hit rate (open): {sum(x["answer_in_top1"] for x in open_recs) / len(open_recs):.4f}')
    print('Annotations:', out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--data-root', required=True)
    b.add_argument('--output', required=True)
    b.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=DATASETS)
    b.add_argument('--extra', nargs='*', help='JSONL with {"image","caption"} (ROCO/PEIR)')
    b.add_argument('--llm', help='OpenAI model for Fig. 7 captions (optional)')
    s = sub.add_parser('search')
    s.add_argument('--kb', required=True)
    s.add_argument('--dataset-dir', required=True)
    s.add_argument('--split', default='test')
    s.add_argument('--output', required=True)
    s.add_argument('--top-k', type=int, default=3)
    s.add_argument('--feature', choices=('union', 'image', 'text', 'sum'), default='union')
    s.add_argument('--llm', help='OpenAI model for Fig. 8/9 highlights (optional)')
    args = p.parse_args()
    build(args) if args.cmd == 'build' else search(args)


if __name__ == '__main__':
    main()
