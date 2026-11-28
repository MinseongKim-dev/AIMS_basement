"""RAG 용 test 파일 만들기 — VQA-RAD 와 SLAKE 에 같은 파이프라인, 지식베이스(KB)를 선택한다.

KB 구성 (--kb, 쉼표 구분, 기본 peir,roco,trainqa):
  peir     PEIR 캡션 33,572 (PEIR_openai_imgid2caption.json, 저자 배포본 — 이 레포에 없음)
  roco     ROCO 캡션 (HF eltorio/ROCO-radiology, train; 캡션 열만 읽는다. 공개본 65,423 중 65,422 사용)
  trainqa  VQA-RAD · SLAKE 의 **train** QA. 같은 이미지의 QA 를 한 항목으로 합치고 답이 'no' 인 QA 는 뺀다.
           문장은 "질문? 답." 형식(팀원 phi3_urp 의 qa_sentence 와 같음). val / test 는 쓰지 않는다.
           --pathvqa 를 주면 PathVQA train QA(HF flaviagiammarino/path-vqa, 고정 커밋)도 같은 규칙으로 넣는다.
논문의 KB 는 ROCO 70,306 + PEIR 33,572 + 세 데이터셋 train QA(GPT-4 로 문장화)다. 논문 구성은 --pathvqa 를 준 것이고, 그래도
ROCO 는 공개본이며 문장화는 GPT-4 대신 위 템플릿이다. 결과표에는 "자체 검색 + KB 구성"으로 표기한다.

검색 (--retrieval):
  union  (train QA 가 있으면 기본) max(이미지 유사도, 질문 텍스트 유사도). 이미지 유사도는 trainqa 항목이면 이미지↔이미지, PEIR · ROCO 면 이미지→텍스트.
  cross  이미지 쿼리 x KB 텍스트만 (train QA 가 없으면 기본).
하이라이트(정답 미사용): PEIR 항목은 PEIR match_keyword, 그 외 항목은 "검색된 캡션과 질문이 공유하는 1~3-gram".
누수 점검: 같은 이미지의 train QA 가 top-1 로 검색될 수 있다(VQA-RAD 는 train / test 이미지가 많이 겹침). top-1 출처 비율과 같은 이미지 비율을 보고한다.

입력: 공통 분할 JSON(레포의 data_common/vqarad_img/{trainset,testset}.json)과 VQA-RAD 이미지 폴더. 원본 VQA-RAD JSON 은 필요 없다.
실행 (Colab, GPU):
  python -u common/build_rag_testjson.py --peir_dir <PEIR 폴더> --vqarad_common data_common/vqarad_img/testset.json       --vqarad_train data_common/vqarad_img/trainset.json --vqarad_img <VQA-RAD 이미지 폴더> --out_dir <출력 폴더> --out_suffix _v2
출력: rag_{vqarad,slake}_<kb>[_pathvqa][<접미어>].json, rag_meta_<kb>[_pathvqa][<접미어>].json, 텍스트 피처 캐시(.npy)
"""
import argparse
import hashlib
import json
import os
import re
import zipfile

import numpy as np

MID = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
TOP_K = 5
PATHVQA_REPO = 'flaviagiammarino/path-vqa'
PATHVQA_REVISION = '1685832883334b5bb5beaf4e4b333fdeecaa4ad9'
PATHVQA_TRAIN_QA = 19654

ORGAN_WORDS = {
    'chest': ['chest', 'lung', 'thorax', 'thoracic', 'heart', 'cardiac', 'cardiovascular', 'pleura', 'rib'],
    'lung': ['lung', 'chest', 'thorax', 'thoracic', 'pulmonary', 'pleura'],
    'head': ['head', 'brain', 'skull', 'neuro', 'cerebr', 'intracranial', 'sinus'],
    'brain': ['brain', 'head', 'neuro', 'cerebr', 'intracranial', 'skull'],
    'abdomen': ['abdomen', 'abdominal', 'liver', 'kidney', 'renal', 'bowel', 'spleen', 'pancrea', 'gi', 'stomach', 'gastro', 'colon'],
    'abd': ['abdomen', 'abdominal', 'liver', 'kidney', 'renal', 'bowel', 'spleen', 'pancrea', 'gi', 'stomach', 'gastro', 'colon'],
    'liver': ['liver', 'hepat', 'abdomen', 'gi'],
    'kidney': ['kidney', 'renal', 'abdomen', 'genitourinary'],
    'heart': ['heart', 'cardiac', 'cardiovascular', 'chest'],
    'neck': ['neck', 'thyroid'],
    'pelvic cavity': ['pelvi', 'bladder', 'uterus', 'prostate', 'genitourinary'],
    'spinal cord': ['spine', 'spinal', 'vertebra', 'musculoskeletal'],
}
STOP = set('a an the of in on at to for with and or is are was were be been this that these those there it its what which who whom how where when why do does did can could would should may might will shall any some no not yes by from as into than then also if'.split())


# ───────────────────────── 순수 함수 (GPU 불필요) ─────────────────────────
def peir_highlight_index(match_keyword):
    idx = {}
    for _cap, info in match_keyword.items():
        for pid in info.get('img_ids', []):
            idx.setdefault(str(pid), []).extend(info.get('match_keyword', []))
    return idx


def qmatch_highlights(question, caption, max_n=3):
    """질문과 캡션이 공유하는 1~3-gram (양 끝이 불용어가 아닌 것). 캡션의 원래 표기(대소문자)로 돌려준다 — 프로세서가 대소문자를 구분해 찾기 때문."""
    qw = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", str(question).lower())
    found, seen = [], set()
    for n in range(max_n, 0, -1):
        for i in range(len(qw) - n + 1):
            g = qw[i:i + n]
            if g[0] in STOP or g[-1] in STOP or (n == 1 and len(g[0]) < 3):
                continue
            m = re.search(r'\b' + re.escape(' '.join(g)) + r'\b', caption, flags=re.I)
            if m and m.group(0).lower() not in seen:
                seen.add(m.group(0).lower())
                found.append(m.group(0))
    return found


def organ_hit(rows, seed=0):
    rng = np.random.default_rng(seed)
    pool = [c for r in rows for c in r['top_k_captions']]
    n = t1 = t5 = r1 = r5 = 0
    for r in rows:
        organ = str(r.get('organ', '')).strip().lower()
        if not organ:
            continue
        words = ORGAN_WORDS.get(organ, [organ])
        has = lambda c: any(w in c.lower() for w in words)
        n += 1
        t1 += has(r['top_k_captions'][0])
        t5 += any(has(c) for c in r['top_k_captions'])
        rnd = [pool[i] for i in rng.integers(0, len(pool), TOP_K)]
        r1 += has(rnd[0])
        r5 += any(has(c) for c in rnd)
    n = max(n, 1)
    return {'n': n, 'top1': t1 / n, 'top5_any': t5 / n, 'random_top1': r1 / n, 'random_top5_any': r5 / n}


class KB:
    """검색 지식베이스: 텍스트, 출처(peir/roco/trainqa), 텍스트 피처, (PEIR 는 id, trainqa 는 원본 이미지 이름)."""
    def __init__(self):
        self.texts, self.src, self.key, self.feats, self.img_feats = [], [], [], [], []

    def add(self, texts, src, keys, feats, img_feats=None):
        """img_feats: 항목에 이미지가 있으면(trainqa) 그 이미지의 BiomedCLIP 피처 (union 검색의 이미지↔이미지 유사도에 쓴다)."""
        assert len(texts) == len(keys) == len(feats), (len(texts), len(keys), len(feats))
        if img_feats is not None:
            assert len(img_feats) == len(texts)
        self.texts += list(texts)
        self.src += [src] * len(texts)
        self.key += list(keys)
        self.feats.append(np.asarray(feats, dtype=np.float32))
        self.img_feats.append(None if img_feats is None else np.asarray(img_feats, dtype=np.float32))

    def finalize(self):
        self.mat = np.concatenate(self.feats)
        assert len(self.mat) == len(self.texts)
        self.img_mat = np.zeros_like(self.mat)
        self.has_img = np.zeros(len(self.mat), dtype=bool)
        off = 0
        for f, imf in zip(self.feats, self.img_feats):
            if imf is not None:
                self.img_mat[off:off + len(f)] = imf
                self.has_img[off:off + len(f)] = True
            off += len(f)
        return self

    def retrieve(self, qfeat, k=TOP_K, qtxt=None, mode='cross'):
        """mode='cross': 이미지 쿼리 x KB 텍스트 (기존 자체 검색).
        mode='union' (팀원 phi3_urp · 논문 Table 2 Union 의 해석): max(이미지 유사도, 질문 텍스트 x KB 텍스트 유사도).
          이미지 유사도 = 이미지가 있는 항목(trainqa)은 이미지↔이미지, 없는 항목(peir/roco)은 이미지→텍스트."""
        s_img = self.mat @ qfeat
        if mode == 'union':
            assert qtxt is not None
            s_img = np.where(self.has_img, self.img_mat @ qfeat, s_img)
            score = np.maximum(s_img, self.mat @ qtxt)
        else:
            score = s_img
        return np.argsort(score)[::-1][:k]


def rows_with_rag(rows, qfeats, kb, hl_index, k=TOP_K, qtxts=None, mode='cross'):
    """rows 각각에 top_k_captions / top_k_sources / highlights / top1_same_image 를 붙인다. 하이라이트는 정답을 쓰지 않는다."""
    out = []
    for n, r in enumerate(rows):
        top = kb.retrieve(qfeats[r['img_id']], k, qtxt=None if qtxts is None else qtxts[n], mode=mode)
        caps = [kb.texts[i] for i in top if kb.texts[i]]
        hls = []
        for i in top:
            if not kb.texts[i]:
                continue
            hls += hl_index.get(kb.key[i], []) if kb.src[i] == 'peir' else qmatch_highlights(r['question'], kb.texts[i])
        t1 = top[0]
        out.append({**r, 'top_k_captions': caps or [''], 'top_k_sources': [kb.src[i] for i in top],
                    'highlights': list(dict.fromkeys(hls)),
                    'top1_same_image': bool(kb.src[t1] == 'trainqa' and kb.key[t1] == r['img_id'])})
    return out


def paper_trainqa_groups(items):
    """논문 §4.1 · 부록 B 와 팀원 phi3_urp(retrieval build) 방식: 같은 이미지의 train QA 를 한 항목으로 합치고,
    답이 'no' 인 QA 는 뺀다. 문장은 "질문? 답." 형식(팀원의 qa_sentence 와 같음, GPT-4 문장화는 쓰지 않음).
    items: (이미지 이름, 이미지 경로, 질문, 답). 반환: (이미지 이름, 이미지 경로, 합친 텍스트) 목록 (입력 순서 유지)."""
    groups = {}
    for name, path, q, a in items:
        if str(a).strip().lower() == 'no':
            continue
        groups.setdefault((name, path), []).append(f"{str(q).strip().rstrip('?').strip()}? {str(a).strip()}.")
    return [(n, p, ' '.join(sents)) for (n, p), sents in groups.items()]


def kb_tag(kbs, pathvqa):
    """출력 파일 이름의 KB 표시. PathVQA 를 넣은 파일은 이름이 달라 기존 파일을 덮지 않는다."""
    if pathvqa and 'trainqa' not in kbs:
        raise ValueError('--pathvqa 는 --kb 에 trainqa 가 있어야 합니다')
    return '_'.join(kbs) + ('_pathvqa' if pathvqa else '')


def pathvqa_items(rows, img_dir):
    """rows: (이미지 바이트, 질문, 답). 반환: (이미지 이름, 경로, 질문, 답).
    PathVQA 는 행마다 파일 이름이 달라, 같은 사진은 내용의 해시를 이름으로 써서 묶는다."""
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for data, q, a in rows:
        name = f'pathvqa_{hashlib.sha256(data).hexdigest()[:16]}.jpg'
        path = os.path.join(img_dir, name)
        if not os.path.exists(path):
            with open(path, 'wb') as fh:
                fh.write(data)
        items.append((name, path, q, a))
    return items


# ───────────────────────── 인코더 · 데이터 ─────────────────────────
class Encoder:
    def __init__(self):
        import torch
        import open_clip
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.pre = open_clip.create_model_and_transforms(MID)
        self.model = self.model.to(self.dev).eval()
        self.tok = open_clip.get_tokenizer(MID)

    def text(self, texts, bs=256):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                f = self.model.encode_text(self.tok(texts[i:i + bs]).to(self.dev)).float()
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)

    def images(self, paths, bs=32):
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), bs):
                x = self.torch.stack([self.pre(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).to(self.dev)
                f = self.model.encode_image(x).float()
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def load_vqarad(path):
    rows = json.load(open(path, encoding='utf-8'))
    return rows


def vqarad_test_rows(path):
    rows = load_vqarad(path)
    assert len(rows) == 451, len(rows)
    return [{'img_id': r['image_name'], 'question': r['question'], 'answer': str(r['answer']),
             'answer_type': str(r['answer_type']).strip().upper(), 'organ': str(r.get('image_organ', ''))} for r in rows]


def slake_split(split, root='/content/slake_data'):
    from huggingface_hub import hf_hub_download
    if not os.path.isdir(f'{root}/imgs'):
        with zipfile.ZipFile(hf_hub_download('BoKelvin/SLAKE', 'imgs.zip', repo_type='dataset')) as z:
            z.extractall(root)
    return [r for r in json.load(open(hf_hub_download('BoKelvin/SLAKE', f'{split}.json', repo_type='dataset'))) if r['q_lang'] == 'en']


def slake_test_rows():
    rows = slake_split('test')
    assert len(rows) == 1061, len(rows)
    return [{'img_id': r['img_name'], 'question': r['question'], 'answer': str(r['answer']),
             'answer_type': str(r['answer_type']).upper(), 'organ': str(r.get('location', ''))} for r in rows], '/content/slake_data/imgs'


def load_roco_captions(split='train'):
    """HF eltorio/ROCO-radiology 에서 caption 열만 읽는다 (parquet 열 단위 읽기 — 이미지 용량은 받지 않는다)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f'datasets/eltorio/ROCO-radiology/data/{split}-*.parquet'))
    assert files, f'ROCO parquet 을 찾지 못했습니다 (split={split})'
    caps = []
    for f in files:
        with fs.open(f, 'rb') as fh:
            caps += pq.read_table(fh, columns=['caption']).column('caption').to_pylist()
    caps = [str(c).strip() for c in caps if c and str(c).strip()]
    return caps


def load_pathvqa_train():
    """PathVQA train 의 (이미지 바이트, 질문, 답). row group 단위로 읽는다 (약 480 MB)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f'datasets/{PATHVQA_REPO}/data/train-*.parquet', revision=PATHVQA_REVISION))
    assert files, 'PathVQA train parquet 을 찾지 못했습니다'
    for f in files:
        with fs.open(f, 'rb', revision=PATHVQA_REVISION) as fh:
            table = pq.ParquetFile(fh)
            for g in range(table.num_row_groups):
                t = table.read_row_group(g, columns=['image', 'question', 'answer'])
                for image, q, a in zip(t.column('image').to_pylist(), t.column('question').to_pylist(),
                                       t.column('answer').to_pylist()):
                    yield image['bytes'], q, a


def cached_text_feats(enc, texts, path, texts_path):
    if os.path.exists(path) and os.path.exists(texts_path):
        saved = json.load(open(texts_path, encoding='utf-8'))
        if saved == texts:
            return np.load(path)
        print(f'캐시의 텍스트가 달라 다시 인코딩합니다: {path}')
    F = enc.text(texts)
    np.save(path, F)
    json.dump(texts, open(texts_path, 'w', encoding='utf-8'), ensure_ascii=False)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peir_dir', required=True)
    ap.add_argument('--vqarad_common', required=True, help='공통 split testset.json (451문항)')
    ap.add_argument('--vqarad_train', default=None, help='공통 split trainset.json (kb 에 trainqa 가 있으면 필수)')
    ap.add_argument('--vqarad_img', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--kb', default='peir,roco,trainqa', help='peir, roco, trainqa 의 쉼표 조합')
    ap.add_argument('--roco_split', default='train')
    ap.add_argument('--trainqa', choices=('paper',), default='paper',
                    help="train QA 형식. 이미지별로 합치고 'no' 답 제외, '질문? 답.' (논문 · 팀원 방식)만 지원한다. 옛 방식(pair)은 제거했다. 기존 명령과 호환하려고 받기만 한다")
    ap.add_argument('--retrieval', choices=('cross', 'union'), default=None,
                    help='cross: 이미지 쿼리 x 텍스트 / union: max(이미지 유사도, 질문 텍스트 유사도). 기본: trainqa 가 KB 에 있으면 union, 없으면 cross')
    ap.add_argument('--out_suffix', default='', help='출력 파일 이름 접미어 (예: _v2). 기존 결과 파일을 덮어쓰지 않기 위함')
    ap.add_argument('--slake_img', default='/content/slake_data/imgs', help='SLAKE train 이미지 폴더 (trainqa=paper 에서 이미지 피처에 필요)')
    ap.add_argument('--pathvqa', action='store_true', help='train QA 에 PathVQA train 을 추가한다 (논문 KB 구성)')
    ap.add_argument('--pathvqa_img', default='/content/pathvqa_train_imgs', help='PathVQA train 이미지를 풀어 둘 폴더')
    a = ap.parse_args()
    kbs = [x.strip() for x in a.kb.split(',') if x.strip()]
    assert kbs and set(kbs) <= {'peir', 'roco', 'trainqa'}, kbs
    tag = kb_tag(kbs, a.pathvqa)
    suffix = ('' if tag == 'peir' else f'_{tag}') + a.out_suffix
    if a.retrieval is None:
        a.retrieval = 'union' if 'trainqa' in kbs else 'cross'
    if a.retrieval == 'union':
        assert 'trainqa' in kbs, 'union 검색은 이미지가 있는 trainqa 항목이 있어야 의미가 있습니다 (--kb 에 trainqa 를 넣거나 --retrieval cross)'
    print(f'KB {kbs} | 검색 {a.retrieval}')

    enc = Encoder()
    kb = KB()
    hl_index = {}
    counts = {}
    if 'peir' in kbs:
        imgid2cap = json.load(open(f'{a.peir_dir}/PEIR_openai_imgid2caption.json', encoding='utf-8'))
        ids = sorted(imgid2cap, key=lambda x: int(x))
        caps = [imgid2cap[p]['caption'] for p in ids]
        hl_index = peir_highlight_index(json.load(open(f'{a.peir_dir}/PEIR_openai_match_keyword.json', encoding='utf-8')))
        F = np.load(f'{a.out_dir}/peir_text_feats_ours.npy') if os.path.exists(f'{a.out_dir}/peir_text_feats_ours.npy') else enc.text(caps)
        if not os.path.exists(f'{a.out_dir}/peir_text_feats_ours.npy'):
            np.save(f'{a.out_dir}/peir_text_feats_ours.npy', F)
        assert len(F) == len(ids), f'PEIR 캐시 {len(F)} != {len(ids)}'
        kb.add(caps, 'peir', ids, F); counts['peir'] = len(caps)
    if 'roco' in kbs:
        caps = load_roco_captions(a.roco_split)
        F = cached_text_feats(enc, caps, f'{a.out_dir}/roco_{a.roco_split}_text_feats.npy', f'{a.out_dir}/roco_{a.roco_split}_captions.json')
        kb.add(caps, 'roco', [None] * len(caps), F); counts['roco'] = len(caps)
    if 'trainqa' in kbs:
        assert a.vqarad_train, '--vqarad_train 이 필요합니다'
        items = [(r['image_name'], os.path.join(a.vqarad_img, r['image_name']), r['question'], r['answer']) for r in load_vqarad(a.vqarad_train)]
        items += [(r['img_name'], os.path.join(a.slake_img, r['img_name']), r['question'], r['answer']) for r in slake_split('train')]
        if a.pathvqa:
            added = pathvqa_items(load_pathvqa_train(), a.pathvqa_img)
            assert len(added) == PATHVQA_TRAIN_QA, len(added)
            items += added
        groups = paper_trainqa_groups(items)
        if a.pathvqa:
            counts['trainqa_pathvqa'] = sum(n.startswith('pathvqa_') for n, _, _ in groups)
        texts = [t for _, _, t in groups]
        imf = enc.images([pth for _, pth, _ in groups]) if a.retrieval == 'union' else None
        kb.add(texts, 'trainqa', [n for n, _, _ in groups], enc.text(texts), imf); counts['trainqa'] = len(texts)
    kb.finalize()
    print('KB 구성:', counts, '| 합계', len(kb.texts))

    def encode_queries(rows, img_root):
        uniq = sorted({r['img_id'] for r in rows})
        return dict(zip(uniq, enc.images([os.path.join(img_root, u) for u in uniq])))

    meta = {'encoder': MID, 'top_k': TOP_K, 'kb': kbs, 'kb_counts': counts,
            'method': '이미지 직접 인코딩 x KB 텍스트 인코딩, 코사인 top-k (자체 검색)' if a.retrieval == 'cross'
                      else 'union = max(이미지 유사도, 질문 텍스트 유사도), 이미지 유사도는 trainqa 항목만 이미지↔이미지 (팀원 phi3_urp 방식 해석)',
            'trainqa_style': 'paper', 'retrieval': a.retrieval,
            'pathvqa': {'repo': PATHVQA_REPO, 'revision': PATHVQA_REVISION} if a.pathvqa else None}
    sl_rows, sl_img = slake_test_rows()
    for name, rows, img_root in (('vqarad', vqarad_test_rows(a.vqarad_common), a.vqarad_img), ('slake', sl_rows, sl_img)):
        qtxts = enc.text([str(r['question']) for r in rows]) if a.retrieval == 'union' else None
        out = rows_with_rag(rows, encode_queries(rows, img_root), kb, hl_index, qtxts=qtxts, mode=a.retrieval)
        json.dump(out, open(f'{a.out_dir}/rag_{name}{suffix}.json', 'w', encoding='utf-8'), ensure_ascii=False)
        s = organ_hit(out)
        top1_src = {k: sum(r['top_k_sources'][0] == k for r in out) / len(out) for k in kbs}
        same_img = sum(r['top1_same_image'] for r in out) / len(out)
        meta[name] = {'organ_hit': s, 'top1_source_share': top1_src, 'top1_same_image_trainqa': same_img}
        print(f'{name}: {len(out)}문항 | top-1 출처 {top1_src} | top-1 이 같은 이미지의 train QA: {same_img:.3f} | '
              f'장기 일치율 top1 {s["top1"]:.3f} (무작위 {s["random_top1"]:.3f})')
    json.dump(meta, open(f'{a.out_dir}/rag_meta{suffix}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('저장:', f'{a.out_dir}/rag_vqarad{suffix}.json, rag_slake{suffix}.json')


if __name__ == '__main__':
    main()
