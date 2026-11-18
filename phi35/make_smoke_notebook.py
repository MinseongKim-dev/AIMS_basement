"""phi35/smoke_check.ipynb 를 만든다 (transformers 4.44.2 에서 Phi-3.5V 가 동작하는지 점검하는 Colab 노트북).
  python phi35/make_smoke_notebook.py
점검 항목: 로드(어댑터 포함) / 기존 저장 예측 재현 / CFG 훅 중립·논문 설정 / KV 캐시 유무 디코딩 일치 / 환경 기록.
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    CELLS.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.strip('\n').splitlines(keepends=True)})


md('''
# Phi-3.5V · transformers 4.44.2 스모크 점검

팀 결정(transformers==4.44.2)에서 Phi-3.5V가 동작하는지 확인한다. 이전 결과는 transformers 5.x 패치 환경에서 나온 것이라, 환경이 바뀌어도 같은 결과가 나오는지가 핵심이다.

| 점검 | 통과 기준 |
|---|---|
| 1. 로드 | 기반 모델과 VQA-RAD 어댑터(ckpt-257)가 4.44.2 + peft 0.12.0 에서 로드됨 |
| 2. 기존 예측 재현 | 저장된 베이스라인 예측과 문항 50개 비교, 일치율 보고 (100% 가 아니어도 원인을 적는다) |
| 3. CFG 훅 | 중립 설정이면 일반 생성과 같은 문자열, 논문 설정이면 하이라이트가 있는 문항에서 logits 가 달라짐 |
| 4. KV 캐시 | 캐시 유무로 같은 토큰 |

**순서대로 실행.** 1번 셀은 설치 후 런타임을 다시 시작해야 할 수 있다(numpy<2).
''')

code(r'''
# 1) 환경 고정 — 설치 뒤 런타임 다시 시작이 필요할 수 있다 (그 뒤 이 셀을 다시 실행해 버전 확인)
!pip -q install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 huggingface-hub==0.25.2 open_clip_torch==2.26.1 "numpy<2" scikit-learn pandas pyarrow
import transformers, peft, torch, numpy
print('transformers', transformers.__version__, '| peft', peft.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)
assert transformers.__version__ == '4.44.2', '런타임을 다시 시작한 뒤 이 셀을 다시 실행하세요'
print('GPU:', torch.cuda.get_device_name(), '| bf16 지원:', torch.cuda.is_bf16_supported())
''')

code(r'''
# 2) 경로 · 기반 모델 · 저자 processor 파일
from google.colab import drive; drive.mount('/content/drive')
import os, json, glob, shutil, time, urllib.request, inspect
from PIL import Image
E = '/content/drive/MyDrive/expert_cfg'; R = '/content/drive/MyDrive/ExpertCFG_replication'
ADAPTER  = f'{E}/logs_phi35_vqarad_continue_e10_env_common/checkpoint-257'
BASELINE = f'{E}/vqarad_continue_e10_env_common_ckpt257_test.json'
RAG      = f'{R}/rag_vqarad_peir_roco_trainqa_v2.json'            # 질문 · 이미지 이름 · 하이라이트
IMG_ROOT = '/content/drive/MyDrive/vqa_rad/VQA_RAD Image Folder'
BASE_DIR = '/content/Phi-3.5-vision-instruct'
for p in (ADAPTER, BASELINE, RAG, IMG_ROOT): assert os.path.exists(p), f'없음: {p}'

from huggingface_hub import snapshot_download
if not os.path.exists(f'{BASE_DIR}/config.json'):
    snapshot_download('microsoft/Phi-3.5-vision-instruct', local_dir=BASE_DIR)
# 저자 processor (하이라이트 마스크를 만든다). 저자 레포에 라이선스가 없어 레포에 복사하지 않고 고정 커밋에서 받는다
REV = 'edcb03c1ad406cc87171319d59a628ccefa9fbdc'
for f in ('processing_phi3_v_cfg.py', 'processor_config.json'):
    url = f'https://raw.githubusercontent.com/ecoxial2007/Expert-CFG/{REV}/Phi-3.5-vision-instruct/{f}'
    open(f'{BASE_DIR}/{f}', 'wb').write(urllib.request.urlopen(url, timeout=60).read())
print('기반 모델과 저자 processor 준비 완료')
''')

code(r'''
# 3) 로드 — 어댑터가 더 새로운 peft 로 저장됐다면 4.44.2 의 peft 0.12.0 에서 설정 키 오류가 날 수 있다 → 알 수 없는 키를 걸러서 재시도
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel, LoraConfig
DEV = 'cuda'; DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
processor = AutoProcessor.from_pretrained(BASE_DIR, trust_remote_code=True, num_crops=16)
tok = processor.tokenizer
base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=DTYPE, trust_remote_code=True, _attn_implementation='eager', device_map={'': DEV})
LOAD_NOTE = '어댑터를 그대로 로드'
try:
    model = PeftModel.from_pretrained(base, ADAPTER)
except TypeError as e:
    print('어댑터 설정 로드 실패:', str(e)[:200])
    clean = '/content/adapter_clean'; shutil.rmtree(clean, ignore_errors=True); shutil.copytree(ADAPTER, clean)
    cfg = json.load(open(f'{clean}/adapter_config.json')); ok = set(inspect.signature(LoraConfig).parameters)
    removed = sorted(k for k in cfg if k not in ok); cfg = {k: v for k, v in cfg.items() if k in ok}
    json.dump(cfg, open(f'{clean}/adapter_config.json', 'w'), indent=2)
    print('제거한 설정 키:', removed)
    model = PeftModel.from_pretrained(base, clean); LOAD_NOTE = f'adapter_config 의 키 {removed} 를 제거하고 로드'
model.eval()
END_ID = tok.convert_tokens_to_ids('<|end|>'); EOS = {tok.eos_token_id, END_ID}
print('로드 완료 |', LOAD_NOTE, '| dtype', DTYPE)
''')

code(r'''
# 4) 점검 2 — 기존 저장 예측 재현 (model.generate, greedy). 같은 프롬프트: "<|image_1|>\n{질문}", 하이라이트 없음
rows = json.load(open(RAG)); base_recs = json.load(open(BASELINE))
by = {b['idx']: b for b in base_recs}
N_CHECK = 50
def gen_plain(img, question, caption=''):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': (f'{caption} ' if caption else '') + f'<|image_1|>\n{question}'}], tokenize=False, add_generation_prompt=True)
    inp = processor(prompt, [img], return_tensors='pt', qs_highlighted_parts=[]); inp.pop('highlight_attention_mask', None)
    inp = {k: v.to(DEV) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False, num_beams=1, eos_token_id=list(EOS), use_cache=True)
    return processor.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().strip('.')
same, diffs, t0 = 0, [], time.time()
for i in range(N_CHECK):
    r = rows[i]; img = Image.open(os.path.join(IMG_ROOT, r['img_id'])).convert('RGB')
    p = gen_plain(img, r['question']); s = by[i]['prediction']
    if p.strip().lower() == str(s).strip().lower(): same += 1
    else: diffs.append((i, r['question'][:50], s, p))
print(f'일치 {same}/{N_CHECK} ({100 * same / N_CHECK:.0f}%) | {time.time() - t0:.0f}초')
for d in diffs[:10]: print('  #%d Q: %s | 저장: %r | 지금: %r' % d)
REPRO = {'n': N_CHECK, 'same': same}
''')

code(r'''
# 5) 점검 3 — CFG 훅 (논문 식 7·9·10). 팀원 phi3_urp 와 같은 방식: attention 모듈의 positional/keyword 마스크를 모두 처리한다
import re, math, numpy as np
ATTN_BIAS = {'b': None}; EMBED_SCALE = {'s': None}
def _attn_pre(module, args, kwargs):
    b = ATTN_BIAS['b']
    if b is None: return None
    positional = 'attention_mask' not in kwargs
    mask = args[1] if positional and len(args) > 1 else kwargs.get('attention_mask')
    if mask is None or mask.dim() != 4: raise RuntimeError(f'4D additive mask 가 아닙니다: {None if mask is None else tuple(mask.shape)} (eager attention 인지 확인)')
    kv = mask.shape[-1]; bb = b
    if kv > bb.shape[1]: bb = torch.nn.functional.pad(bb, (0, kv - bb.shape[1]))
    mask = mask + bb[:, None, None, :kv].to(device=mask.device, dtype=mask.dtype)
    if positional: args = (args[0], mask, *args[2:])
    else: kwargs['attention_mask'] = mask
    return args, kwargs
def _embed_post(module, args, output):
    s = EMBED_SCALE['s']
    if s is None or output.shape[1] < s.shape[1]: return output
    out = output.clone(); out[:, :s.shape[1]] = out[:, :s.shape[1]] * s[:, :, None].to(device=out.device, dtype=out.dtype); return out
HOOKS = []
attns = [m for n, m in model.named_modules() if re.search(r'layers\.\d+\.self_attn$', n) and 'img_processor' not in n]
embs = [m for n, m in model.named_modules() if n.endswith('vision_embed_tokens')]
assert attns and len(embs) == 1, (len(attns), len(embs))
for m in attns: HOOKS.append(m.register_forward_pre_hook(_attn_pre, with_kwargs=True))
HOOKS.append(embs[0].register_forward_hook(_embed_post))
print(f'훅 설치: attention {len(attns)}개, embedding 1개')

def build(img, question, caption, highlights, alpha=0.01, beta=3.0, delta=None):
    delta = math.log(beta) + 2.0 if delta is None else delta
    prompt = tok.apply_chat_template([{'role': 'user', 'content': (f'{caption} ' if caption else '') + f'<|image_1|>\n{question}'}], tokenize=False, add_generation_prompt=True)
    inp = processor(prompt, [img], return_tensors='pt', qs_highlighted_parts=highlights).to(DEV)
    m = (inp['highlight_attention_mask'].float() == 1).float()[0]
    bias = torch.stack([math.log(beta) * m, -delta * m]); scale = torch.stack([torch.ones_like(m), 1 - (1 - alpha) * m])
    batch = {'input_ids': inp['input_ids'].repeat(2, 1), 'pixel_values': inp['pixel_values'].repeat(2, 1, 1, 1, 1),
             'image_sizes': inp['image_sizes'].repeat(2, 1), 'attention_mask': inp['attention_mask'].repeat(2, 1)}
    return batch, bias, scale, int(m.sum())

@torch.no_grad()
def decode(batch, bias, scale, gamma, max_new=24, use_cache=True):
    ATTN_BIAS['b'] = bias; EMBED_SCALE['s'] = scale
    seq = batch['input_ids']; past = None; new = []
    try:
        for step in range(max_new):
            if use_cache and past is not None:
                out = model(input_ids=seq[:, -1:], attention_mask=torch.ones(2, seq.shape[1], dtype=torch.long, device=DEV),
                            position_ids=torch.full((2, 1), seq.shape[1] - 1, dtype=torch.long, device=DEV), past_key_values=past, use_cache=True)
            else:
                b2 = dict(batch); b2['input_ids'] = seq.clone(); b2['attention_mask'] = torch.ones_like(seq)
                out = model(**b2, use_cache=use_cache)
            past = out.past_key_values if use_cache else None
            lp = torch.log_softmax(out.logits[:, -1].float(), -1)
            nxt = int((gamma * (lp[0] - lp[1]) + lp[1]).argmax())
            if nxt in EOS: break
            new.append(nxt); seq = torch.cat([seq, torch.full((2, 1), nxt, dtype=seq.dtype, device=DEV)], 1)
    finally:
        ATTN_BIAS['b'] = None; EMBED_SCALE['s'] = None
    return new
dec = lambda ids: tok.decode(ids, skip_special_tokens=True).strip().strip('.')

# 하이라이트 토큰이 실제로 잡히는 문항 5개를 고른다 (점검이 의미를 가지려면 1개 이상 필요)
picked = []
for k, r in enumerate(rows[:300]):
    img = Image.open(os.path.join(IMG_ROOT, r['img_id'])).convert('RGB')
    n = build(img, r['question'], r['top_k_captions'][0], r['highlights'])[3]
    if n > 0: picked.append((k, n))
    if len(picked) == 5: break
print('하이라이트가 있는 점검 문항(번호, 토큰 수):', picked)
assert picked, '하이라이트 토큰이 잡히는 문항이 없습니다 — 하이라이트 문자열 매칭을 확인하세요'

neutral_ok = 0; logit_rows = []
for k, n in picked:
    r = rows[k]; img = Image.open(os.path.join(IMG_ROOT, r['img_id'])).convert('RGB'); cap = r['top_k_captions'][0]
    # (a) 중립 설정(α=1, β=1, δ=0, γ=1): 일반 생성(model.generate, 같은 프롬프트)과 같은 문자열이어야 한다
    b, bi, sc, _ = build(img, r['question'], cap, r['highlights'], alpha=1.0, beta=1.0, delta=0.0)
    mine = dec(decode(b, bi, sc, gamma=1.0)); ref = gen_plain(img, r['question'], cap)
    neutral_ok += (mine.lower() == ref.lower())
    if mine.lower() != ref.lower(): print(f'  [중립 불일치] #{k} 훅 루프: {mine!r} | generate: {ref!r}')
    # (b) 논문 설정: 두 가지 logits 가 달라져야 한다
    for name, kw in (('중립', dict(alpha=1.0, beta=1.0, delta=0.0)), ('논문', {})):
        b, bi, sc, n_hl = build(img, r['question'], cap, r['highlights'], **kw)
        ATTN_BIAS['b'] = bi; EMBED_SCALE['s'] = sc
        try: lg = model(**b, use_cache=False).logits[:, -1].float()
        finally: ATTN_BIAS['b'] = None; EMBED_SCALE['s'] = None
        logit_rows.append((k, name, n_hl, float((lg[0] - lg[1]).abs().max())))
print(f'중립 설정이 일반 생성과 같음: {neutral_ok}/{len(picked)}')
for row in logit_rows: print('  #%d %s 설정 | 하이라이트 토큰 %d | 두 가지 마지막 logits 최대 차이 %.4f' % row)
''')

code(r'''
# 6) 점검 4 — KV 캐시 유무 디코딩 일치 (논문 설정 CFG, γ=1.3) + 환경 기록
cache_ok = 0
for k, n in picked[:3]:
    r = rows[k]; img = Image.open(os.path.join(IMG_ROOT, r['img_id'])).convert('RGB')
    b, bi, sc, _ = build(img, r['question'], r['top_k_captions'][0], r['highlights'])
    a = decode(b, bi, sc, gamma=1.3, use_cache=True); c = decode(b, bi, sc, gamma=1.3, use_cache=False)
    cache_ok += (a == c)
    print(f'  #{k} 캐시 {dec(a)!r} | 비캐시 {dec(c)!r} | 토큰 일치 {a == c}')
print(f'KV 캐시 유무 일치: {cache_ok}/3')

META = {'transformers': transformers.__version__, 'peft': peft.__version__, 'torch': torch.__version__, 'numpy': numpy.__version__,
        'gpu': torch.cuda.get_device_name(), 'dtype': str(DTYPE), 'attn': 'eager', 'adapter_load': LOAD_NOTE,
        'baseline_repro': REPRO, 'neutral_equals_generate': f'{neutral_ok}/{len(picked)}', 'paper_cfg_logit_diff': logit_rows,
        'kv_cache_equal': f'{cache_ok}/3'}
json.dump(META, open(f'{E}/smoke_phi35_transformers4.44.2.json', 'w'), ensure_ascii=False, indent=2, default=str)
print(json.dumps(META, ensure_ascii=False, indent=2, default=str))
''')

nb = {'cells': CELLS, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = Path(__file__).resolve().parent / 'smoke_check.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('작성:', out)
