"""phi3/run_teammate_stage.ipynb 를 만든다.
팀원 코드베이스(each-yi/phi3_urp, 고정 커밋)로 RAG · Expert-RAG · Expert-CFG 단계를 돌리고,
RAG 캡션 · 하이라이트 마스킹 · CFG 가 실제로 동작하는지 진단한다.
  python phi3/make_smoke_notebook.py 와 같은 방식: python phi3/make_teammate_stage_notebook.py
선행 조건: 팀원 Phi3V_teamproto_colab.ipynb 로 학습 · val 선택 · test 베이스라인까지 끝나 있어야 한다 (Drive: phi3_teamproto).
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    CELLS.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.strip('\n').splitlines(keepends=True)})


md('''
# 팀원 코드베이스(phi3_urp)로 RAG · Expert-CFG 단계 실행과 진단

Phi-3V-Med. 팀원 레포 `each-yi/phi3_urp`를 **고정 커밋**(cab1984)으로 받아 쓴다. 코드는 수정하지 않는다.

| 단계 | 내용 |
|---|---|
| A | CPU 테스트 — **하이라이트 마스크 정렬, 훅 동작, KV 캐시 일치, 중립 설정, 가이던스 효과** (팀원 `tests/`) |
| B | KB 구축과 검색 (union), 주석 생성 |
| C | **RAG 캡션 진단** — 출처, 같은 이미지 비율, 정답 포함 비율, 예시 |
| D | `rag` · `rag5` · `expert_rag5` · `expert_cfg5` 추론과 평가 |
| E | **하이라이트와 CFG 진단** — 적용 비율, 하이라이트 토큰 수, 답이 바뀐 문항 |

**선행 조건**: 팀원 `Phi3V_teamproto_colab.ipynb`로 학습, val 선택, test 베이스라인이 끝나 있어야 한다 (Drive `phi3_teamproto/work/<데이터셋>/selected.json`, `test_ckpt<step>.jsonl`).

**해석 시 주의**: 팀원 코드는 open 문항의 전문가 주석이 **정답**이라 expert 행은 **규칙 완화**다. KB는 train QA만이고(ROCO · PEIR 이미지가 없어 `--extra`를 쓰지 않음), 두 데이터셋의 train QA가 한 KB에 섞인다.
''')

code(r'''
# 0) 환경 · 팀원 레포(고정 커밋) · 경로
!pip -q install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 huggingface-hub==0.25.2 datasets==2.21.0 open_clip_torch==2.26.1 "numpy<2" scikit-learn
import os, sys, json, glob, re, collections, random, time
import torch, transformers, peft
from pathlib import Path
from google.colab import drive; drive.mount('/content/drive')
print('transformers', transformers.__version__, '| peft', peft.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name() if torch.cuda.is_available() else '없음')
assert transformers.__version__ == '4.44.2', '런타임을 다시 시작한 뒤 다시 실행하세요'

CODE, REV = '/content/phi3_urp', 'cab1984'
if not os.path.isdir(CODE):
    !git clone -q https://github.com/each-yi/phi3_urp {CODE}
!git -C {CODE} checkout -q {REV}
os.chdir(CODE); sys.path.insert(0, CODE)
BASE = '/content/drive/MyDrive/phi3_teamproto'; DATA = f'{BASE}/data'; WORK = f'{BASE}/work'
DATASETS = ['vqa_rad', 'slake']
LIMIT = None                      # 스모크 테스트는 20. 전체는 None
RUN_RAG_ALL = True                # 전 문항 w/ RAG (VQA-RAD 451 / SLAKE 1,061 생성). 오래 걸리면 False
GAMMA = 1.5                       # 팀원 기본값(논문 본문). 논문 Table 7 최고는 1.3 → 아래에서 1.3 도 같이 실행
PREC = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
TAG = f'_lim{LIMIT}' if LIMIT else ''; LIM = f'--limit {LIMIT}' if LIMIT else ''; PARTIAL = '--partial' if LIMIT else ''
print('팀원 레포 커밋:', os.popen(f'git -C {CODE} log --oneline -1').read().strip(), '| precision', PREC)
''')

code(r'''
# A) CPU 테스트 — 하이라이트 마스크 정렬 · 훅 · KV 캐시 일치 · 중립 설정 == 일반 생성 · 가이던스 효과 (팀원 tests/)
!python tests/test_logic.py
from huggingface_hub import snapshot_download
snapshot_download('microsoft/Phi-3-vision-128k-instruct', allow_patterns=['*.json', '*.py', 'tokenizer*'], local_dir='/content/phi3v_small')
!python tests/cpu_smoke.py /content/phi3v_small
# 기대: test_logic 3개 ok, cpu_smoke 마지막 줄 {"ok": true}. 실패하면 4.44.2 에서 팀원 구현이 이 환경과 안 맞는다는 뜻이므로 이후 단계는 의미가 없다
''')

code(r'''
# 선행 파일 확인 (팀원 베이스라인 노트북 결과)
SEL, BASEPRED = {}, {}
for ds in DATASETS:
    sj = f'{WORK}/{ds}/selected{TAG}.json'
    assert os.path.exists(sj), f'없음: {sj} — 팀원 Phi3V_teamproto_colab.ipynb 로 학습·선택·test 베이스라인을 먼저 끝내세요'
    SEL[ds] = json.load(open(sj))['checkpoint']; step = int(SEL[ds].rsplit('-', 1)[1])
    BASEPRED[ds] = f'{WORK}/{ds}/test_ckpt{step}{TAG}.jsonl'
    assert os.path.exists(BASEPRED[ds]), f'없음: {BASEPRED[ds]}'
    assert os.path.exists(f'{DATA}/{ds}/test.jsonl'), f'없음: {DATA}/{ds}/test.jsonl'
    print(f'{ds}: 선택 {os.path.basename(SEL[ds])} | 베이스라인 {os.path.basename(BASEPRED[ds])}')
''')

code(r'''
# B) KB 구축 (최초 1회) 과 검색 — KB = vqa_rad + slake 의 train QA 를 이미지별로 합친 것('no' 답 제외), union 검색(이미지 · 텍스트 유사도의 max)
if not os.path.exists(f'{WORK}/kb/kb_features.pt'):
    !python -m phi3cfg.retrieval build --data-root {DATA} --output {WORK}/kb --datasets vqa_rad slake
for ds in DATASETS:
    ann = f'{WORK}/{ds}/annotations{TAG}.jsonl'
    if not os.path.exists(ann):
        !python -m phi3cfg.retrieval search --kb {WORK}/kb --dataset-dir {DATA}/{ds} --split test --output {ann}
    print(ds, '주석 파일:', ann)
''')

code(r'''
# C) RAG 캡션 진단 (CPU) — 검색된 top-1 캡션이 잘 동작하는가
def rj(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
norm = lambda s: re.sub(r'\s+', ' ', str(s).lower().replace('?', ' ')).strip()
kb = rj(f'{WORK}/kb/kb.jsonl'); print('KB 항목', len(kb), collections.Counter(e['source'] for e in kb))
random.seed(0); DIAG = {}
for ds in DATASETS:
    rows = rj(f'{DATA}/{ds}/test.jsonl'); ann = {a['id']: a for a in rj(f'{WORK}/{ds}/annotations{TAG}.jsonl')}
    src = collections.Counter(); n = len(rows); same = cross = qin = 0; ans_open = [0, 0]; scores = []; exs = []
    for r in rows:
        a = ann[r['id']]; h = a['retrieved'][0]; e = kb[h['kb_index']]
        src[e['source']] += 1; scores.append(h['score'])
        is_same = e['source'] == ds and Path(e['image']).name == r['image']
        same += is_same; cross += (e['source'] != ds)
        qin += norm(r['question']) in norm(h['caption'])
        if r['answer_type'] == 'open':
            ans_open[1] += 1; ans_open[0] += norm(r['answer']) in norm(h['caption'])
        exs.append((r, a, h, is_same))
    kinds = collections.Counter(a['expert_source'] for a in ann.values())
    DIAG[ds] = dict(n=n, top1_source=dict(src), same_image=same / n, other_dataset=cross / n, question_in_caption=qin / n,
                    open_answer_in_top1=ans_open[0] / max(1, ans_open[1]), mean_top1_score=sum(scores) / n, expert_source=dict(kinds))
    print(f'\n=== {ds} (test {n}문항) ===')
    for k, v in DIAG[ds].items(): print(f'  {k}: {v if not isinstance(v, float) else round(v, 3)}')
    print('  (same_image = top-1 이 같은 이미지의 train QA, question_in_caption = 질문 문장이 캡션에 그대로 있음, open_answer_in_top1 = open 정답이 캡션에 있음)')
    print('  예시 8개 (질문 | 정답 | 같은 이미지? | top-1 캡션 앞부분 | 하이라이트):')
    for r, a, h, sm in random.sample(exs, 8):
        print(f"   - Q: {r['question'][:70]} | GT: {r['answer'][:30]} | 같은 이미지: {sm}\n       캡션: {h['caption'][:170]}\n       하이라이트: {a['highlights'][:2]}")
''')

code(r'''
# D) 추론 — rag(전 문항) / rag5 / expert_rag5 / expert_cfg5 (γ 1.5 와 1.3). 모든 출력은 jsonl 에 한 줄씩 저장되어 끊겨도 이어진다
for ds in DATASETS:
    ann = f'{WORK}/{ds}/annotations{TAG}.jsonl'
    c = f'--dataset-dir {DATA}/{ds} --split test --adapter {SEL[ds]} --precision {PREC} --crops 16 {LIM} --annotations {ann}'
    S = f'--select-from {BASEPRED[ds]} --top-percent 5'
    if RUN_RAG_ALL:
        !python -m phi3cfg.infer {c} --mode rag --output {WORK}/{ds}/rag{TAG}.jsonl
    !python -m phi3cfg.infer {c} {S} --mode rag --output {WORK}/{ds}/rag5{TAG}.jsonl
    !python -m phi3cfg.infer {c} {S} --mode expert_rag --output {WORK}/{ds}/expert_rag5{TAG}.jsonl
    !python -m phi3cfg.infer {c} {S} --mode expert_cfg --gamma 1.5 --output {WORK}/{ds}/expert_cfg5{TAG}.jsonl
    !python -m phi3cfg.infer {c} {S} --mode expert_cfg --gamma 1.3 --output {WORK}/{ds}/expert_cfg5_g13{TAG}.jsonl
''')

code(r'''
# 평가 — 논문 Table 1 의 Phi3V-Med 행과 함께 (팀원 evaluate). 베이스라인과 선택되지 않은 문항은 그대로
for ds in DATASETS:
    names = [n for n in ('rag', 'rag5', 'expert_rag5', 'expert_cfg5', 'expert_cfg5_g13') if os.path.exists(f'{WORK}/{ds}/{n}{TAG}.jsonl')]
    runs = ' '.join(f'{n}={WORK}/{ds}/{n}{TAG}.jsonl' for n in names)
    print(f'\n===== {ds} =====')
    !python -m phi3cfg.evaluate --dataset-dir {DATA}/{ds} --split test --baseline {BASEPRED[ds]} --runs {runs} --output {WORK}/{ds}/rag_report{TAG}.json {PARTIAL}
''')

code(r'''
# E) 하이라이트 마스킹과 CFG 진단 — expert_cfg5 출력의 highlight_tokens / cfg_applied 로 실제 적용 여부를 본다
import statistics as st
for ds in DATASETS:
    cfg = rj(f'{WORK}/{ds}/expert_cfg5{TAG}.jsonl'); erag = {r['id']: r for r in rj(f'{WORK}/{ds}/expert_rag5{TAG}.jsonl')}
    base = {r['id']: r for r in rj(BASEPRED[ds])}
    ok = [r for r in cfg if r.get('cfg_applied')]; skipped = [r for r in cfg if not r.get('cfg_applied')]
    byt = lambda rs, t: sum(r['answer_type'] == t for r in rs)
    print(f'\n=== {ds}: 선택 {len(cfg)}문항 (open {byt(cfg, "open")}, closed {byt(cfg, "closed")}) ===')
    print(f'  CFG 가 실제로 적용된 문항: {len(ok)}/{len(cfg)} (open {byt(ok, "open")}, closed {byt(ok, "closed")}) | 가이던스 없이 expert_rag 로 폴백: {len(skipped)} (closed {byt(skipped, "closed")})')
    ht = [r['highlight_tokens'] for r in ok]
    if ht: print(f'  하이라이트 토큰 수: 평균 {st.mean(ht):.1f}, 최소 {min(ht)}, 최대 {max(ht)}, 0개 {sum(h == 0 for h in ht)}')
    ch_rag = [r for r in cfg if r['prediction'].strip() != erag[r['id']]['prediction'].strip()]
    ch_base = [r for r in cfg if r['prediction'].strip() != base[r['id']]['prediction'].strip()]
    print(f'  expert_rag5 와 답이 다른 문항(= CFG 가 바꾼 것): {len(ch_rag)}/{len(cfg)} | 베이스라인과 다른 문항: {len(ch_base)}/{len(cfg)}')
    d = [abs((r.get('entropy') or 0) - (r.get('entropy_normal_branch') or 0)) for r in ok]
    if d: print(f'  가이던스 분포와 정상 가지 엔트로피 차이(절대값) 평균 {st.mean(d):.3f} → 0 에 가까우면 가이던스가 거의 영향이 없다는 뜻')
    print('  CFG 가 답을 바꾼 예 (최대 8개): 질문 | 정답 | expert_rag5 답 | expert_cfg5 답')
    for r in ch_rag[:8]:
        print(f"   - {r['question'][:60]} | {r['answer'][:25]} | {erag[r['id']]['prediction'][:30]} | {r['prediction'][:30]}")
    print('  ※ open 문항의 하이라이트와 참조는 정답에서 만들어진다 → 이 행들은 규칙 완화(논문 방식 모사)이며 규칙 준수 결과로 보고하지 않는다')
''')

code(r'''
# 환경 기록 — 결과와 함께 보관 (results/SCHEMA.md 의 env)
META = dict(transformers=transformers.__version__, peft=peft.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            precision=PREC, teammate_repo=f'each-yi/phi3_urp@{REV}', limit=LIMIT, gamma_values=[1.5, 1.3], caption_diag=DIAG)
json.dump(META, open(f'{WORK}/stage_meta{TAG}.json', 'w'), ensure_ascii=False, indent=2, default=str)
print(json.dumps({k: v for k, v in META.items() if k != 'caption_diag'}, ensure_ascii=False, indent=2))
''')

nb = {'cells': CELLS, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = Path(__file__).resolve().parent / 'run_teammate_stage.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('작성:', out)
