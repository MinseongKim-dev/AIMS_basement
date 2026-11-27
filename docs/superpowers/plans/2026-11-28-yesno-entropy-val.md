# yes/no 불확실성 val 비교 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** yes/no 문항에서 엔트로피 정의 D0 · D1 · D2의 AUC를 val에서 비교한다. 공식 정의는 D0 그대로다.

**Architecture:** `phi35cfg/yesno.py`(순수 계산), `infer --mode yesno`(문항당 forward 1번, 원시 확률 저장), `evaluate yesno`(CPU 보고). 기존 JSONL 이어하기(`open_run`)와 문항 짝 검증(`check_records`)을 그대로 쓴다.

**Tech Stack:** Python, PyTorch, transformers 4.44.2, scikit-learn, 평문 테스트 스크립트(pytest 없음).

**Spec:** `docs/superpowers/specs/2026-11-28-yesno-entropy-val-design.md`

## Global Constraints
- 정답은 모델 입력에 넣지 않는다. 정답은 yes/no 문항을 **고르고 채점**할 때만 쓴다.
- 모델: 4.44.2, bf16, eager (`model_lib.load`). 버전 분기(`__version__` 등)를 넣지 않는다(`test_delivery.test_no_version_branches`).
- 오답: `score_phi` < 1. 부트스트랩 10000회, seed 0, 95% 구간.
- 공식 정의는 D0. 채택 판정과 test 계산을 만들지 않는다. `yesno` 모드와 보고는 val 전용. `docs/UNCERTAINTY_SPEC.md`와 게이팅(`select_ids`)은 바꾸지 않는다.
- 브랜치 `phi35cfg-modularization`. `git add`는 파일을 지정한다(`-A` 금지). 푸시는 하지 않는다.
- 커밋 메시지 끝: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`

## Review Focus
- 빈 답이라 D0가 None인 문항: 세 정의 모두에서 같은 문항을 빼고 개수를 보고한다 (Task 1).
- 예측 파일과 확률 파일의 split · 어댑터가 다름: 거부한다 (Task 3).
- 중간에 끊긴 확률 파일(yes/no 문항 일부 없음): 점수를 내지 않고 거부한다 (Task 3).
- 오답이 없어 AUC가 계산되지 않음: 오류 없이 AUC를 비워 둔다 (Task 1).
- `yesno` 모드를 test에서 실행: 거부한다 (Task 2).

---

### Task 1: 정의 (순수 계산)

**Files:** Create `phi35/phi35cfg/yesno.py`, `phi35/tests/test_yesno.py`

**Interfaces (Produces):**
- `LABELS = ('yes', 'no')`
- `label_of(answer: str) -> str | None` — `strip().lower()`가 LABELS에 있으면 그 값
- `label_token_ids(tokenizer) -> {'yes': [int], 'no': [int]}` — `range(len(tokenizer))` 중 `tokenizer.decode([i]).strip().lower()`가 일치하는 id. 한쪽이라도 비면 `ValueError`
- `probe_record(logp: Tensor[V], ids: dict) -> {'entropy_first': float, 'label_probs': {'yes': {str(id): p}, 'no': {...}}}`
- `merged_entropy(entropy_first, label_probs) -> float` (D1) = `H + Σ_g (Σ_{i∈g} p_i ln p_i − P_g ln P_g)`, p=0 항은 0
- `binary_entropy(label_probs) -> float | None` (D2) — P_yes+P_no == 0 이면 None
- `compare(items, n_boot=10000, seed=0) -> {'n', 'n_wrong', 'n_excluded', 'auc': {'D0','D1','D2'}, 'diff': {'D1': {'value', 'ci95': [lo, hi]}, 'D2': {...}}}` — item = `{'wrong': 0|1, 'D0': float|None, 'D1': ..., 'D2': ...}`. 부트스트랩은 문항을 짝지어 재표집하고 한 클래스뿐인 표본은 건너뛴다. AUC는 `common.gating.auc_wrong_vs_entropy`. AUC가 None이면 `diff`의 값도 None

- [ ] **Step 1: 실패하는 테스트** (`from _util import raises, run` 방식, 파일 끝 `run(globals())`)

```python
def test_definitions():
    lp = {'yes': {'1': .5, '2': .5}, 'no': {'3': 0.0}}
    assert abs(yesno.merged_entropy(math.log(2), lp)) < 1e-9
    two = {'yes': {'1': .3, '2': .15}, 'no': {'3': .05}}
    assert abs(yesno.binary_entropy(two) - 0.325083) < 1e-5
    assert yesno.binary_entropy({'yes': {'1': 0.0}, 'no': {'3': 0.0}}) is None
    assert [yesno.label_of(a) for a in (' Yes', 'NO', 'yes it is', 'left')] == ['yes', 'no', None, None]

def test_token_ids():   # Tok: len 5, decode -> [' Yes', 'yes', 'No', 'maybe', '<s>']
    assert yesno.label_token_ids(Tok()) == {'yes': [0, 1], 'no': [2]}
    with raises(ValueError):
        yesno.label_token_ids(TokNoNo())

def test_probe_record():
    p = torch.tensor([.4, .3, .2, .1])
    r = yesno.probe_record(p.log(), {'yes': [0, 1], 'no': [2]})
    assert abs(r['entropy_first'] + float((p * p.log()).sum())) < 1e-6
    assert abs(r['label_probs']['yes']['1'] - .3) < 1e-6

def test_compare_and_missing():
    d0, d1 = [.1, .5, .3, .2, .4, .6], [.1, .2, .3, .4, .5, .6]
    items = [{'wrong': int(i >= 3), 'D0': d0[i], 'D1': d1[i], 'D2': d0[i]} for i in range(6)]
    items.append({'wrong': 1, 'D0': None, 'D1': .0, 'D2': .0})
    c = yesno.compare(items, n_boot=500)
    assert (c['n'], c['n_wrong'], c['n_excluded']) == (6, 3, 1)
    assert abs(c['auc']['D0'] - 6 / 9) < 1e-9 and c['auc']['D1'] == 1.0
    assert c['diff']['D2'] == {'value': 0.0, 'ci95': [0.0, 0.0]}
    one = yesno.compare([{'wrong': 0, 'D0': .1, 'D1': .1, 'D2': .1}], n_boot=10)
    assert one['auc']['D0'] is None
    assert one['diff']['D1']['value'] is None

```

- [ ] **Step 2:** `python phi35/tests/test_yesno.py` → `ImportError`(yesno 없음)로 실패
- [ ] **Step 3:** `yesno.py` 구현
- [ ] **Step 4:** 같은 명령 → `4 tests passed`
- [ ] **Step 5:** 두 파일 커밋 `feat(phi35cfg): yes/no entropy definitions`

### Task 2: `infer --mode yesno`

**Files:** Modify `phi35/phi35cfg/generation.py`, `phi35/phi35cfg/infer.py`; Test `phi35/tests/test_infer.py`

**Interfaces:**
- Consumes: `yesno.label_of`, `label_token_ids`, `probe_record`
- Produces: `generation.first_token_logp(model, processor, image, question) -> Tensor[V]` — `build_prompt(tok, question)`로 forward 1번(`use_cache=False`), 마지막 프롬프트 위치의 `float().log_softmax(-1)`
- Produces: `infer --mode yesno` — `label_of(answer)`가 있는 문항만 처리. JSONL 줄 `{'idx','question','answer','answer_type','mode':'yesno','entropy_first','label_probs'}`. 설정 기록과 이어하기는 다른 모드와 같은 `open_run`(`split`, `adapter_sha256` 포함). `--retrieval`, `--predictions` 불필요. `--split val`이 아니면 parser error

- [ ] **Step 1: 실패하는 테스트** (`test_infer.py`에 추가. `model_lib.load`는 `.tokenizer`를 가진 가짜 processor를 돌려주고 `yesno.label_token_ids`와 `generation.first_token_logp`를 patch)

```python
def test_yesno_mode_runs_only_yes_no_questions_and_resumes():
    # val split(315문항, yes/no 184), --limit 3 으로 한 번, limit 없이 한 번 더. --split test 는 SystemExit
    assert len(first) == 3 and len(all_lines) == 184 and len({l['idx'] for l in all_lines}) == 184
    assert all(yesno.label_of(l['answer']) for l in all_lines)
    assert set(all_lines[0]) == {'idx', 'question', 'answer', 'answer_type', 'mode', 'entropy_first', 'label_probs'}
    assert json.loads(meta.read_text(encoding='utf-8'))['split'] == 'val'
    assert all(set(c) == {'question'} for c in calls)          # 입력은 질문뿐
```

그리고 `_fakes.Model`로 `first_token_logp`를 직접 확인한다: 반환 shape `(V,)`, `exp().sum()≈1`, `argmax()==2`, forward 1번.

- [ ] **Step 2:** `python phi35/tests/test_infer.py` → `invalid choice: 'yesno'` 또는 속성 없음으로 실패
- [ ] **Step 3:** 구현. `main`의 설정 기록 부분은 baseline 흐름과 공유한다(복제하지 않음). 모듈 docstring에 모드 한 줄 추가
- [ ] **Step 4:** `test_infer.py`, `test_generation.py` 통과
- [ ] **Step 5:** 커밋 `feat(phi35cfg): yesno inference mode storing first-token label probabilities`

### Task 3: `evaluate yesno`

**Files:** Modify `phi35/phi35cfg/evaluate.py`; Test `phi35/tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 전체, Task 2의 파일 형식, `_read_meta`, `check_records`
- Produces: `yesno --vqarad PREDS PROBS --slake PREDS PROBS [--data-root] --out report.json` (split은 val 고정, 두 데이터셋 모두 필요).
  파일 내용: `{'split': 'val', 'official': 'D0', 'datasets': {name: compare결과 + 'adapter_sha256'}}`.
  검증: PREDS의 `mode`가 `baseline`, 두 meta의 `split == 'val'`, `adapter_sha256` 동일, PREDS는 split 전 문항, PROBS의 idx 집합 == yes/no 문항 집합
  - item: `wrong = score_phi(pred) < 1`, `D0 = pred['entropy']`, D1 · D2는 PROBS에서 계산

- [ ] **Step 1: 실패하는 테스트** (`test_evaluate.py`에 추가. VQA-RAD는 실제 split, SLAKE는 `load_split`을 patch한 작은 split. 확률은 D1이 오답에서 크게 만든다)

```python
def test_yesno_report():
    assert v['split'] == 'val' and v['official'] == 'D0' and 'adopted' not in v
    assert set(v['datasets']) == {'vqarad', 'slake'} and v['datasets']['vqarad']['n'] == 184
    assert v['datasets']['vqarad']['auc']['D1'] == 1.0

def test_yesno_refusals():
    # 각각 raises(ValueError), 그리고 --out 파일이 생기지 않는다
    #  PROBS 마지막 줄 삭제 / PREDS meta split='test' / adapter_sha256 불일치 / PREDS mode='rag'
```

- [ ] **Step 2:** `python phi35/tests/test_evaluate.py` → `invalid choice: 'yesno'`로 실패
- [ ] **Step 3:** 구현. 모든 입력을 검증한 뒤에 파일을 쓴다. 표 출력: 데이터셋별 n · 오답 · 제외 · 세 AUC · 차이 [구간], "official: D0"
- [ ] **Step 4:** `test_evaluate.py` 통과
- [ ] **Step 5:** 커밋 `feat(phi35cfg): yes/no entropy comparison on val`

### Task 4: 실행 노트북 · 문서

**Files:** Modify `phi35/Phi35V_run.ipynb`, `phi35/README.md`; Test `phi35/tests/test_delivery.py`

- [ ] **Step 1:** 작업 폴더의 `Phi35V_run.ipynb`에는 사용자의 Colab 출력이 있다. `phi35/Phi35V_run_executed_local.ipynb`(git 무시 대상)로 **복사해 보관**하고, 파일이 같은지 확인한 뒤 `git checkout -- phi35/Phi35V_run.ipynb`
- [ ] **Step 2: 실패하는 테스트** — `test_notebook_commands_parse`의 기대 집합에 `'infer:yesno:all'`, `'evaluate:yesno'` 추가 → 실행해 실패 확인
- [ ] **Step 3:** 끝에 "8. yes/no 불확실성 비교" 절 추가
  - 설정 셀: `YESNO_ROOT = '/content/drive/MyDrive/phi35cfg_runs'`
  - val 셀(현재 `DATASET`): `--mode baseline --split val` → `{OUT}/val_release.jsonl`, `--mode yesno --split val` → `{OUT}/yesno_val.jsonl`
  - 보고 셀(두 데이터셋의 val 파일이 모두 있을 때): `evaluate yesno ... --out "{YESNO_ROOT}/yesno_report.json"`
  - 출력 없이 저장
- [ ] **Step 4:** README의 실험 설명에 두 문장: 목적과 "공식 정의는 D0이며 D1 · D2는 val 확인용이다"
- [ ] **Step 5:** `phi35/tests/test_*.py` 전부와 레포 루트의 `python tests/test_common.py` 통과
- [ ] **Step 6:** 설계 · 계획 문서와 이 Task의 파일을 커밋 `feat(phi35): notebook cells for the yes/no entropy comparison`
