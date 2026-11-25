# phi35cfg Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Phi-3.5V notebook logic into a `phi35/phi35cfg/` package with CLIs, on transformers 4.44.2, reporting only rule-compliant rows with the original scoring.

**Architecture:** The package mirrors `phi3/phi3cfg/` (data, model, generation, retrieval, train, infer, evaluate) and depends only on `common/`. Heavy libraries are imported lazily so every logic test runs on CPU. A thin Colab notebook unpacks a hashed code zip and calls the CLIs.

**Tech Stack:** Python 3.12 (Colab) / 3.14 (local CPU tests), torch, transformers 4.44.2, peft 0.12.0.

**Spec:** `docs/superpowers/specs/2026-11-26-phi35cfg-modularization-design.md`

**Source notebooks** (cell numbers are 0-based): `phi35/01_Phi35V_baseline_VQARAD_SLAKE.ipynb` = **N1**, `phi35/02_Expert_CFG_RAG_comparison.ipynb` = **N2**, `phi35/smoke_check.ipynb` = **S**.

## Global Constraints

- `transformers==4.44.2`, versions from root `requirements.txt`; no `__version__` branches, no DynamicCache / `rope_scaling` / 5.x patches anywhere in the new code.
- No test answer reaches model input, gating, or setting selection. No inject/oracle rows, no S1+S2.
- Scoring is `common.vqa_scoring.score_phi` only. Gating, closed highlights, retrieval-file building exist only in `common/`.
- Imports: `phi35cfg → common` only. Do not import or edit anything under `phi3/`.
- Inference: bf16, eager, `num_crops=16`, `max_new_tokens=64`, greedy. CFG: `gamma=1.5, alpha=0.01, beta=3.0, delta=log(beta)+2`.
- Package modules take paths as arguments; `/content` and `drive.mount` appear only in the notebook.
- `import phi35cfg.<any>` must work with only torch, numpy, Pillow installed (transformers, peft, huggingface_hub imported inside functions).
- Tests: plain scripts (pytest is not installed locally). Each `phi35/tests/test_*.py` inserts `phi35/` into `sys.path` and ends with a `__main__` block that runs every `test_*` function and prints `N tests passed`.
- Do not delete or edit N1, N2, S, the reports, or existing `common/` functions.
- Git: work on branch `phi35cfg-modularization`; stage explicit paths only (the tree has unrelated uncommitted `phi3/` changes). No push. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A run file missing some target questions must make `evaluate` fail, never fall back to baseline silently (Task 6).
2. Resuming into a JSONL with a different configuration fails; a half-written last line from a crash is dropped and redone (Task 1).
3. An entropy file for another dataset/predictions release, or with missing keys, is rejected (Task 4).
4. A baseline or retrieval file whose order or dataset differs from the data split is rejected by question/answer check (Tasks 1, 3).
5. No highlight found → same output path as RAG with `cfg_applied=False`; hooks are cleared even when the forward raises (Task 2).

## File Structure

| File | Responsibility |
|---|---|
| `phi35/phi35cfg/__init__.py` | puts repo root on `sys.path` so `common` imports |
| `phi35/phi35cfg/data.py` | splits, prediction files, pairing, resumable run files |
| `phi35/phi35cfg/generation.py` | prompt, greedy loop, CFG hooks, eq.4 entropy |
| `phi35/phi35cfg/retrieval.py` | read retrieval file, compliant highlights |
| `phi35/phi35cfg/model.py` | base dir, author processor, adapter, env record |
| `phi35/phi35cfg/infer.py` | CLI: baseline / rag / expert_cfg / entropy |
| `phi35/phi35cfg/evaluate.py` | CLI: select checkpoint, report table |
| `phi35/phi35cfg/train.py` | CLI: downstream fine-tuning |
| `phi35/tests/test_*.py`, `gpu_check.py` | CPU logic tests, GPU verification |
| `tools/pack_phi35_code.py`, `phi35/Phi35V_run.ipynb` | Colab delivery and execution |

---

### Task 1: Data and run files

**Files:** Create `phi35/phi35cfg/__init__.py`, `phi35/phi35cfg/data.py`, `phi35/tests/test_data.py`

**Interfaces — Produces:**
- Row: `{'idx': int, 'image': str, 'question': str, 'answer': str, 'answer_type': 'OPEN'|'CLOSED'}` (question/answer stripped, idx = position in split).
- `load_split(dataset: str, split: str, image_root: str|None = None, data_root: str|None = None) -> tuple[list[dict], Path]` — returns rows and image folder. `vqarad`: `data_common/vqarad_img/{trainset,valset,testset}.json`, `image_root` required. `slake`: HF `BoKelvin/SLAKE` `{train,validation,test}.json`, `q_lang == 'en'`, `imgs.zip` extracted under `data_root`. Missing images raise.
- `to_rows(raw: list[dict]) -> list[dict]` (accepts `image_name`/`img_name`/`img_id`).
- `read_predictions(path) -> dict[int, dict]` — released JSON list or JSONL; duplicate `idx` raises.
- `align(rows, records: list[dict], name: str) -> list[dict]` — records ordered by `idx` if all have it, else file order; length and lower-stripped `(question, answer)` equality asserted for every row, `ValueError` naming `name`.
- `open_run(output: Path, config: dict) -> set[int]` — writes `<output>.meta.json` first time; raises `ValueError` if stored config differs (key `runtime` ignored); truncates an unparsable final line; returns done idx.
- `append_record(output: Path, record: dict) -> None` (flushes).
- `file_sha256(path) -> str`

- [ ] **Step 1:** `git switch -c phi35cfg-modularization`
- [ ] **Step 2: Write failing tests**

```python
def test_vqarad_counts():          # real data_common files; image check disabled by creating empty files in a temp image_root
    rows, _ = load_split('vqarad', 'test', image_root=tmp)
    assert len(rows) == 451 and sum(r['answer_type'] == 'CLOSED' for r in rows) == 272
    assert [r['idx'] for r in rows[:3]] == [0, 1, 2]          # train 2749, val 315 checked the same way

def test_align_rejects_mismatch():  # shuffled records without idx, wrong length, one changed answer -> ValueError
def test_align_uses_idx():          # shuffled records with idx come back in row order
def test_open_run_resume():
    assert open_run(out, {'mode': 'rag'}) == set()
    append_record(out, {'idx': 0}); out.open('a').write('{"idx": 1, "pre')
    assert open_run(out, {'mode': 'rag'}) == {0}
    assert out.read_text(encoding='utf-8').endswith('}\n')
    with raises(ValueError): open_run(out, {'mode': 'baseline'})
```
(`raises` is a small local context manager; reuse it in later tests.)

- [ ] **Step 3:** Run `python phi35/tests/test_data.py` — expect `ModuleNotFoundError`.
- [ ] **Step 4:** Implement. Count assertions from N1 cell 10: vqarad 2749/315/451, test 272/179; slake test 1061 = open 645 + closed 416.
- [ ] **Step 5:** Run again — all pass. Commit `feat(phi35cfg): data loading and resumable run files`.

### Task 2: Generation and CFG hooks

**Files:** Create `phi35/phi35cfg/generation.py`, `phi35/tests/test_generation.py`

**Interfaces — Produces:**
- `@dataclass CFGConfig(gamma=1.5, alpha=0.01, beta=3.0, delta=None)`; `delta` defaults to `log(beta)+2`; `ValueError` unless `gamma >= 1, beta > 0, 0 <= alpha <= 1`.
- `build_prompt(tokenizer, question: str, reference: str|None = None) -> str` — content `f'{reference} <|image_1|>\n{question}'` or `f'<|image_1|>\n{question}'`, `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
- `guided_logp(logp: Tensor, gamma: float) -> Tensor` — `log_softmax(gamma*logp[0] - (gamma-1)*logp[1])`.
- `class CFGHooks(model)` with `set(embed_scale, attn_bias)`, `clear()`, `remove()`; `get_hooks(model) -> CFGHooks` (one per model). Targets: modules matching `layers\.\d+\.self_attn$` without `img_processor`, exactly one `vision_embed_tokens`; else `RuntimeError`. Behaviour copied from S cell 5.
- `generate(model, processor, image, question, reference=None, highlights=None, cfg: CFGConfig|None=None, max_new_tokens=64, use_cache=True) -> {'prediction': str, 'cfg_applied': bool, 'highlight_tokens': int}`
- `answer_entropy(model, processor, image, question, prediction: str) -> float|None`

Rules: processor call is `processor(prompt, [image], return_tensors='pt', qs_highlighted_parts=highlights or [])`, mask is `highlight_attention_mask == 1`. If `cfg` is given and the mask has ≥1 token → two-branch batch (`bias = [log(beta)*m, -delta*m]`, `scale = [1, 1-(1-alpha)*m]`), else single branch with no hooks. Loop as S cell 5 `decode` (explicit `position_ids` on cached steps; clone `input_ids` on full forwards). Stop ids: `eos_token_id` and `<|end|>`, not appended. Prediction: `batch_decode(skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().strip('.')`. Entropy: teacher-force `tok(prediction, add_special_tokens=False)`, vocabulary entropy at `logits[L-1 : L-1+n]` in float32, `common.gating.aggregate_entropy` excluding `tok.all_special_ids`; empty answer → `None`.

- [ ] **Step 1: Write failing tests** using a fake tokenizer/processor and a tiny torch model with `layers.0.self_attn`, `vision_embed_tokens`, recording the masks and embeddings it receives.

```python
def test_cfg_config_delta():  assert abs(CFGConfig().delta - (math.log(3) + 2)) < 1e-12; CFGConfig(gamma=0.5) raises
def test_guided_logp():       # gamma=1 equals logp[0]; argmax equals argmax of gamma*(lp0-lp1)+lp1
def test_hooks_bias_and_scale():  # mask via kwarg AND positional both receive bias; kv longer than prefix padded with 0;
                                  # 1-token step output is not scaled; non-4D mask raises
def test_hooks_cleared_on_error():  # forward raises inside generate -> hooks.embed_scale is None and attn_bias is None
def test_no_highlight_is_single_branch():
    r = generate(model, proc, img, 'q', 'ref', ['absent'], CFGConfig())
    assert r['cfg_applied'] is False and r['highlight_tokens'] == 0 and model.batch_sizes[0] == 1
def test_cache_and_nocache_same():  # identical token output on the fake model
def test_generate_takes_no_answer(): assert 'answer' not in inspect.signature(generate).parameters
def test_entropy_excludes_special(): # fake logits with known entropy; special answer token ignored; '' -> None
```

- [ ] **Step 2:** Run `python phi35/tests/test_generation.py` — fails on import.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run — pass. Commit `feat(phi35cfg): greedy generation, eq.4 entropy and CFG hooks`.

### Task 3: Retrieval

**Files:** Create `phi35/phi35cfg/retrieval.py`, `phi35/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `data.align`, `common.highlights.closed_sentence_highlights`.
- Produces: `load_retrieval(path, rows) -> list[dict]` (each `{'reference': top_k_captions[0], 'file_highlights': highlights}`; also requires `img_id == row['image']`); `compliant_highlights(question: str, answer_type: str, reference: str, file_highlights: list[str]) -> list[str]` (CLOSED → sentence rule, OPEN → `file_highlights`).

- [ ] **Step 1: Write failing tests**

```python
def test_highlights_have_no_answer_arg(): assert 'answer' not in inspect.signature(compliant_highlights).parameters
def test_matches_notebook():
    # exec the N2 cell-18 source from '_STOP = set' up to 'HL = {' (needs re in namespace);
    # for all 451 testset questions with caption = f"{question} yes. Another finding in the {image_organ}. {question}"
    assert compliant_highlights(q, t, cap, ['x']) == ref_compliant_highlights({'question': q, 'answer_type': t, 'top_k_captions': [cap], 'highlights': ['x']})
def test_load_rejects_wrong_dataset():  # wrong img_id or reordered file -> ValueError
```
- [ ] **Step 2:** Run — fails. **Step 3:** Implement. **Step 4:** Run — pass. Commit `feat(phi35cfg): retrieval file loading and compliant highlights`.

### Task 4: Model loading, inference CLI, author assets

**Files:** Create `phi35/phi35cfg/model.py`, `phi35/phi35cfg/infer.py`, `phi35/tests/test_infer.py`; Modify `tools/fetch_author_assets.py`

**Interfaces:**
- Consumes: Tasks 1–3, `common.gating.select_top_percent`.
- `model.BASE_MODEL = 'microsoft/Phi-3.5-vision-instruct'`, `MED_ADAPTER = 'ecoxial2007/Phi-3.5V-Med'`, `ASSETS = <repo>/phi35/assets`
- `prepare_base_dir(base_dir) -> Path` — snapshot_download if no `config.json`, then **always** copy the author `processing_phi3_v_cfg.py` and `processor_config.json` over it; adds `ASSETS/author_src` to `sys.path`; missing assets raise with the fetch command.
- `enable_sdpa(base_dir) -> None` — idempotent `_supports_sdpa = False → True` in `modeling_phi3_v.py` (N1 cell 15).
- `resolve_adapter(name_or_path) -> str` — local dir, or HF snapshot then the folder holding `adapter_config.json`.
- `load(base_dir, adapter, num_crops=16, attn='eager', train=False) -> (model, processor, meta)` — bf16, `torch_dtype=`, `_attn_implementation=attn`, `trust_remote_code=True`; `attn='sdpa'` only with `train=True`; no silent fallback. `generation_config` from base dir (S/N2 cell 13). `meta` = base, adapter, precision, attn, num_crops, gpu, versions of torch/transformers/peft/accelerate.
- `infer.build_parser()`, `infer.main(argv=None)`, `infer.select_ids(entropy: dict[int, float|None], n: int, percent=5.0) -> list[int]`, `infer.load_entropy(path, dataset: str, n: int, predictions: str|None) -> dict[int, float|None]`

CLI: `--mode baseline|rag|expert_cfg|entropy --dataset vqarad|slake --split (default test) --image-root --data-root --base-dir --adapter (required) --output --retrieval --entropy --top-percent --predictions --predictions-name --adapter-name --gamma 1.5 --alpha 0.01 --beta 3.0 --delta --limit`.
Rules: rag/expert_cfg need `--retrieval`; `--top-percent` needs `--entropy`; expert_cfg requires `--top-percent`. rag without `--top-percent` = all questions. baseline records `entropy = answer_entropy(...)`. Records per spec 6.6; meta config includes every argument except `limit`, plus sha256 of retrieval/entropy files and adapter weights if local. `--mode entropy` reads `--predictions` (aligned), saves the UNCERTAINTY_SPEC §5 JSON every 100 items and resumes from it only if dataset/adapter/predictions names match.
`load_entropy`: `dataset` must match, keys must be exactly `0..n-1`, and `predictions` must match when given.

- [ ] **Step 1: Write failing tests** (monkeypatch `model.load` and `generation.generate`/`answer_entropy` with fakes)

```python
def test_select_ids():  assert len(select_ids({i: .1 for i in range(451)}, 451)) == 23; 1061 -> 53; ties -> low idx; None skipped
def test_load_entropy_rejects():  # wrong dataset, missing key, wrong predictions name -> ValueError
def test_cli_requires():  # expert_cfg without --top-percent, rag without --retrieval -> SystemExit
def test_expert_cfg_run():  # only selected idx generated; generate never receives the answer; second run generates 0
def test_prepare_base_dir_overwrites_processor():  # stale processor_config.json replaced by asset copy
def test_enable_sdpa_idempotent():
```
- [ ] **Step 2:** Run — fails.
- [ ] **Step 3:** In `fetch_author_assets.py` add `src/highlighter_modules/utils.py` from the same `REVISION` to `phi35/assets/author_src/src/highlighter_modules/utils.py`, plus an empty `author_src/src/__init__.py`; `--check` covers them.
- [ ] **Step 4:** Implement `model.py`, `infer.py`.
- [ ] **Step 5:** Run tests — pass. Run `python tools/fetch_author_assets.py` then `--check` — every line starts `있음`. Commit `feat(phi35cfg): model loading and inference CLI`.

### Task 5: Training

**Files:** Create `phi35/phi35cfg/train.py`, `phi35/tests/test_train.py`

**Interfaces:**
- Consumes: `data.load_split`, `model.load(train=True)`, `model.enable_sdpa`, `generation.build_prompt`.
- Produces: `build_parser()`, `main(argv=None)`, `resolve_epochs(dataset, epochs: int|None, save_epochs: str|None) -> (int, set[int])`, `make_collator(processor, image_dir) -> callable`.

CLI defaults (N1 cells 6, 17, 18): `--adapter` = `MED_ADAPTER`, `--lr 1e-4 --batch 64` (micro-batch 1, accumulation 64) `--warmup-ratio 0.03 --weight-decay 0.0 --max-grad-norm 1.0 --seed 42 --attn eager|sdpa --grad-ckpt --resume --max-steps`; plus dataset/path args as in `infer`. bf16 only.
Trainer: cosine, `adamw_torch`, `save_strategy='epoch'` with save only at chosen epochs, `save_total_limit=None`, `save_only_model=False`, `remove_unused_columns=False`, `dataloader_num_workers=0`, `logging_steps=10`, `report_to='none'`. `load(train=True)` must give `is_trainable=True`, vision frozen, LoRA rank asserted 64. `--grad-ckpt`: reentrant, `enable_input_require_grads()` and the `vision_embed_tokens` output grad hook. `run_config.json` written; `--resume` refuses if it differs, takes the highest `checkpoint-*`, and applies `torch.load(weights_only=False)`.
Collator: answer text `f"{answer}<|end|>\n<|endoftext|>"`, prompt labels `-100`, no `attention_mask`, no `highlight_attention_mask`, exactly one example.

- [ ] **Step 1: Write failing tests**

```python
def test_resolve_epochs():
    assert resolve_epochs('vqarad', None, None) == (10, {2,4,6,8,10})
    assert resolve_epochs('slake', None, None) == (5, {1,2,3,4,5})
    # '0' or '11' in save_epochs -> ValueError
def test_collator():  # fake processor: labels == [-100]*L + answer ids; keys exclude both masks; 2 examples -> error
def test_defaults(): a = build_parser().parse_args([...required]); assert (a.lr, a.batch, a.weight_decay, a.attn) == (1e-4, 64, 0.0, 'eager')
```
- [ ] **Step 2:** Run — fails. **Step 3:** Implement. **Step 4:** Run — pass. Commit `feat(phi35cfg): downstream training CLI`.

### Task 6: Evaluation

**Files:** Create `phi35/phi35cfg/evaluate.py`, `phi35/tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 readers, `infer.load_entropy`, `infer.select_ids`, `common.vqa_scoring.{score_phi, summarize}`, `common.gating.auc_wrong_vs_entropy`.
- Produces: `select_checkpoint(scores: dict[int, float]) -> int` (max overall, tie → smaller step); `merge(base: dict, run: dict, targets: set[int]) -> dict` (`ValueError` unless run idx == targets); `paired_ci(new, ref, n=10000, seed=0) -> (lo, hi)` in points; `PAPER`; `build_parser()`, `main(argv=None)`.
- CLI: `select --dataset … --run STEP=path …` (val predictions; prints and writes chosen step; refuses `--split test`); `report --dataset … --baseline path [--entropy path --rag100 path --rag5 path --cfg5 path --check-baseline path] --out-dir results [--checkpoint-name]`.

Row names `baseline`, `RAG-100`, `RAG-5`, `ExpertCFG-5`, `baseline-4.44.2-check` (separate row, never used for merging). Each saved file follows `results/SCHEMA.md` with `model='phi35'`, `rule='compliant'`, `scoring='phi'`, env and cfg from the run's `.meta.json`. Targets: RAG-100 all idx; 5% rows `select_ids(entropy)`. `PAPER` (open, closed, overall) from N2 cell 20 for baseline, rag, rag5, expert_cfg; print "원문 대조 필요" under the table.

- [ ] **Step 1: Write failing tests**

```python
def test_select_checkpoint(): assert select_checkpoint({100: 70.0, 200: 71.0, 300: 71.0}) == 200
def test_merge_requires_exact_targets():  # one missing, one extra -> ValueError
def test_report_rows():  # 4-question fixture: expected open/closed/overall, result JSON has all SCHEMA required keys
def test_no_s1s2():  assert 's1s2' not in Path(evaluate.__file__).read_text(encoding='utf-8').lower()
def test_released_baselines():  # only if env PHI35_BASELINES points to the two released folders, else print 'skipped'
    assert round(overall_vqarad, 2) == 79.59 and round(overall_slake, 2) == 85.63
```
- [ ] **Step 2:** Run — fails. **Step 3:** Implement. **Step 4:** Run — pass. Ask the user for the two released baseline folders and run the last test with them; if unavailable, report it as not verified. Commit `feat(phi35cfg): checkpoint selection and report`.

### Task 7: GPU check, packaging, notebook, docs

**Files:** Create `phi35/tests/gpu_check.py`, `tools/pack_phi35_code.py`, `phi35/Phi35V_run.ipynb`, `phi35/tests/test_delivery.py`; Modify `README.md`, `phi35/README.md`, `docs/RULES.md`, `docs/UNCERTAINTY_SPEC.md`

**Interfaces:**
- Consumes: every CLI's `build_parser()`, `generation`, `model.load`.
- `gpu_check.py` args `--dataset --base-dir --adapter --image-root --data-root --retrieval --predictions --out --full`: neutral `CFGConfig(gamma=1, alpha=1, beta=1, delta=0)` equals `model.generate` on 5 highlighted items; paper config changes last logits; cache == no cache (3 items, 20 with `--full`); baseline reproduction (50, all with `--full`), fails above 2% mismatch only with `--full`. Writes JSON, exit 1 on failure.
- `pack_phi35_code.py --out <dir>` → `<dir>/phi35-code.zip` containing `common/`, `phi35/phi35cfg/`, `phi35/tests/`, `data_common/vqarad_img/`, `tools/{fetch_author_assets,release_manifest}.py`, `manifests/`, both requirements files; excludes `__pycache__`, `assets/`.
- Notebook order: install pins (S cell 1) → mount Drive, copy release folder, `release_manifest.py verify`, unzip → fetch assets → gpu_check → train (flagged optional) → val inference and `select` (optional) → entropy → baseline check, RAG-100, RAG-5, ExpertCFG-5 → report. Retrieval files built with `common/build_rag_testjson.py` or taken from `AIMS_shared/rag_assets/`. One `DATASET` setting.

- [ ] **Step 1: Write failing tests**

```python
def test_zip_contents():   # required members present; no __pycache__, no 'assets/'
def test_notebook_clean(): # no outputs; no 'Expert-CFG.git'
def test_notebook_commands_parse():  # every '-m phi35cfg.X' command line: placeholders filled, X.build_parser().parse_args succeeds
def test_no_version_branches():  # phi35cfg sources contain none of '__version__', 'DynamicCache', 'rope_scaling'
```
- [ ] **Step 2:** Run — fails. **Step 3:** Implement the script, check, notebook.
- [ ] **Step 4:** Docs per spec 6.8: RULES gets "교수님 지침(2026-11-26): 정답 주입 결과를 내지 않는다. S1+S2도 쓰지 않는다."; root README drops answer-injection rows as primary comparison and marks old numbers as 5.x history; phi35 README documents the new commands and marks N1/N2 as records; UNCERTAINTY_SPEC §4 points to `generation.answer_entropy`.
- [ ] **Step 5: Verify all**

Run: `for f in phi35/tests/test_*.py tests/test_common.py; do python $f || exit 1; done; python common/vqa_scoring.py; git status --short`
Expected: every script passes; no `phi3/` file staged. Commit `feat(phi35): GPU check, code packaging, run notebook and docs`.

---

## After the plan: user-run GPU verification (spec §7, items 5–8)

Not claimable as done until the user runs them on Colab (4.44.2) and shares output: `gpu_check.py` on both datasets, the four-row table on both datasets, a `--max-steps` training run followed by `--resume`.
