# phi35 — Phi-3.5-vision-instruct 4.2B

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3.5-vision-instruct` |
| Med LoRA | `ecoxial2007/Phi-3.5V-Med` (이어서 학습: r64, lr 1e-4, batch 64, cosine, warmup 0.03) |
| processor | 저자 `Phi-3.5-vision-instruct/processing_phi3_v_cfg.py`, `processor_config.json` → `python tools/fetch_author_assets.py` (저자 레포에 라이선스가 없어 복사하지 않는다) |
| 환경 | `transformers==4.44.2`, 추론은 eager attention, bf16 |
| 논문 목표 (Table 1, Phi3.5V-Med) | VQA-RAD 82.04 / SLAKE 85.20 (baseline overall) |

## 패키지 `phi35cfg/`

| 모듈 | 하는 일 |
|---|---|
| `data.py` | VQA-RAD(`../data_common/vqarad_img/`) · SLAKE 분할, 예측 파일 짝짓기, 이어쓰기 |
| `model.py` | 기반 모델 + 어댑터 + 저자 processor |
| `generation.py` | greedy 생성, Expert-CFG(식 7~10), 식 4 엔트로피 |
| `retrieval.py` | 검색 파일 읽기, 규칙 준수 하이라이트 |
| `train.py` | downstream 학습 |
| `infer.py` | `--mode baseline \| rag \| expert_cfg \| entropy` |
| `evaluate.py` | val 체크포인트 선택(`select`), 결과표(`report`) |

채점 · 게이팅 · 하이라이트 규칙 · 검색 파일 생성은 `../common/`에 있다. 어떤 모드도 정답을 모델 입력에 넣지 않는다.

실험 행: 베이스라인(릴리스된 예측), w/ RAG, w/ 5% RAG, w/ 5% Expert-CFG(γ=1.5, α=0.01, β=3, δ=log β+2). 4.44.2에서 새로 만든 베이스라인은 `baseline-4.44.2-check` 행으로 따로 적는다.

## 실행

Colab: `python tools/pack_phi35_code.py --out <폴더>` → `AIMS_shared/code/`에 올리고 `Phi35V_run.ipynb`를 연다.

GPU 서버(레포가 있는 곳, `phi35/`에서):

```bash
pip install -r requirements.txt
python ../tools/fetch_author_assets.py
python tests/gpu_check.py --dataset vqarad --image-root <이미지> --base-dir <기반 모델> --adapter <어댑터> --retrieval <rag.json> --predictions <test_predictions.json> --out gpu_check.json
python -m phi35cfg.infer --mode expert_cfg --dataset vqarad --image-root <이미지> --base-dir <기반 모델> --adapter <어댑터>     --retrieval <rag.json> --entropy <entropy.json> --top-percent 5 --output cfg5.jsonl
python -m phi35cfg.evaluate report --dataset vqarad --baseline <test_predictions.json> --entropy <entropy.json> --cfg5 cfg5.jsonl --out-dir ../results
```

CPU 테스트(레포 루트): `python phi35/tests/test_data.py` 등 `phi35/tests/test_*.py`.

## 검증 상태

CPU 로직 테스트만 통과했다. **GPU 점검, 4.44.2의 RAG · CFG 점수, 학습 이어하기, 릴리스 베이스라인 재채점(`PHI35_BASELINES` 설정 후 `test_evaluate.py`)은 아직이다.**

## 이전 기록 (실행에 쓰지 않는다)

| 파일 | 내용 |
|---|---|
| `01_Phi35V_baseline_VQARAD_SLAKE.ipynb` | 베이스라인을 만든 노트북 (Expert-CFG 레포에서 이식, `PORTED_FROM.md`) |
| `02_Expert_CFG_RAG_comparison.ipynb` | 이전 RAG · CFG 평가. transformers 5.x 패치와 정답 주입 행을 포함한다. Expert-CFG 레포를 clone한다 |
| `smoke_check.ipynb` | 4.44.2 점검 기록. `tests/gpu_check.py`가 대신한다 |
| `reports/` | 베이스라인 · RAG · CFG 과거 결과, 논문 · 저자 코드 대조(`RAG_CFG_paper_audit.md`) |
