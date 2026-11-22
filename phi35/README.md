# phi35 — Phi-3.5-vision-instruct 4.2B

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3.5-vision-instruct` |
| Med LoRA | `ecoxial2007/Phi-3.5V-Med` (이어서 학습: r64, lr 1e-4, batch 64, cosine, warmup 0.03) |
| processor | 저자 `Phi-3.5-vision-instruct/processing_phi3_v_cfg.py`, `processor_config.json` → `python tools/fetch_author_assets.py` (저자 레포에 라이선스가 없어 복사하지 않는다) |
| 환경 | `transformers==4.44.2`, eager attention. 모델 · 훅 · KV 캐시는 두 데이터셋 전 문항으로 점검 통과 (아래 `smoke_check.ipynb`) |
| 논문 목표 (Table 1, Phi3.5V-Med) | VQA-RAD 82.04 / SLAKE 85.20 (baseline overall) |

## 구성 (Expert-CFG 레포에서 이식, 출처는 `PORTED_FROM.md`)

| 파일 | 내용 |
|---|---|
| `01_Phi35V_baseline_VQARAD_SLAKE.ipynb` | 베이스라인 학습 · val 체크포인트 선택 · test 1회 평가 (공통 분할, `DATASET`으로 VQA-RAD / SLAKE) |
| `02_Expert_CFG_RAG_comparison.ipynb` | RAG · Expert-RAG · Expert-CFG 평가 (논문 식 4 · 7 · 9 · 10, union 검색, 5% 게이팅), 4-2절 진단 · 검색 방식 비교 |
| `smoke_check.ipynb` | transformers 4.44.2 점검 (어댑터 로드, 베이스라인 재현, 중립 CFG, KV 캐시). `FULL_CHECK`로 전 문항 |
| `reports/Phi35V_results.md` | 베이스라인 결과와 체크포인트 선택 |
| `reports/RAG_CFG_results.md` | RAG · CFG 전체 과정과 결과 (12절이 통일 정의 v2 결과) |
| `reports/weekly_report_2026-11-22.html` | 2026-11-22 주간 보고 (페이지 형식) |

공용 코드는 `../common/`(채점, 검색 파일 생성, 게이팅, 하이라이트), 분할은 `../data_common/vqarad_img/`, 불확실성 계산 명세는 `../docs/UNCERTAINTY_SPEC.md`다.

## 실행 순서 (Colab, GPU)

1. `01`로 어댑터를 학습하거나, 이미 올라간 어댑터를 `AIMS_shared/checkpoints/phi35/`에서 받는다(`../manifests/`로 해시 검증).
2. 검색 파일은 `../common/build_rag_testjson.py`로 만들거나 `AIMS_shared/rag_assets/`에서 받는다.
3. `02`를 데이터셋마다 **런타임을 재시작**하고 한 번씩 실행한다.
4. 환경을 바꿨다면 `smoke_check.ipynb`로 먼저 점검한다.

## 알려진 의존성과 한계 (이식 직후 상태)

- **노트북이 코드를 Expert-CFG 레포에서 clone한다.** `01`과 `02`의 첫 셀이 Expert-CFG의 `handoff/phi35v-vqarad-slake-rag-cfg` 브랜치를 clone해서 `vqa_scoring.py`, 저자 `src/highlighter_modules`, processor, 검색 파일 빌더를 쓴다. 이 레포는 비공개라 Colab에서 바로 clone할 수 없기 때문이다(토큰을 요구하지 않는다). **그 브랜치의 최신 커밋이 원격에 푸시되기 전에는 Colab이 옛 버전을 받는다**(예: 빌더의 `--retrieval` 옵션이 없다). 푸시 전에는 노트북이 Colab에서 정상 동작하지 않을 수 있다.
- **`02`는 transformers 5.x 호환 패치(12~16번 셀)를 포함한다.** 4.44.2에서 RAG · CFG **점수를 재현한 적은 없다**(스모크는 모델 · 훅 · 캐시 점검일 뿐이다). 4.44.2로 옮기려면 패치를 빼고 다시 실행해야 한다.
- 결과표의 점수는 5.x 패치 환경에서 나왔다.

## 다음 단계

1. 노트북이 Expert-CFG를 clone하지 않도록 코드를 공유 폴더(`AIMS_shared/code/`, 해시 기록)로 올려 쓰게 한다. 먼저 `common/`의 채점과 빌더, 저자 processor 받기부터.
2. 02를 4.44.2로 옮기고(5.x 패치 제거) 점수를 다시 낸다.
3. 02의 생성 · 훅 · 검색 로직을 모듈(`common/retrieval.py`, `records.py`, `phi35/` 모듈)로 나눈다. 이때 정답 주입 행의 closed 하이라이트 규칙을 `common/highlights.py`와 하나로 정한다.
