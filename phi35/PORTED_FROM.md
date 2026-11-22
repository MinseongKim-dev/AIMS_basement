# 이식 기록 (Expert-CFG → AIMS_basement/phi35)

이 폴더의 Phi-3.5V 노트북과 보고서는 `MinseongKim-dev/Expert-CFG`에서 가져왔다. 이 레포가 앞으로 기준이다. 수정은 여기서 하고 Expert-CFG에는 되돌리지 않는다.

| 이 폴더의 파일 | 출처 | 커밋 | 이식 때 변경 |
|---|---|---|---|
| `01_Phi35V_baseline_VQARAD_SLAKE.ipynb` | `handoff/phi35v-vqarad-slake-rag-cfg` | `9514ad7` (로컬 커밋, **원격 푸시 전**) | 없음 (출력 없음) |
| `02_Expert_CFG_RAG_comparison.ipynb` | 같음 | `9514ad7` | 없음 (출력 없음, 5.x 호환 패치 포함) |
| `smoke_check.ipynb` | 이 레포에서 작성 (Expert-CFG의 `Phi35V_transformers4442_smoke_check.ipynb`와 같은 내용) | | |
| `reports/Phi35V_results.md` | `VQA_RAD_reproduction` | `1469883` | 없음 |
| `reports/RAG_CFG_results.md` | 같음 | `1469883` | 없음 |
| `reports/weekly_report_2026-11-22.html` | 2026-11-22 주간 보고(페이지로 게시한 것의 사본) | | |

이식한 시점의 해시는 `PORTED.sha256`에 있다. 이후 수정은 git 이력으로 추적한다.

## 가져오지 않은 것
| 안 가져옴 | 이유 / 대신 |
|---|---|
| `src/highlighter_modules/`, `Phi-3.5-vision-instruct/processing_phi3_v_cfg.py` 등 저자 코드 | 저자 레포에 라이선스가 없다. `tools/fetch_author_assets.py`로 고정 커밋에서 받는다. Expert-CFG의 사본에는 수정이 있을 수 있는데, 스모크 점검은 저자 원본 커밋으로 통과했다 |
| `vqa_scoring.py`, `tools/build_rag_testjson.py` | 이미 `../common/`에 있다 (`vqa_scoring.py`는 Expert-CFG 작업 브랜치와 동일) |
| `data_common/` | 이미 `../data_common/`에 있다 |
| MUMC · M3AE · LLaVA-Med 노트북과 보고서 | Phi 계열이 아니라서 Expert-CFG에 둔다 |
| 02 노트북의 실행 출력 | 실행 결과는 Drive의 `ragcfg_*` 폴더와 보고서에 있다 |
