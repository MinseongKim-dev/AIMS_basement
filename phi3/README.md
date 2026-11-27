# phi3 — Phi-3-vision-128k-instruct 4.2B

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3-vision-128k-instruct` |
| Med LoRA | `ecoxial2007/Phi-3V-Med` |
| processor | 저자 `Phi-3-vision-128k-instruct/processing_phi3_v_cfg.py`, `preprocessor_config.json` → `python tools/fetch_author_assets.py` |
| 환경 | `requirements.txt` = **팀원 저장소의 requirements.txt를 따름** (`transformers==4.44.2`, eager attention). 다른 점은 `bitsandbytes==0.44.1`을 주석 처리한 것 하나 (최신 Colab에서 peft 로드를 깨뜨림). Python 3.13은 `Phi3V_team_A100.ipynb`의 환경 셀이 처리 |
| 논문 목표 (Table 1, Phi3V-Med) | VQA-RAD 79.38 / SLAKE 83.98 (baseline overall) |

## 팀원 코드 (가져온 복사본)

팀원 구현(`each-yi/phi3_urp@cab1984`)을 이 폴더로 **복사해 왔다.** 출처, 가져온 파일과 제외한 파일, 수정 규칙, 라이선스 · 동의 상태는 `UPSTREAM.md`에 있다. 수정은 이 복사본에서만 하고 팀원 저장소에는 커밋하지 않는다.

```
phi3cfg/                    generation.py(논문 식 4 · 7~10) · retrieval.py(KB, union 검색, 전문가 주석) · train.py · infer.py · evaluate.py · 데이터 준비
tests/                      test_logic.py(순수 로직) · cpu_smoke.py(중립 설정, 가이던스 효과, KV 캐시 일치)
Phi3V_teamproto_colab.ipynb 팀 프로토콜: 학습 · val 선택 · test 베이스라인 (코드를 내장). 결과는 본인 Drive 의 phi3_teamproto 에 저장 (run_teammate_stage 가 읽는다)
Phi3V_team_A100.ipynb       **우리가 만든 파생본**: 위 A100 노트북의 학습 코드를 쓰되 데이터 · epoch · 선택 규칙을 팀 프로토콜(Phi-3.5V와 같음)로 바꿈. 결과는 phi3_team_a100 (run_teammate_stage 가 읽는다)
Phi3V_baseline_A100.ipynb   (팀원 원본을 수정, UPSTREAM.md 이력) A100(bf16)으로 논문 Table 1 의 Phi3V-Med 베이스라인(VQA-RAD 79.38 / SLAKE 83.98) 재현. VQA-RAD는 팀 공통 분할 + val 체크포인트 선택, 기존 어댑터 평가(7-1), 선택적 train+val relaxed 행(7-3). 결과는 phi3_a100
run_teammate_stage.ipynb    위 결과를 읽어 RAG · Expert-RAG · Expert-CFG 단계를 실행하고 진단
UPSTREAM.md                 출처 · 수정 규칙 · 수정 이력   (UPSTREAM.sha256 = 원본 해시)
```

`phi35`와 같은 점검(중립 설정이면 두 가지 logits 동일, 논문 설정이면 달라짐, KV 캐시 유무 디코딩 일치)을 이 복사본에서도 한다. `tests/cpu_smoke.py`가 CPU에서 그 일부를 확인한다. 로컬 확인: `cd phi3 && python tests/test_logic.py` (3개 통과, 2026-11-22).
