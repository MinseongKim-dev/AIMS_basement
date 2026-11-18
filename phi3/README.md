# phi3 — Phi-3-vision-128k-instruct 4.2B

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3-vision-128k-instruct` |
| Med LoRA | `ecoxial2007/Phi-3V-Med` |
| processor | 저자 `Phi-3-vision-128k-instruct/processing_phi3_v_cfg.py`, `preprocessor_config.json` → `python tools/fetch_author_assets.py` |
| 환경 | `transformers==4.44.2` (팀원 구현이 이미 사용), eager attention |
| 논문 목표 (Table 1, Phi3V-Med) | VQA-RAD 79.38 / SLAKE 83.98 (baseline overall) |

## 이식 대기 (아직 이 폴더에 없음)

현재 구현은 팀원 레포 `each-yi/phi3_urp`(패키지 `phi3cfg`)에 있다: `generation.py`(논문 식 4·7~10), `retrieval.py`(KB, union 검색, 전문가 주석 시뮬레이션), `train.py`, `infer.py`, `evaluate.py`.
이 폴더로 옮길 때는 팀원의 동의를 받고, 코드를 옮기기 전에 아래 점검을 같은 방식으로 `phi35`와 맞춘다.

- 중립 설정(α=1, β=1, δ=0)이면 두 가지 logits 동일, 논문 설정이면 달라짐 (`tests/cpu_smoke.py`에 있음).
- KV 캐시 유무 디코딩 일치.
