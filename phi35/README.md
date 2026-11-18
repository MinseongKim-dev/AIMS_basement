# phi35 — Phi-3.5-vision-instruct 4.2B

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3.5-vision-instruct` |
| Med LoRA | `ecoxial2007/Phi-3.5V-Med` (이어서 학습: r64, lr 1e-4, batch 64, cosine, warmup 0.03) |
| processor | 저자 `Phi-3.5-vision-instruct/processing_phi3_v_cfg.py`, `processor_config.json` → `python tools/fetch_author_assets.py` |
| 환경 | `transformers==4.44.2` (검증 전), eager attention |
| 논문 목표 (Table 1, Phi3.5V-Med) | VQA-RAD 82.04 / SLAKE 85.20 (baseline overall) |

## 이식 대기 (아직 이 폴더에 없음)

현재 구현은 `Expert-CFG` 레포의 노트북에 있다.
- `01_Phi35V_baseline...` : 베이스라인 학습·평가 (공통 분할, val 선택)
- `Expert_CFG_RAG_comparison.ipynb` : RAG · Expert-CFG (논문 식 7·9·10, 식 4 엔트로피, union 검색)
  - 단, 지금 노트북은 transformers 5.x용 패치를 포함하고 있다. 4.44.2에서는 그 패치를 빼야 한다.

## 먼저 할 점검 (4.44.2)

1. 모델과 어댑터(VQA-RAD ckpt-257) 로드.
2. 기존 저장 예측과 비교: 일부 문항을 다시 추론해 일치율 확인.
3. 하이라이트가 있는 문항에서 중립 설정(두 가지 logits 동일)과 논문 설정(달라짐) 점검.
4. KV 캐시 유무 디코딩 일치.
5. 실행 환경(버전, GPU)을 결과와 함께 기록.
