# AIMS — Adaptive Inference with Medical Similarity

Medical VQA (Visual Question Answering) 실험 프레임워크.
BiomedCLIP 기반 분류기에 불확실성 추정 + RAG를 결합해 yes/no 질문에 답하는 파이프라인입니다.

---

## 프로젝트 구조

```
aims/
├── data/
│   └── dataset.py          # VQA-RAD 데이터셋 로드 (yes/no 필터링)
├── models/
│   ├── medcnn.py           # SimpleMedCNN (baseline CNN)
│   ├── ViT.py              # ViT frozen / full fine-tuning
│   ├── biomedclip.py       # BiomedCLIP VQA 모델 (메인 모델)
│   └── train_ViT.py        # 4개 모델 학습 루프 + 체크포인트 저장
├── rag/
│   ├── embed.py            # SentenceTransformer + FAISS 인덱스
│   ├── retriever.py        # Hybrid Retriever (dense + BM25 RRF)
│   └── pipeline.py         # AIMS 파이프라인 (4가지 라우팅 모드)
├── uncertainty/
│   └── metrics.py          # entropy / ECE / MC Dropout 계산
├── experiments/
│   ├── run_comparison.py   # 4-mode 파이프라인 비교 실험
│   ├── tune_tau.py         # tau_low / tau_high 그리드 서치
│   ├── run_mcdropout.py    # MC Dropout vs Entropy 비교
│   ├── run_ece.py          # 4개 모델 ECE 캘리브레이션 평가
│   └── visualize.py        # 결과 차트 생성 (results/*.png)
└── check.py                # 환경 설치 확인용
results/
├── comparison_results.json
├── mcdropout_results.json
├── ece_results.json
├── fig_comparison.png
├── fig_mcdropout.png
└── fig_ece.png
data/
├── checkpoints/            # 학습된 모델 가중치 (.pt)
└── faiss_index/            # FAISS 인덱스 + 문서 목록 (.pkl)
```

---

## 환경 설정

Python 3.12 권장.

```bash
python -m venv aims-env
# Windows
aims-env\Scripts\activate
# macOS / Linux
source aims-env/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## 실험 실행 순서

### 1단계 — 모델 학습 (체크포인트 생성)

```bash
python -m aims.models.train_ViT
```

- SimpleMedCNN / ViT-frozen / ViT-full / **BiomedCLIP** 순서로 학습
- 최고 val accuracy 기준 `data/checkpoints/*.pt` 자동 저장
- 이미 체크포인트가 있으면 건너뜀 (재실행 안전)

### 2단계 — 4-mode 파이프라인 비교

```bash
python -m aims.experiments.run_comparison
```

| 모드 | 설명 |
|------|------|
| `baseline` | 1차 추론만, RAG 없음 |
| `always_rag` | 항상 RAG → BiomedCLIP에 컨텍스트 입력 |
| `adaptive` | entropy < τ_low → direct, 그 외 → RAG |
| `expert` | entropy > τ_high → 전문가 이관 (B방식) |

결과: `results/comparison_results.json` + W&B 대시보드

### 3단계 — tau 임계값 튜닝 (선택)

```bash
python -m aims.experiments.tune_tau
```

validation set (train의 15%)으로 `tau_low`, `tau_high` 최적값 탐색.

### 4단계 — 추가 실험

```bash
# MC Dropout vs Entropy 불확실성 비교
python -m aims.experiments.run_mcdropout

# 4개 모델 ECE 캘리브레이션 평가
python -m aims.experiments.run_ece
```

### 5단계 — 시각화

```bash
python -m aims.experiments.visualize
```

`results/` 폴더에 PNG 3개 생성. 실험 결과 JSON이 없으면 더미 데이터로 레이아웃만 확인.

---

## 파이프라인 구조

```
이미지 + 질문
     │
 BiomedCLIP (1차 추론)
     │
 entropy 계산
     ├── entropy < τ_low  → direct 답변
     ├── τ_low ≤ entropy  → RAG 검색 → BiomedCLIP (2차 추론, 컨텍스트 포함)
     └── entropy > τ_high → expert 이관 (expert mode만)
```

RAG 2차 추론 프롬프트 형식:
```
Medical image. Context: Q: ... A: ... Q: ... A: ... Question: {question}
```

---

## 모델 설명

**BiomedCLIP** (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
- image_encoder (ViT-B/16) + text_encoder (PubMedBERT)
- image(512) + text(512) → concat(1024) → Linear(256) → 2-class logits
- `freeze_backbone=True`: classifier head만 학습 (권장)

---

## 주의사항

- `data/checkpoints/biomedclip.pt` 없이 `run_comparison.py`를 실행하면  
  linear head가 랜덤 초기화 상태라 **모든 모드가 ~53% (다수 클래스)** 로 동일하게 나옴.  
  → **반드시 `train_ViT.py` 먼저 실행할 것.**
- `data/` 폴더 (체크포인트, FAISS 인덱스)는 `.gitignore`로 추적 제외됨.  
  팀원은 각자 `train_ViT.py`를 실행해 로컬에 생성해야 함.
- W&B 연동: `wandb login` 후 실행하거나 `WANDB_MODE=offline` 환경변수 설정.

---

## 환경 설치 확인

```bash
python -m aims.check
```
