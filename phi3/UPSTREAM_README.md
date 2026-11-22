# Phi3V-Med 전용 Expert-CFG 재현 (arXiv:2507.09209)

**Phi-3-Vision(`microsoft/Phi-3-vision-128k-instruct`) + Phi-3V-Med LoRA(`ecoxial2007/Phi-3V-Med`)만** 사용한다.
Phi-3.5는 코드에서 아예 지원하지 않는다. 목표는 논문 Table 1의 Phi3V-Med 5개 행이다.

| Table 1 행 | 이 코드의 run 이름 | 내용 |
|---|---|---|
| Phi3V-Med 4.2B | `baseline` | downstream fine-tuning 후 그대로 추론, Eq.4 entropy 기록 |
| w/ RAG | `rag` | 모든 질문에 top-1 검색 caption 추가 |
| w/ 5% RAG | `rag5` | entropy 상위 5%만 RAG, 나머지는 baseline 답 유지 |
| w/ 5% Expert-RAG | `expert_rag5` | 상위 5%에 전문가 reference를 prompt에 추가 (guidance 없음) |
| w/ 5% Expert-CFG | `expert_cfg5` | 상위 5%에 reference + highlight + CFG (Eq.7–10) |

## 실행 (Colab, GPU)

**Baseline, Colab A100 (VQA-RAD, SLAKE, PathVQA)**: [Phi3V_baseline_A100.ipynb](Phi3V_baseline_A100.ipynb). bf16, 80GB에서는 gradient checkpointing 끔, 결과는 `MyDrive/phi3_a100`에 저장. 논문에 없는 warmup·weight decay·grad clip은 공개 어댑터의 `training_args.bin`에서 읽어 온다 (pickle 코드를 실행하지 않는 방식).

**Baseline만 (VQA-RAD, SLAKE)**: [Phi3V_baseline_colab.ipynb](Phi3V_baseline_colab.ipynb)를 Colab에 업로드하고 GPU 런타임에서 위부터 실행.
노트북 하나에 필요한 코드가 모두 들어 있어 폴더 업로드가 필요 없다.

**팀 Phi-3.5V와 같은 프로토콜**: [Phi3V_teamproto_colab.ipynb](Phi3V_teamproto_colab.ipynb). 팀 공통 VQA-RAD 분할(train 2,749 / val 315 / test 451, `phi3cfg.vqarad_common`), SLAKE 공식 val, VQA-RAD 10 / SLAKE 5 epoch, 지정 epoch 체크포인트 보존(`train.py --save-epochs`), val overall 최고 체크포인트로 test 1회. 결과는 `MyDrive/phi3_teamproto`에 따로 저장된다. 팀 코드와 남은 차이는 노트북 첫 셀에 정리했다. 아래 셸 스크립트는 RAG/Expert-CFG까지 포함한 전체 파이프라인이다.

```bash
cd /content/drive/MyDrive/phi3_expert_cfg     # 이 폴더를 Drive에 업로드한 경로
pip install -r requirements.txt
bash run_phi3.sh vqa_rad          # slake, pathvqa 도 동일
```

- 스모크 테스트: `LIMIT=5 SKIP_TRAIN=1 bash run_phi3.sh vqa_rad`
- `SKIP_TRAIN=1`: downstream 학습 없이 공개 Phi-3V-Med LoRA로 바로 추론 (Table 3 zero-shot 조건에 가까움)
- 메모리: T4는 기본 `PREC=fp16`, 부족하면 `PREC=4bit`. A100/L4는 `PREC=bf16`.
- 학습이 T4에서 OOM이면 `python -m phi3cfg.train ... --crops 4`
- 모든 추론 출력은 JSONL에 한 줄씩 append되므로 세션이 끊겨도 같은 명령으로 이어서 실행된다.

단계별 명령은 [run_phi3.sh](run_phi3.sh)에 있다:
`download_data` → `train` → `infer --mode baseline` → `retrieval build/search` → `infer`(rag/expert) → `evaluate`.

## 구현 대응표

| 논문 | 위치 |
|---|---|
| Eq.4 길이 정규화 predictive entropy (greedy) | [generation.py](phi3cfg/generation.py) `generate` |
| Eq.7 unconditional branch에서 highlight 임베딩 × α(0.01) | `_Hooks._embed` |
| Eq.9/10 normal branch attention score + log β(β=3) | `_Hooks._attn` (batch 0) |
| unconditional branch attention −δ·m, δ=log β+2 | `_Hooks._attn` (batch 1) |
| Eq.8 γ log P(c) − (γ−1) log P(c̄), γ=1.5 | `generate` |
| BiomedCLIP 검색, Union feature (Table 2) | [retrieval.py](phi3cfg/retrieval.py) |
| KB = 세 데이터셋 train QA를 이미지별 병합, "no" 답 제외 (App. B) | `retrieval build` |
| 전문가 주석 시뮬레이션 (Sec 5.2) | `retrieval.annotate` |
| downstream FT: LoRA r64, lr 1e-4, batch 64, 10 epoch, cosine (Sec 4.3) | [train.py](phi3cfg/train.py) |
| Open recall / Closed accuracy / Overall (Sec 4.3) | [evaluate.py](phi3cfg/evaluate.py) |

## 저자 공개 demo와 다른 점 (의도적)

저자 `demo.py`는 `where(highlight, 0.01, 3.0)` 같은 실수값 attention mask를 HF 기본 mask 변환에 그대로 넘긴다.
HF는 이 값을 `1 - mask`로 뒤집어 0이 아니면 모두 −inf 처리하므로, unconditional branch의 prompt 전체가
가려지고 Eq.7/9는 적용되지 않는다. 여기서는 decoder 각 self-attention의 additive mask에 bias를
직접 더하고 임베딩을 직접 스케일링해 **논문 수식대로** 구현했다 (eager attention 필수).

## 논문과 다를 수 있는 부분

- **데이터 split**: HF mirror(flaviagiammarino/vqa-rad, BoKelvin/SLAKE, flaviagiammarino/path-vqa)를 사용한다. 논문 split과 동일한지는 검증되지 않았다.
- **VQA-RAD open/closed 라벨**: HF mirror에는 `answer_type`이 없어 download 직후에는 "답이 yes/no면 closed"다 (test 251/200). `python -m phi3cfg.vqarad_labels --dataset-dir DATA/vqa_rad`로 원본 VQA-RAD 라벨(test closed 272 / open 179, "left or right?" 같은 선택형도 closed)로 바꾼다. 라벨 출처는 팀 repo `MinseongKim-dev/Expert-CFG`의 원본 3,515문항 JSON(commit 고정, sha256 확인)이고, (질문, 답) 정규화 문자열로 매칭한다 (test 451/451, train 1,793/1,793 일치). SLAKE는 원본 라벨이 이미 들어 있다.
- **closed 채점**: 팀 Phi-3.5V 베이스라인과 같은 규칙. 소문자로 바꾼 정답 전체가 예측의 토큰 중 하나이면 1점 (예측 "yes, no"는 yes/no 모두 정답, "right side" 같은 여러 단어 closed 정답은 항상 0점 — VQA-RAD test에서 1문항). open은 정답 토큰 recall.
- **KB caption**: 논문은 GPT-4로 QA를 caption으로 변환(Fig.7)했다. 기본값은 `"질문? 답."` 템플릿이다. `--llm gpt-4o`(OPENAI_API_KEY 필요, 데이터 텍스트가 OpenAI로 전송됨)로 논문 방식을 쓸 수 있다.
- **ROCO/PEIR**: 자동 다운로드하지 않는다. `{"image": 절대경로, "caption": ...}` JSONL을 만들어 `retrieval build --extra`로 추가한다.
- **closed 질문 highlight**: 논문은 GPT-4 키워드 추출(Fig.8/9)을 쓴다. 기본값은 질문과 내용어가 가장 많이 겹치는 KB 문장이다 (`--llm` 사용 시 Fig.8/9 프롬프트). 겹치는 문장이 없으면 해당 질문은 guidance 없이 expert-RAG로 처리되고 `cfg_applied=false`로 기록된다.
- **γ**: 본문은 1.5, 부록 Table 7의 최고값은 1.3이다. `--gamma`로 변경한다. δ 정의는 본문 표현이 모호해 `log β + 2`로 해석했다.
- **Prompt 형식**: 저자 demo와 같은 `"{reference} <|image_1|>\n{question}"`.
- T4는 bf16이 안 되므로 fp16을 쓴다. 논문은 3090 + bf16.

## 테스트

```bash
python tests/test_logic.py                       # 채점, top-% 선택, 전문가 주석 로직
python tests/cpu_smoke.py phi3v_small            # 실제 Phi-3-V 코드 + 작은 랜덤 모델 (가중치 불필요)
```
`cpu_smoke.py`는 하이라이트 토큰 정렬, KV-cache와 비캐시 디코딩 일치, 중립 설정(γ=1, α=1, β=1, δ=0)이
일반 디코딩과 같은지, 실제 CFG가 출력을 바꾸는지를 검사한다. 파일 상단에 준비 방법이 있다.
