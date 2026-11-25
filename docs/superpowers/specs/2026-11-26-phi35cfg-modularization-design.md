# 설계: Phi-3.5V 노트북 모듈화 (`phi35cfg`)

작성 2026-11-26 · 상태: **승인됨 (2026-11-26)** · "(작성자 판단)" 표시는 작성자가 제안했고 검토에서 함께 승인된 항목이다.

## 1. 목적

`phi35/`의 노트북에 들어 있는 로직을 `phi3/phi3cfg/`의 구조를 따른 패키지로 나누고, 전부 `transformers==4.44.2`로 통일한다.
결과는 규칙 준수 행만 내고, 채점은 공식(원본 Phi) 채점을 쓴다.

## 2. 목표 · 하지 않는 것 · 실패 조건

**목표**
- `phi35/phi35cfg/` 패키지와 CLI. 노트북은 환경 준비와 CLI 호출만 한다.
- 학습, val 선택, 베이스라인, 5% 게이팅, RAG, Expert-CFG, 결과표를 패키지로 실행한다.
- 채점 · 게이팅 · 하이라이트 · 검색 파일 생성은 `common/` 한 곳에서만 구현한다.

**하지 않는 것**
- `phi3/` 수정, 두 모델의 공통 코어 만들기.
- 어댑터 재학습, 새 점수 목표.
- `phi3cfg`의 CFG 코드를 그대로 가져오기 (processor와 마스크 처리가 다르다).
- 기존 노트북 · 보고서 · `common/`의 함수 삭제.

**실패 조건**
- test 정답이 모델 입력, 게이팅, 설정 선택에 들어간다.
- 저장된 베이스라인 예측을 다시 채점한 값이 VQA-RAD 79.59 / SLAKE 85.63이 아니다.
- 5.x 패치나 transformers 버전 분기가 새 패키지에 남는다.

## 3. 제약

- 환경: 루트 `requirements.txt` (transformers 4.44.2, peft 0.12.0 …). Colab Python 3.12, bf16, 추론은 eager. bitsandbytes 없음.
- 교수님 규칙: 정답 주입 금지. 이에 따라 S1+S2 채점도 쓰지 않는다 (2026-11-26 결정).
- 저자 코드는 라이선스가 없어 커밋하지 않고 `tools/fetch_author_assets.py`로 받는다.
- 레포가 비공개라 Colab에서 clone할 수 없다.
- 규칙: `docs/RULES.md` (val로 선택, test 한 번), 결과 형식 `results/SCHEMA.md`.

## 4. 검토한 접근법

| 안 | 내용 | 장단점 |
|---|---|---|
| **A (선택)** | `phi35cfg/` 패키지를 `phi3cfg`와 같은 구성으로 만들고 `common/` 사용 | Phi-3와 대응, 팀원 코드 불변. 비슷한 코드가 두 곳에 남는다 |
| B | 두 모델 공통 코어 + 모델별 어댑터 | 중복 최소. 팀원 코드를 크게 고치고 다시 검증해야 한다 |
| C | 셀을 `.py`로 옮기기만 함 | 빠르다. CLI와 구조를 얻지 못하고 전역변수 의존이 남는다 |

2026-11-26에 A를 선택했다.

## 5. 패키지 구성

| `phi35/phi35cfg/` | 하는 일 | 원천 |
|---|---|---|
| `data.py` | VQA-RAD(`data_common/vqarad_img/`)와 SLAKE(HF 영어 분할) 읽기, 문항 수 검사, 검색 파일 · 예측 · 엔트로피의 문항 짝짓기 | `01` 3절, `02` 1절 |
| `model.py` | 기반 모델 + 어댑터 + 저자 processor 로드, 학습용 sdpa 선택 옵션, 환경 기록 | `01` 4절, `smoke_check` 2~3 |
| `generation.py` | 프롬프트, greedy 생성(KV 캐시), 식 4 엔트로피, CFG 훅(식 7~10), `CFGConfig` | `smoke_check` 5, `phi3cfg/generation.py`의 인터페이스 |
| `retrieval.py` | 검색 파일 읽기, 규칙 준수 하이라이트. 파일 생성은 `common/build_rag_testjson.py` | `02` 실험 셀 |
| `train.py` | downstream 학습 CLI, 지정 epoch 저장, 이어하기 | `01` 5절, `phi3cfg/train.py` |
| `infer.py` | `--mode baseline\|rag\|expert_cfg`, `entropy` 계산, 5% 선택, 문항 단위 JSONL 이어쓰기 | `02` 2~3절, `phi3cfg/infer.py` |
| `evaluate.py` | val 체크포인트 선택, 베이스라인과 병합, 채점, 논문 비교, 부트스트랩 CI, AUC, `results/` 형식 저장 | `01` 6절, `02` 4절, `phi3cfg/evaluate.py` |

함께 두는 것: `phi35/tests/`(CPU 로직 테스트, GPU 점검 스크립트), `phi35/Phi35V_run.ipynb`(얇은 실행 노트북).

의존 방향은 `phi35cfg → common`뿐이다. `phi3cfg`는 import하지 않는다.

## 6. 동작 정의

### 6.1 베이스라인
- 기준은 릴리스된 예측 `phi35-vqarad-baseline-ckpt257-v1`, `phi35-slake-baseline-ckpt380-v1`이다 (`manifests/checkpoints.json`). 다시 생성한 값으로 바꾸지 않는다.
- 4.44.2에서 새로 생성한 베이스라인은 **확인용 행**으로 따로 적는다. 알려진 재현율은 444/451, 1054/1061이다. (작성자 판단)
- 채점은 `common/vqa_scoring.score_phi`만 쓴다.

### 6.2 실험 행
| 행 | 대상 | 입력 | γ |
|---|---|---|---|
| 베이스라인 | 전 문항 | 질문 | 없음 |
| w/ RAG | 전 문항 | top-1 캡션 + 질문 | 없음 |
| w/ 5% RAG | 엔트로피 상위 5% | 같음 | 없음 |
| w/ 5% Expert-CFG | 상위 5% | 캡션 + 하이라이트 | 1.5 |

- 선택되지 않은 문항은 베이스라인 예측을 그대로 쓴다.
- γ는 논문 본문값 1.5로 고정한다. test로 고른 값이 되지 않도록 1.3 행은 내지 않는다. (작성자 판단)
- α=0.01, β=3, δ=log β + 2.
- 하이라이트: closed는 `common/highlights.closed_sentence_highlights`, open은 검색 파일의 `highlights`. 하이라이트 토큰이 없으면 RAG와 같은 입력으로 생성하고 `cfg_applied=false`로 기록한다.
- 하이라이트 토큰 위치는 저자 processor(`qs_highlighted_parts`)가 만든 마스크를 쓴다.
- 정답 주입 · oracle 행, 이전 자체 CFG 정의, S1+S2 출력은 옮기지 않는다.
- 생성 함수는 정답을 인자로 받지 않는다. 테스트로 확인한다. (작성자 판단)

### 6.3 생성 · 훅
- 모든 모드가 같은 greedy 루프를 쓴다. 중립 설정(α=1, β=1, δ=0, γ=1)이면 `model.generate`와 같은 문자열이어야 한다.
- attention 훅은 decoder self-attention의 4D additive mask에 bias를 더하고, 위치 · 키워드 인자를 모두 처리한다. 임베딩 훅은 prefill에만 적용한다.

### 6.4 게이팅
- `common/gating.select_top_percent(…, 5)`: VQA-RAD 23문항, SLAKE 53문항.
- 엔트로피 정의는 식 4(특수 토큰 제외), 저장된 베이스라인 예측의 teacher forcing (`docs/UNCERTAINTY_SPEC.md`).
- `--entropy <entropy.json>`으로 팀 기준 파일을 받는다. 파일이 없으면 `infer.py`로 같은 형식을 만들 수 있다. 팀원 파일이 나오면 그것이 기준이다.

### 6.5 학습
- `01`의 보고 설정만 남긴다: Med LoRA 이어서 학습, LoRA r64, lr 1e-4, 배치 64(1×64), cosine, warmup 0.03, weight decay 0.0, vision 고정, seed 42, VQA-RAD 10 epoch(2·4·6·8·10 저장), SLAKE 5 epoch(매 epoch 저장).
- val overall(원본 채점)로 선택하고 동점이면 이른 체크포인트. test는 선택한 것으로 한 번.
- `public` 분할, `merge_then_new`, train+val 학습은 옮기지 않는다.
- attention 기본은 eager, `--attn sdpa` 선택. sdpa 플래그 패치는 모델 파일 수정이므로 남긴다.
- `torch.load(weights_only=False)` 우회는 이어하기에 필요하다(torch ≥ 2.6).

### 6.6 파일 형식 (작성자 판단)
- 새 실행의 예측은 문항 단위 JSONL(`idx`, `question`, `answer`, `answer_type`, `prediction`, `entropy`, `mode`, `cfg_applied`)과 설정을 담은 `.meta.json`. 다른 설정의 파일에 이어쓰면 멈춘다.
- 릴리스된 기존 JSON 예측도 읽는다.
- 짝짓기는 `idx`를 쓰고 모든 문항에서 `(질문, 정답)` 일치를 assert한다.

### 6.7 Colab 전달 (작성자 판단)
- `common/`, `phi35/phi35cfg/`, 필요한 도구와 데이터 분할을 압축해 `AIMS_shared/code/phi35-<버전>/`에 올리고 sha256을 매니페스트에 기록한다. 노트북은 복사 후 검증해서 쓴다.
- `/content`, `drive.mount` 같은 Colab 경로는 노트북에만 두고 패키지는 인자로 받는다. 랩 서버에서도 CLI가 그대로 돈다.

### 6.8 함께 고칠 것
- `tools/fetch_author_assets.py`: processor가 import하는 `src/highlighter_modules/utils.py`를 같은 고정 커밋에서 받도록 추가.
- `docs/RULES.md`: 정답 주입 결과를 내지 않는다는 교수님 지침 추가.
- 루트 `README.md`, `phi35/README.md`: "정답 주입이 기본 비교 행" 문구와 실행 방법 갱신.
- 기존 `01` · `02` 노트북은 과거 결과의 기록으로 남기고 README에 그렇게 표시한다. (작성자 판단)
- 옮기지 않는 것: 4-2절 검색 진단 · 변형 비교, AUC 정의 비교, 손실 곡선 · 불확실성 그림, 업로드 셀. (작성자 판단)

## 7. 완료 기준

**CPU (로컬)**
1. 릴리스 예측 재채점이 79.59 / 85.63 (파일이 있어야 하므로 Drive에서 받아 확인).
2. 게이팅 개수 23 / 53, 동점 규칙.
3. 프롬프트와 하이라이트 규칙의 출력이 `02`의 규칙 준수 구현과 같다.
4. `tests/test_common.py`와 새 테스트 통과.

**GPU (Colab, 4.44.2)** — 사용자가 실행한다
5. 중립 설정 = `model.generate`, 논문 설정에서 logits가 달라짐, KV 캐시 유무 일치.
6. 베이스라인 재현 불일치율이 약 2% 이내.
7. 두 데이터셋에서 4개 행 결과표 생성.
8. 학습은 몇 스텝 실행과 이어하기만 확인한다.

RAG · CFG 점수는 4.44.2에서 처음 내는 값이므로 맞출 목표가 없다. 5.x 환경의 이전 값(5% Expert-CFG VQA-RAD +0.44, SLAKE +0.77)과 다를 수 있다.

## 8. 남은 위험

- VQA-RAD는 검색 top-1의 약 90%가 같은 이미지의 train QA다. 정답 주입은 아니지만 교수님 확인을 권한다. 답에 따라 검색 파일만 바뀌고 코드 구조는 그대로다.
- 릴리스 어댑터에 `run_config.json`이 없어 학습 설정의 근거는 `01` 노트북뿐이다.

## 9. 만드는 사람에게 맡기는 것

함수 이름과 모듈 내부 구조, 테스트 구성, 구현 순서.
