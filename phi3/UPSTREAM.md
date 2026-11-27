# 팀원 코드(phi3_urp)를 가져온 기록

| 항목 | 내용 |
|---|---|
| 출처 | `https://github.com/each-yi/phi3_urp` |
| 커밋 | `bedc923` ("SLAKE까지 실행 완료", 2026-11-23). 그 전은 `97fdb48`, 처음 가져온 것은 `cab1984` |
| 가져온 날 | 2026-11-22 (cab1984), 2026-11-23 (97fdb48로 동기화), 2026-11-28 (bedc923의 `Phi3V_baseline_A100.ipynb`로 동기화) |
| 수정 | **있음 (2026-11-28부터)**: `phi3cfg/evaluate.py` · `model.py` · `train.py`, `tests/test_logic.py`, `Phi3V_baseline_A100.ipynb`, `Phi3V_team_A100.ipynb`의 내장 코드. 아래 "수정 이력" 참고. `UPSTREAM.sha256`은 원본 해시 그대로라 이 파일들은 `sha256sum -c`에서 FAILED가 정상 |

## 가져온 것과 가져오지 않은 것

| 가져옴 | 위치 |
|---|---|
| 패키지 `phi3cfg/` (`generation.py` 논문 식 4 · 7~10, `retrieval.py` KB · union 검색 · 전문가 주석, `train.py`, `infer.py`, `evaluate.py`, 데이터 준비 모듈 등 11개) | `phi3/phi3cfg/` |
| CPU 테스트 (`test_logic.py`, `cpu_smoke.py`) | `phi3/tests/` |
| 팀 프로토콜 학습 · val 선택 · test 베이스라인 노트북 (출력 없음) | `phi3/Phi3V_teamproto_colab.ipynb` |
| **A100 전용 논문 재현 베이스라인 노트북** (출력 없음) | `phi3/Phi3V_baseline_A100.ipynb` |
| 원본 README | `phi3/UPSTREAM_README.md` |

| 가져오지 않음 | 이유 |
|---|---|
| `phi3cfg/__pycache__/*.pyc` | 원본이 추적 중인 캐시 파일 |
| `Phi3V_baseline_colab.ipynb`, `Phi3V_resume_colab.ipynb` | 옛 T4 기준 노트북이고 실행 출력이 들어 있음 |
| `2507.09209v1.pdf` | 논문 PDF(5.7MB). 저작물이라 복사하지 않음. arXiv에서 받는다 |
| `requirements.txt` | 복사하지 않고 `phi3/requirements.txt`에 **내용을 따라 새로 썼다**(팀원 파일 기준, `bitsandbytes==0.44.1` 한 줄만 주석 처리) |
| `run_phi3.sh` | 쓰지 않음 |

## 규칙

1. **이 복사본에서 수정하고, 팀원 저장소에는 커밋하지 않는다.**
2. 수정할 때마다 아래 "수정 이력"에 파일, 이유, 날짜를 적고 `UPSTREAM.sha256`은 **원본 해시로 남겨 둔다**(원본과 비교하는 기준이므로 갱신하지 않는다). 수정 여부는 `sha256sum -c UPSTREAM.sha256`로 확인한다.
3. 원본이 바뀌어도 자동으로 반영되지 않는다. 새 커밋을 가져오려면 파일을 비교해서 직접 합치고 이 문서의 커밋과 날짜를 갱신한다.
4. 두 노트북(`Phi3V_teamproto_colab.ipynb`, `Phi3V_baseline_A100.ipynb`)은 같은 코드를 `%%writefile`로 내장한다. `teamproto`의 내장본은 cab1984의 `phi3cfg/`와 동일함을 확인했고, **97fdb48에서 바뀐 `train.py`(아래 이력)는 반영되어 있지 않다**(기본값이 같아 동작은 같다). **코드를 수정하면 노트북 내장본과 어긋난다.** 수정한 코드로 학습하려면 노트북을 다시 만들거나 수정본을 Drive에서 복사해 쓴다.

## 라이선스와 동의

- 원본 저장소에 LICENSE 파일이 없어 재배포 권한이 명시되어 있지 않다. 이 레포는 **비공개 팀 저장소**이고 공개하지 않는다.
- 이 폴더의 기존 방침(`phi3/README.md`)은 "옮길 때 팀원의 동의를 받는다"였다. **복사에 대한 팀원의 명시적 동의는 확인하지 못했다.** 2026-11-23에 레포 소유자가 비공개 팀 레포에 올려도 된다고 판단해 푸시했다(팀원이 같은 코드를 A100 노트북으로 돌려 보라고 했던 맥락이다). 팀원이 반대하면 `phi3/phi3cfg/`와 팀원 노트북을 레포에서 빼고 이 폴더는 받는 방법만 남긴다. 동의를 받으면 날짜를 적는다: 동의 확인일 2026-11-28 (팀원 본인이 bedc923 노트북과 코드를 이 레포에 올리겠다고 요청)

## Colab에서 쓰는 법과 남은 문제

이 레포는 비공개라 Colab에서 바로 clone할 수 없다(토큰을 요구하지 않는다).
- **학습 · 베이스라인**: `Phi3V_teamproto_colab.ipynb`가 코드를 내장하므로 이 노트북을 그대로 열어 실행한다.
- **RAG · CFG 단계**: `run_teammate_stage.ipynb`는 지금 팀원 공개 저장소를 고정 커밋으로 clone한다(수정 없는 원본). **수정한 코드를 쓰려면** `phi3/phi3cfg/`를 Drive의 공유 폴더(`AIMS_shared/code/...`, 릴리스 규칙과 해시 기록 적용)에 올리고 노트북이 거기서 복사하도록 바꿔야 한다. 아직 하지 않았다.

## 수정 이력

| 날짜 | 파일 | 내용 | 이유 |
|---|---|---|---|
| 2026-11-28 | `phi3cfg/evaluate.py` | `--rule {compliant,relaxed}` · `--rule-note` 인자 추가. 논문 대비 차이(pp)를 출력하고 report에 `paper_diff`로 저장(test일 때만). infer가 쓴 `<예측>.meta.json`의 runtime(환경 버전, GPU, attention)과 rule, split을 report에 함께 저장. **`score()` · `summarize()` · AUC 계산은 바꾸지 않음** | 결과마다 환경 버전과 규칙 구분을 남기고(팀 `docs/RULES.md` 3절), 논문 대비 차이를 evaluate가 직접 출력하게 하려고 |
| 2026-11-28 | `phi3cfg/model.py` | `load()`의 meta에 `attn`(실제 `_attn_implementation`, 이 코드에서는 항상 eager) 추가. 로딩 동작은 같음 | sdpa로 학습한 체크포인트를 eager로 평가하므로 추론 attention을 결과에 기록 |
| 2026-11-28 | `phi3cfg/train.py` | `--train-splits`(기본 `train`) 인자와 `load_train()` 추가. `train,validation`이면 두 분할을 합쳐 학습. 기본값이면 동작이 이전과 같음 | 선택 옵션: train+val 3,064 학습(`relaxed` 행) |
| 2026-11-28 | `tests/test_logic.py` | `test_load_train`, `test_evaluate_report`(차이 · rule · runtime 저장, AUC) 추가 | 위 변경 확인 (CPU) |
| 2026-11-28 | `Phi3V_baseline_A100.ipynb` | VQA-RAD 데이터를 `download_data`(HF mirror, train 1,793)에서 `vqarad_common`(팀 공통 분할 train 2,749 / val 315 / test 451)으로 교체하고 `vqarad_common.py` 내장 셀 추가. VQA-RAD 결과 폴더를 `lora_vqa_rad_common` · `vqa_rad_common`으로 분리(이전 결과 재사용 방지). VQA-RAD 체크포인트 `--save-epochs 2,4,6,8,10` → val overall 최고, 동점이면 이른 epoch → test 한 번. 7-1절 기존 어댑터(`phi3-vqarad-e10-s42-v1`) 평가, 7-3절 `RUN_TRAINVAL` 옵션(train+val, `relaxed`), 결과마다 `result_*.json`(점수, 논문 차이, AUC, rule, 환경, sha256, 학습 하이퍼파라미터), 9절 하이퍼파라미터 차이 표. **SLAKE의 데이터 경로 · 결과 폴더 · 선택 규칙(최근 2개 중 val, 동점이면 뒤 epoch)은 그대로.** 내장 코드는 위 3개 파일과 같게 맞춤 | 58.32의 주원인이 학습 분할(1,793)이라 팀 공통 분할로 다시 재현하고, epoch를 test가 아닌 val로 고르기 위해 |
| 2026-11-28 | `Phi3V_team_A100.ipynb` | 내장 `evaluate.py` · `model.py` · `train.py`만 위 변경과 같게 맞춤. 노트북 로직은 그대로 | 코드와 내장본 일치 (`Phi3V_teamproto_colab.ipynb` 내장본은 여전히 cab1984 그대로라 맞추지 않음) |
| 2026-11-23 | `Phi3V_team_A100.ipynb` (**새 파일**, 팀원 `Phi3V_baseline_A100.ipynb` 파생) | VQA-RAD를 팀 공통 분할(`vqarad_common`)로, epoch를 VQA-RAD 10 / SLAKE 5로, 체크포인트를 `--save-epochs`(2·4·6·8·10 / 1~5)로, val 선택 동점을 이른 epoch로, 결과 위치를 `phi3_team_a100`으로 바꾸고 `run_teammate_stage.ipynb`가 읽는 `selected.json` · `test_ckpt<step>.jsonl`을 만드는 함수를 추가. 팀원 코드(`phi3cfg/`)는 수정하지 않음 | 팀원이 A100 노트북으로 실행하라고 했고, 우리 팀 Phi-3.5V와 같은 프로토콜로 비교하려면 데이터와 선택 규칙이 같아야 함 |

### 원본의 변경 (우리 수정 아님)

| 커밋 | 내용 |
|---|---|
| `97fdb48` → `bedc923` | `Phi3V_baseline_A100.ipynb`만 바뀜(나머지 파일 동일). **실제로 VQA-RAD · SLAKE 결과를 낸 버전.** `GRAD_CKPT = GPU_GB < 70`(80GB A100이면 끔), 3.13 설치 셀 변경, `PYTORCH_CUDA_ALLOC_CONF` 설정과 백그라운드 실행(`run_bg`/`status`) 제거. 원본에는 실행 출력이 들어 있고, 여기에는 출력을 지워서 올렸다(RULES 4). 결과: VQA-RAD test 34.66/73.90/58.32(open/closed/overall, epoch 10 = `checkpoint-280`, val 없음), SLAKE test 84.13/86.78/85.17(val best epoch 9 = `checkpoint-691`). 체크포인트는 팀원 Drive `phi3_a100/work/lora_{vqa_rad,slake}/` |
| `cab1984` → `97fdb48` | `phi3cfg/train.py`: `--warmup-ratio`(0.03), `--weight-decay`(0.0), `--max-grad-norm`(1.0) 인자 추가. 기본값은 이전 하드코딩과 같아 기본 동작은 같다. `Phi3V_baseline_A100.ipynb` 추가(공개 어댑터의 `training_args.bin`에서 값을 읽어 넘김). README 갱신. 논문 PDF 추가(복사 안 함) |
