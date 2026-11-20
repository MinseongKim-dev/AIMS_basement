# 불확실성(엔트로피) 계산 명세 — Phi-3.5V

RAG · Expert-CFG의 5% 게이팅은 베이스라인 예측의 **엔트로피 상위 5% 문항**을 고른다. 이 선택이 팀원 값과 우리 값에서 달라지면 두 결과를 비교할 수 없으므로,
**팀원이 계산한 파일을 기준으로 삼고** 아래 설정과 형식을 맞춘다. (Phi-3.5V 먼저. 이 명세는 모델 로드 부분만 Phi-3에서 바뀐다.)

## 1. 입력 (모두 `AIMS_shared`)

| 무엇 | 위치 | 매니페스트 이름 |
|---|---|---|
| 어댑터 (VQA-RAD) | `checkpoints/phi35/phi35-vqarad-e10-s42-v1/` | 같은 이름 |
| 어댑터 (SLAKE) | `checkpoints/phi35/phi35-slake-e5-s42-v1/` | 같은 이름 |
| 베이스라인 test 예측 (VQA-RAD 451) | `baselines/phi35-vqarad-baseline-ckpt257-v1/test_predictions.json` | 같은 이름 |
| 베이스라인 test 예측 (SLAKE 1061) | `baselines/phi35-slake-baseline-ckpt380-v1/test_predictions.json` | 같은 이름 |

받은 폴더는 복사한 뒤 `python tools/release_manifest.py verify --name <이름> --path <복사한 폴더>` 로 해시를 검증한다.
예측 파일에는 **엔트로피 필드를 일부러 뺐다**(옛 저장값은 `<|end|>`를 평균에 포함한 다른 정의다).

## 2. 모델 설정 — 베이스라인 예측을 만들 때와 같아야 한다

`run_config.json`이 어댑터 폴더에 없어서 이 표가 유일한 기록이다.

| 항목 | 값 |
|---|---|
| 기반 모델 | `microsoft/Phi-3.5-vision-instruct` (HF). **Med LoRA를 따로 받을 필요 없음** — 어댑터 하나를 기반 모델에 바로 얹는다 |
| 어댑터 로드 | `PeftModel.from_pretrained(base, <어댑터 폴더>)` (peft 0.12.0에서 설정 키 수정 없이 로드됨, 스모크 점검 확인) |
| 정밀도 | bf16 (`torch_dtype=torch.bfloat16`; transformers 5.x의 `dtype=` 인자와 이름이 다르다) |
| attention | `_attn_implementation='eager'` |
| **이미지 크롭 수** | **`num_crops=16`** (`AutoProcessor.from_pretrained(..., trust_remote_code=True, num_crops=16)`). 기본값과 다를 수 있으니 반드시 지정한다. 다르면 같은 답에 대해 다른 이미지 토큰으로 엔트로피를 구하게 된다 |
| 프롬프트 | `tok.apply_chat_template([{'role':'user','content': "<|image_1|>\n{질문}"}], tokenize=False, add_generation_prompt=True)`. 시스템 문구와 캡션 없음 |
| 디코딩(예측을 만들 때) | greedy, `max_new_tokens=64` |
| processor | 저자 `processing_phi3_v_cfg.py`를 `tools/fetch_author_assets.py`로 받아 쓴 환경에서 예측을 재현했다(50/50). HF 기본 processor가 같은 이미지 토큰을 만드는지는 **확인하지 않았으니**, 같은 결과가 필요하면 저자 processor를 쓴다 |

환경: `requirements.txt` (transformers 4.44.2, peft 0.12.0, accelerate 0.34.2 …). Colab Python 3.12 / torch 2.11에서 검증했다.
**bitsandbytes는 설치하지 않는다.** 0.44.1은 이 환경의 triton에 없는 `triton.ops`를 import해서 LoRA를 만들 때 실패한다. 설치돼 있으면 `pip uninstall -y bitsandbytes` 후 런타임을 재시작한다.

## 3. 문항 짝짓기

- VQA-RAD: `data_common/vqarad_img/testset.json` (451문항)의 순서가 `idx`와 같다. 이미지는 `image_name`으로 VQA-RAD 이미지 폴더에서 찾는다(원본 JSON은 필요 없다).
- SLAKE: HF `BoKelvin/SLAKE`의 영어 문항 test 분할 1061문항. 순서는 예측 파일의 `idx`를 따른다.
- **모든 문항에서 `(질문, 정답)`이 예측 파일의 값과 일치하는지 assert한다.** 어긋나면 멈춘다(순서가 달라 엉뚱한 문항에 엔트로피가 붙는 것을 막는다).

## 4. 계산 (논문 식 4)

예측 파일의 답을 그대로 **teacher forcing**한다(다시 생성하지 않는다).

1. 프롬프트 토큰 `L`개 뒤에 답 토큰(`tok(예측, add_special_tokens=False)`)을 이어 붙여 forward 1번(`use_cache=False`).
2. 답 토큰을 예측하는 위치(`logits[L-1 : L-1+답길이]`)마다 **어휘 전체**의 엔트로피 `H = -Σ p·log p`를 구한다 (`log_softmax`를 float32로).
3. **특수 토큰(`tok.all_special_ids`)인 답 토큰은 평균에서 뺀다.** 답 뒤의 `<|end|>`를 예측하는 위치는 애초에 포함하지 않는다.
4. 남은 토큰 엔트로피의 평균이 그 문항의 엔트로피. 남은 토큰이 없으면 `null`.

참고용 구현은 이 레포의 02 노트북(Expert-CFG 레포)의 "게이팅 엔트로피" 셀과 팀원 `phi3cfg`의 식 4 구현이며, 두 정의는 같다. 문항당 forward 1번이라 약 2초/문항이다(VQA-RAD 약 15분, SLAKE 약 35분, L4 기준 이전 실행 추정).

## 5. 출력 형식

`AIMS_shared/uncertainty/phi35-<데이터셋>-eq4-v1/entropy.json` (한 번 올리면 덮어쓰지 않는다. 바뀌면 `-v2`).

```json
{
  "model": "phi35",
  "dataset": "vqarad",
  "definition": "eq4: answer-token vocabulary entropy, special tokens excluded, mean over answer tokens",
  "mode": "teacher_forcing_on_baseline_predictions",
  "adapter": "phi35-vqarad-e10-s42-v1",
  "predictions": "phi35-vqarad-baseline-ckpt257-v1",
  "n": 451,
  "env": {"python": "", "transformers": "", "peft": "", "torch": "", "gpu": "", "num_crops": 16},
  "entropy": {"0": 0.1234, "1": null, "...": 0.0}
}
```

키는 문항 번호(`idx`)의 문자열, 값은 엔트로피(없으면 `null`). 올린 뒤 `release_manifest.py make`로 해시를 기록한다.

## 6. 5% 선택 규칙 (두 쪽이 같은 함수를 쓴다)

`common/gating.py`의 `select_top_percent(entropies, 5)`: 개수 `max(1, round(N × 5 / 100))` (VQA-RAD 451 → 23, SLAKE 1061 → 53), 엔트로피가 높은 순, **동점이면 번호가 작은 문항 먼저**, `null`은 고르지 않는다.

## 7. 올리기 전 점검

1. 예측 재현: 50문항을 greedy로 생성해 저장된 예측과 같은지 본다(`phi35/smoke_check.ipynb`의 4번 셀 방식, 50/50이어야 한다).
2. 짝짓기 assert를 전 문항에서 통과한다.
3. 엔트로피 분포가 비정상이 아닌지 본다: `null` 개수, 음수 없음, 최댓값.
4. 우리 쪽 참고값과 비교한다. VQA-RAD에 대해 02 노트북이 계산한 식 4 값이 있고, **선택된 상위 23문항이 거의 겹쳐야 한다**(정확한 일치 여부는 토큰화 차이 때문에 장담하지 못한다). 겹침이 크게 낮으면 위 2번 표(특히 `num_crops`)부터 의심한다. 참고값이 필요하면 알려 달라.

## 8. 아직 맞추지 못한 것

- 논문 Table 6(closed 전용, fine-tuned)의 AUC 0.958을 우리 closed AUC(약 0.80)가 재현하지 못한다. 집계 정의를 바꿔도 같았고 원인은 모른다. 미검증 가설: closed 문항에 yes/no 두 토큰만의 이진 엔트로피를 쓴다.
- 논문 Fig. 3의 AUC(>0.8)는 우리 전체 문항 AUC(약 0.82)와 모순되지 않는다. 그림의 정확한 값과 문항 구성(전체/closed)은 원문 텍스트에 없다.
