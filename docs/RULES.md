# 규칙

## 1. 프로토콜 구분 (교수님 규칙)

| 구분 | 조건 |
|---|---|
| `compliant` (규칙 준수) | train으로만 학습, val로 모델·epoch 선택(val은 이미지 단위로 분리, 시드 고정), test는 선택한 체크포인트로 **한 번만** 평가. 정답 후보·라벨 목록은 train 정답만 |
| `relaxed` (규칙 완화) | train+val 학습, 후보 목록에 test 정답 포함, test로 epoch/설정 선택, 정답을 입력(참조 텍스트·하이라이트)에 넣는 것, test 오답에서 도출한 채점 규칙 등 |

- 규칙 완화 결과는 **항상 별도 행**으로 쓰고 규칙 준수 결과처럼 보고하지 않는다.
- test로 설정(프롬프트, 규칙, epoch)을 고르는 탐색은 `relaxed`다.
- test 문항 자체를 학습에 쓰지 않는다.

## 2. 채점

- **공식: 원본(Phi 방식)** — closed는 정답 단어가 예측 토큰에 있으면 정답, open은 정답 토큰 recall. 논문이 recall 기준이라고 밝힌 것과 같은 계열이다.
- 교수님 지침(통일 평가): closed는 같은 정규화 + accuracy, open은 normalized exact match와 token-level F1. recall은 별도로 적되 accuracy와 직접 비교하지 않는다 → `vqa_scoring.summarize_unified`.
- `S1+S2`(open의 표기 차이·일반어 생략 허용)는 Phi의 VQA-RAD test 오답에서 도출한 규칙이라 **참고용(`relaxed`)** 이다. 공식 결과로 쓰지 않는다.
- 모델별 원래 채점(MUMC 자체 EM, LLaVA-Med 평가 코드)은 **논문 재현 판정용**으로 따로 쓸 수 있다. 이때도 어떤 채점인지 표에 적는다.

## 3. 결과 기록

모든 결과 행에 아래를 적는다 (`results/SCHEMA.md`): 모델, 데이터셋, 규칙 구분, 채점 방식, 시드, 체크포인트(매니페스트 이름과 해시), 환경 버전(transformers, peft, torch, GPU).

## 4. 작업 규칙

- `main`은 PR로만 병합한다. 공통 코드(`common/`, CFG 훅)를 바꾸는 PR은 아래를 통과해야 한다.
  - 채점 단위 검사(`python common/vqa_scoring.py`).
  - 중립 설정(α=1, β=1, δ=0)이면 정상 가지와 무조건부 가지의 logits가 같고, 논문 설정이면 하이라이트가 있는 문항에서 달라지는 점검.
- 노트북은 출력을 비우고 커밋한다(출력에 경로, 토큰, 이메일이 남을 수 있다).
- 커밋 이메일은 GitHub noreply 주소를 쓴다.
- 비밀값은 환경변수(`HF_TOKEN`) 또는 Colab Secrets로만 쓴다.
