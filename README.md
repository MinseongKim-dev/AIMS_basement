# AIMS_basement — Phi-3 / Phi-3.5V 기반 RAG · Expert-CFG 공통 레포

논문 *Uncertainty-Driven Expert Control* (arXiv:2507.09209)의 Phi3V-Med / Phi3.5V-Med 재현을 팀이 함께 쓰는 레포다.
체크포인트와 큰 파일은 여기에 올리지 않고 **공유 Drive 폴더 `AIMS_shared`** 에 두며, 이 레포에는 위치와 sha256(`manifests/`)만 둔다.

## 구조

```
common/            모델과 무관한 코드: 채점(vqa_scoring.py), RAG 검색 파일 생성(build_rag_testjson.py)
phi3/              Phi-3-vision-128k 전용 코드·설정·requirements (비어 있음: 이식 대기)
phi35/             Phi-3.5-vision: 패키지 phi35cfg/ (학습 · 추론 · 평가 CLI), 실행 노트북, 테스트, 이전 노트북 · 보고서
data_common/       팀 공통 VQA-RAD 분할 vqarad_img/ (train 2,749 / val 315 / test 451, 이미지 단위 val, 시드 42). 이 JSON과 VQA-RAD 이미지 폴더만 있으면 되고 원본 VQA-RAD JSON은 필요 없다
manifests/         체크포인트·큰 파일의 위치 라벨과 sha256
results/           실행 결과 요약 (형식: results/SCHEMA.md)
tools/             저자 파일 받기, 릴리스 매니페스트 기록·검증
docs/RULES.md      규칙(프로토콜 구분, 공식 채점, 작업 규칙)
docs/UNCERTAINTY_SPEC.md  불확실성(엔트로피) 계산 명세와 출력 형식 (Phi-3.5V 먼저, 팀원 담당)
```

Phi-3와 Phi-3.5V는 **폴더를 나눈다.** 모델 내부 구조(모듈 이름, 마스크 처리)에 의존하는 CFG 훅이 모델마다 다르기 때문이다.
모델과 무관한 로직(채점, 검색, 엔트로피 집계)만 `common/`에 둔다.

## 환경

- `transformers==4.44.2` 로 고정한다 (팀 결정). 나머지 패키지는 Phi-3.5V 스모크 점검을 마친 뒤 고정한다 → 각 폴더의 `requirements.txt`.
- Phi-3.5V는 4.44.2에서 **전 문항 점검을 통과**했다 (VQA-RAD ckpt-257 451문항, SLAKE ckpt-380 1061문항, Colab Python 3.12 / torch 2.11 / eager). KV 캐시 유무 디코딩 20/20 일치(두 데이터셋), 중립 CFG = 일반 생성 5/5, 논문 설정 CFG는 하이라이트 위치에서 logits이 바뀐다. 저장된 베이스라인 예측 재현은 VQA-RAD 444/451(98.4%), SLAKE 1054/1061(99.3%)로 **bit 단위 일치는 아니다**: 불일치 7문항씩이고 점수 영향은 VQA-RAD −0.52%p, SLAKE +0.22%p다 (방향이 일정하지 않아 수치 흔들림으로 보이며, 원인은 확정하지 못했다. 가설: 저장된 예측은 `ATTN_IMPL='sdpa'`(불가하면 eager)와 torch 2.3 계열, 점검은 eager와 torch 2.11). **RAG · CFG 점수를 4.44.2에서 재현한 것은 아니다.** 결과표에 환경 버전을 같이 적는다.
- attention은 eager. CFG가 4D attention mask에 bias를 더하기 때문이며 flash-attention은 bias를 무시한다.

## 시작하기

```bash
pip install -r requirements.txt
python tools/fetch_author_assets.py        # 저자 레포의 processor 파일을 고정 커밋에서 받는다 (저자 레포에 라이선스가 없어 복사해 올리지 않음)
python tools/release_manifest.py verify --name <manifests/checkpoints.json 의 이름> --path <복사해 온 폴더>   # 받은 체크포인트 검증
```

## 꼭 지킬 것

1. 결과 한 줄마다 **규칙 구분(`compliant` / `relaxed`)과 채점 방식**을 적는다. → `docs/RULES.md`
2. 공식 채점은 **원본(Phi 방식)** 이다. 다른 채점은 참고 열로만 쓴다.
3. 비밀값(토큰, 키), 개인 이메일, 개인 PC 경로를 올리지 않는다. 커밋 이메일은 GitHub noreply를 쓴다.
4. 체크포인트(`*.safetensors`, `*.pt`), 특징 파일(`*.npy`), 이미지, 원본 데이터셋은 올리지 않는다.
5. `main`에는 직접 푸시하지 않고 PR로 병합한다.

## 현재 상태

- Phi-3.5V: `phi35/phi35cfg/` 패키지(4.44.2). CPU 테스트는 `phi35/tests/`. **GPU 점검과 점수 재실행은 아직이다.**
- Phi-3V: 팀원 코드 복사본 `phi3/phi3cfg/`. 두 패키지는 `common/`만 공유하며 한 모듈로 합치지 않는다.

## 결과 현황 (Phi-3.5V, 원본 채점 = 공식)

**교수님 지침(2026-11-26): 정답 주입 결과를 내지 않는다. S1+S2 채점도 쓰지 않는다.** 아래는 규칙 준수 행이다.

| | VQA-RAD | SLAKE |
|---|---|---|
| 베이스라인 | 79.59 (논문 82.04) | 85.63 (논문 85.20) |
| 5% Expert-CFG (정답 미사용) | +0.44 (논문 +2.66) | +0.77 [−0.20, +1.74] (논문 +2.55) |

위 RAG · CFG 수치는 **transformers 5.x 호환 패치 환경에서 낸 이전 기록**이고 4.44.2에서는 아직 내지 않았다(`phi35/phi35cfg`로 다시 낸다). SLAKE 실행의 게이팅 정의(식 4 / 저장값)는 기록이 없다. 과거의 정답 주입 행은 `phi35/reports/RAG_CFG_results.md`에 기록으로만 남긴다. 논문 · 저자 코드와의 대조는 `phi35/reports/RAG_CFG_paper_audit.md`.
불확실성(엔트로피) 계산은 팀원 담당이며 `docs/UNCERTAINTY_SPEC.md`를 따른다.

## 아직 정하지 않은 것

| 항목 | 상태 |
|---|---|
| 공용 저장 위치 | **공유 Drive 폴더 `AIMS_shared`로 정함.** 랩 서버로 옮기게 되면 위치 라벨만 바뀐다 (`manifests/` 규칙은 같다) |
| 실행 환경 | Colab 또는 랩 GPU 서버 (미정). 서버로 가면 Colab 의존 코드(`/content/...`, `drive.mount`, `!pip`)를 걷어내고 설정 파일로 실행하는 진입점이 필요하다 |
| 4.44.2에서 Phi-3.5V 동작 | 이전 스모크 점검은 통과. 새 패키지의 GPU 점검(`phi35/tests/gpu_check.py`)과 RAG · CFG 점수는 미실행 |
| Hugging Face | 쓰지 않기로 함 (기반 모델과 공개 데이터셋 다운로드에만 사용, 로그인 불필요) |
