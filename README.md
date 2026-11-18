# AIMS_basement — Phi-3 / Phi-3.5V 기반 RAG · Expert-CFG 공통 레포

논문 *Uncertainty-Driven Expert Control* (arXiv:2507.09209)의 Phi3V-Med / Phi3.5V-Med 재현을 팀이 함께 쓰는 레포다.
체크포인트와 큰 파일은 여기에 올리지 않고 **Hugging Face Hub(비공개)** 에 두며, 이 레포에는 위치와 해시(`manifests/`)만 둔다.

## 구조

```
common/            모델과 무관한 코드: 채점(vqa_scoring.py), RAG 검색 파일 생성(build_rag_testjson.py)
phi3/              Phi-3-vision-128k 전용 코드·설정·requirements (비어 있음: 이식 대기)
phi35/             Phi-3.5-vision 전용 코드·설정·requirements (비어 있음: 이식 대기)
data_common/       팀 공통 VQA-RAD 분할 (train 2,749 / val 315 / test 451, 이미지 단위 val, 시드 42)
manifests/         체크포인트·큰 파일의 HF 위치, 커밋 해시, sha256
results/           실행 결과 요약 (형식: results/SCHEMA.md)
tools/             저자 파일 받기, HF 업로드·다운로드
docs/RULES.md      규칙(프로토콜 구분, 공식 채점, 작업 규칙)
legacy/aims_v1/    이전 AIMS 프로젝트 코드 (참고용. ECE, MC Dropout 코드가 있음)
```

Phi-3와 Phi-3.5V는 **폴더를 나눈다.** 모델 내부 구조(모듈 이름, 마스크 처리)에 의존하는 CFG 훅이 모델마다 다르기 때문이다.
모델과 무관한 로직(채점, 검색, 엔트로피 집계)만 `common/`에 둔다.

## 환경

- `transformers==4.44.2` 로 고정한다 (팀 결정). 나머지 패키지는 Phi-3.5V 스모크 점검을 마친 뒤 고정한다 → 각 폴더의 `requirements.txt`.
- 4.44.2에서 Phi-3.5V가 동작하는지, 이전 환경(5.x)의 결과와 얼마나 같은지는 **아직 검증하지 않았다.** 검증 전까지 결과표에 환경 버전을 같이 적는다.
- attention은 eager. CFG가 4D attention mask에 bias를 더하기 때문이며 flash-attention은 bias를 무시한다.

## 시작하기

```bash
pip install -r requirements-common.txt
python tools/fetch_author_assets.py        # 저자 레포의 processor 파일을 고정 커밋에서 받는다 (저자 레포에 라이선스가 없어 복사해 올리지 않음)
python tools/hf_sync.py download --entry <manifests/checkpoints.json 의 이름>     # 체크포인트 받기
```

## 꼭 지킬 것

1. 결과 한 줄마다 **규칙 구분(`compliant` / `relaxed`)과 채점 방식**을 적는다. → `docs/RULES.md`
2. 공식 채점은 **원본(Phi 방식)** 이다. 다른 채점은 참고 열로만 쓴다.
3. 비밀값(토큰, 키), 개인 이메일, 개인 PC 경로를 올리지 않는다. 커밋 이메일은 GitHub noreply를 쓴다.
4. 체크포인트(`*.safetensors`, `*.pt`), 특징 파일(`*.npy`), 이미지, 원본 데이터셋은 올리지 않는다.
5. `main`에는 직접 푸시하지 않고 PR로 병합한다.

## 현재 상태

- 이 레포는 골격이다. Phi-3.5V 쪽 구현은 `Expert-CFG` 레포의 노트북(`Expert_CFG_RAG_comparison.ipynb`)에 있고, Phi-3V 쪽은 팀원 레포(`each-yi/phi3_urp`)에 있다. 두 구현은 논문 식 7~10 정의로 통일했고 아직 한 모듈로 합치지 않았다.
- 합치기 전에 할 일: (1) Phi-3.5V를 4.44.2에서 스모크 점검, (2) 점검에서 쓰는 중립 설정 테스트(두 가지 logits 동일)와 KV 캐시 유무 디코딩 일치 테스트를 양쪽에 같은 방식으로 추가.
