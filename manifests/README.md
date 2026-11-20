# 매니페스트

체크포인트와 큰 파일은 공유 Drive 폴더 **`AIMS_shared`** 에 두고, 이 폴더의 JSON이 **무엇이 어디에 있는지**를 기록한다.
랩 서버로 옮기게 되면 위치만 바뀌고 아래 규칙은 같다.

1. 릴리스 폴더는 `AIMS_shared/checkpoints/<phi3|phi35>/<이름>-v1/` 에 올리고 **한 번 올리면 덮어쓰지 않는다.** 바뀌면 이름 끝을 `-v2`로 바꿔 새로 올린다.
2. 공유는 `AIMS_shared` 폴더만 한다. 작업용 개인 Drive 전체는 공유하지 않는다. (서버로 옮기면 그룹 읽기 권한과 올린 뒤 `chmod -R a-w`)
3. 올린 직후 `python tools/release_manifest.py make ...` 로 sha256을 기록하고 이 폴더의 JSON을 커밋한다.
4. 받는 쪽은 `AIMS_shared`를 **자기 Drive 최상위에 같은 이름(`AIMS_shared`)의 바로가기로 추가**해야 Colab에서 `/content/drive/MyDrive/AIMS_shared`로 마운트된다(경로가 사람마다 같아야 코드를 고치지 않는다). 원본에서 작업하지 않고 **복사해서** 쓰고, 쓰기 전에 `python tools/release_manifest.py verify ...` 로 검증한다.
5. 토큰, 이메일, 개인 경로가 들어간 로그는 릴리스에 넣지 않는다. 매니페스트에는 개인 PC 경로를 적지 않는다(`--where`는 공용 위치 안의 상대 경로 라벨).

## `AIMS_shared` 폴더 구조

```
AIMS_shared/
  checkpoints/phi35/<이름>-v1/   어댑터 (adapter_model.safetensors, adapter_config.json, 있으면 run_config.json. optimizer.pt 등 학습 상태는 제외)
  checkpoints/phi3/<이름>-v1/
  baselines/<이름>-v1/           베이스라인 test 예측 (test_predictions.json). 불확실성 필드는 뺀다
  uncertainty/<이름>-v1/         불확실성(엔트로피) 파일 entropy.json. 형식과 계산은 docs/UNCERTAINTY_SPEC.md
  rag_assets/<이름>-v1/          검색 파일(rag_*_v2.json), 텍스트 특징(.npy)
  results/<날짜>_<이름>/         문항 단위 예측(.jsonl), 요약(.json)
```
올리지 않는 것: 기반 모델 가중치(HF), 학습 상태 파일, 공개 데이터셋 이미지(VQA-RAD, SLAKE), 저자 배포 파일(PEIR 캡션 · 키워드, processor — `tools/fetch_author_assets.py`와 README의 링크로 받는다), 토큰 · 개인 정보가 든 로그.
현재 올라간 릴리스: `phi35-vqarad-e10-s42-v1`, `phi35-slake-e5-s42-v1`(어댑터), `phi35-vqarad-baseline-ckpt257-v1`, `phi35-slake-baseline-ckpt380-v1`(예측). 어댑터에는 `run_config.json`이 없다(학습 설정 기록 없음) — 추론 설정은 `docs/UNCERTAINTY_SPEC.md`에 있다.
팀원은 자기 Phi-3V 어댑터를 `checkpoints/phi3/`에 올린다. 두 사람이 모두 편집자 권한으로 공유하고, 덮어쓰기는 규칙과 `verify`로 막는다.

`checkpoints.json` 항목 형식:

```json
{
  "phi35-vqarad-e10-s42-v1": {
    "where": "AIMS_shared/checkpoints/phi35/phi35-vqarad-e10-s42-v1",
    "files": {"adapter_model.safetensors": "<sha256>", "adapter_config.json": "<sha256>", "run_config.json": "<sha256>"},
    "protocol": "compliant",
    "note": "train만 학습, val 선택 ep6",
    "recorded": "2026-11-18"
  }
}
```

기록과 검증: `python tools/release_manifest.py --help`.
