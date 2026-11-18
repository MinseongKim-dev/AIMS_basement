# 매니페스트

체크포인트와 큰 파일은 공용 저장소에 두고, 이 폴더의 JSON이 **무엇이 어디에 있는지**를 기록한다.
공용 저장 위치(공유 Drive 폴더 또는 랩 서버 공용 폴더)는 **아직 정하지 않았다.** 어느 쪽이든 아래 규칙은 같다.

1. 릴리스 폴더는 `AIMS_release/v1/<이름>/` 에 올리고 **한 번 올리면 덮어쓰지 않는다.** 바뀌면 이름에 `-v2`를 붙여 새로 올린다.
2. 공용 릴리스 위치(`AIMS_release/`)만 팀원에게 공유한다. 작업용 개인 저장소 전체는 공유하지 않는다. (Drive면 폴더 공유, 서버면 그룹 읽기 권한과 올린 뒤 `chmod -R a-w`)
3. 올린 직후 `python tools/release_manifest.py make ...` 로 sha256을 기록하고 이 폴더의 JSON을 커밋한다.
4. 받는 쪽은 원본에서 작업하지 않고 **복사해서** 쓰고, 쓰기 전에 `python tools/release_manifest.py verify ...` 로 검증한다. (Drive면 공유 폴더를 자기 Drive에 같은 이름의 바로가기로 추가해야 Colab에서 마운트된다)
5. 토큰, 이메일, 개인 경로가 들어간 로그는 릴리스에 넣지 않는다. 매니페스트에는 개인 PC 경로를 적지 않는다(`--where`는 공용 위치 안의 상대 경로 라벨).

`checkpoints.json` 항목 형식:

```json
{
  "phi35-vqarad-e10-s42-v1": {
    "where": "AIMS_release/v1/phi35-vqarad-e10-s42-v1",
    "files": {"adapter_model.safetensors": "<sha256>", "adapter_config.json": "<sha256>", "run_config.json": "<sha256>"},
    "protocol": "compliant",
    "note": "train만 학습, val 선택 ep6",
    "recorded": "2026-11-18"
  }
}
```

기록과 검증: `python tools/release_manifest.py --help`.
