# 매니페스트

체크포인트와 큰 파일은 공유 Google Drive 폴더(`AIMS_release/`)에 있고, 이 폴더의 JSON이 **무엇이 어디에 있는지**를 기록한다.
Drive에는 버전 고정이 없으므로 아래 규칙을 지킨다.

1. 릴리스 폴더는 `AIMS_release/v1/<이름>/` 에 올리고 **한 번 올리면 덮어쓰지 않는다.** 바뀌면 이름에 `-v2`를 붙여 새로 올린다.
2. 공유 폴더(`AIMS_release/`)만 팀원에게 **보기 권한**으로 공유한다. 작업용 Drive 전체는 공유하지 않는다.
3. 올린 직후 `python tools/drive_manifest.py make ...` 로 sha256을 기록하고 이 폴더의 JSON을 커밋한다.
4. 받는 쪽은 공유 폴더를 자기 Drive에 **바로가기로 추가**하고(Colab에서 마운트되려면 필요), 원본에서 작업하지 않고 **복사해서** 쓴다. 쓰기 전에 `python tools/drive_manifest.py verify ...` 로 검증한다.
5. 토큰, 이메일, 개인 경로가 들어간 로그는 릴리스에 넣지 않는다. 매니페스트에는 개인 PC 경로를 적지 않는다(`--drive-folder`는 공유 폴더 안의 상대 경로).

`checkpoints.json` 항목 형식:

```json
{
  "phi35-vqarad-e10-s42-v1": {
    "location": "google-drive",
    "drive_folder": "AIMS_release/v1/phi35-vqarad-e10-s42-v1",
    "files": {"adapter_model.safetensors": "<sha256>", "adapter_config.json": "<sha256>", "run_config.json": "<sha256>"},
    "protocol": "compliant",
    "note": "train만 학습, val 선택 ep6",
    "recorded": "2026-11-18"
  }
}
```

기록과 검증: `python tools/drive_manifest.py --help`.
