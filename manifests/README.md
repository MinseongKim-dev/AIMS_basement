# 매니페스트

체크포인트와 큰 파일은 Hugging Face Hub(비공개)에 있고, 이 폴더의 JSON이 **무엇이 어디에 있는지**를 기록한다.
다운로드는 항상 `revision`(커밋 해시)으로 고정하며 "최신"을 받지 않는다. 받은 뒤 파일별 sha256을 검증한다.

`checkpoints.json` 항목 형식:

```json
{
  "phi35-vqarad-e10-s42": {
    "repo_id": "<조직>/<이름>",
    "repo_type": "model",
    "revision": "<커밋 해시>",
    "files": {"adapter_model.safetensors": "<sha256>", "adapter_config.json": "<sha256>"},
    "protocol": "compliant",
    "note": "train만 학습, val 선택 ep6"
  }
}
```

올리기와 받기: `python tools/hf_sync.py --help`. 토큰은 환경변수 `HF_TOKEN` 또는 `huggingface-cli login` 세션에서만 읽는다.
