# 결과 기록 형식

실행 하나당 `results/<날짜>_<모델>_<데이터셋>_<이름>.json` 파일 하나. 문항 단위 예측은 같은 이름의 `.jsonl`에 두고, 크면 Drive 릴리스 폴더에 둔다.

```json
{
  "model": "phi35",                       // phi3 | phi35
  "dataset": "vqarad",                    // vqarad | slake
  "row": "ExpertCFG-5-v2",                // 표의 행 이름
  "rule": "compliant",                    // compliant | relaxed  (docs/RULES.md)
  "rule_note": "정답 미사용, 하이라이트 = 자체 키워드 규칙",
  "scoring": "phi",                       // phi(공식) | unified | llava | native | s1s2(참고)
  "scores": {"open": 0.0, "closed": 0.0, "overall": 0.0, "n_open": 0, "n_closed": 0},
  "baseline_delta": {"overall": 0.0, "ci95": [0.0, 0.0]},
  "gate": {"entropy": "eq4", "top_percent": 5, "n_selected": 23},
  "cfg": {"gamma": 1.3, "alpha": 0.01, "beta": 3.0, "delta": 3.0986},
  "kb": {"name": "peir,roco,trainqa", "variant": "_v2", "retrieval": "union"},
  "seed": 42,
  "checkpoint": {"manifest": "phi35-vqarad-e10-s42-v1"},   // manifests/checkpoints.json 의 이름. sha256 은 매니페스트에 있다
  "env": {"transformers": "4.44.2", "peft": "", "torch": "", "gpu": ""},
  "note": ""
}
```

- 필수: `model`, `dataset`, `row`, `rule`, `scoring`, `scores`, `checkpoint`, `env`.
- `rule`이 `relaxed`면 `rule_note`에 무엇이 완화됐는지 적는다(예: "open 정답을 참조에 덧붙임").
- 점수는 0~100 퍼센트 단위.
