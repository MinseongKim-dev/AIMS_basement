# common — 모델과 무관한 코드

- `vqa_scoring.py` : 채점. 원본(`score_phi`, 공식), 교수님 지침의 통일 채점(`summarize_unified`: closed accuracy, open EM · F1), 모델 원래 채점(`score_native` MUMC, `score_llava` LLaVA-Med 평가 코드 이식), 참고용 `S1+S2`(규칙 완화), 부트스트랩 신뢰구간. `python common/vqa_scoring.py` 로 자체 검사.
- `build_rag_testjson.py` : RAG 검색 파일 생성(BiomedCLIP). KB 구성(`--kb peir,roco,trainqa`), train QA 방식(`--trainqa pair|paper`), 검색 방식(`--retrieval cross|union`), 출력 접미어(`--out_suffix`).
  - 통일 방식(논문 · 팀원): `--trainqa paper --retrieval union --out_suffix _v2`.
  - 알려진 한계: union에서는 train QA 항목이 항상 top-1이 되어 PEIR · ROCO가 쓰이지 않는다(VQA-RAD 100%, SLAKE 99.9%). VQA-RAD는 top-1의 90%가 같은 이미지의 train QA다.
  - 입력 파일(PEIR 캡션, 키워드)은 저자 배포본이며 이 레포에 없다.

두 파일은 `Expert-CFG` 레포에서 가져왔고 거기서 계속 수정되고 있다. 이 레포로 옮긴 뒤에는 이 레포가 기준이다.
