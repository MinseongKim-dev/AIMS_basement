# common — 모델과 무관한 코드

- `vqa_scoring.py` : 채점. 원본(`score_phi`, 공식), 교수님 지침의 통일 채점(`summarize_unified`: closed accuracy, open EM · F1), 모델 원래 채점(`score_native` MUMC, `score_llava` LLaVA-Med 평가 코드 이식), 참고용 `S1+S2`(규칙 완화), 부트스트랩 신뢰구간. `python common/vqa_scoring.py` 로 자체 검사.
- `build_rag_testjson.py` : RAG 검색 파일 생성(BiomedCLIP). KB 구성(`--kb`, 기본 `peir,roco,trainqa`), 검색 방식(`--retrieval cross|union`, 기본: train QA가 있으면 union), 출력 접미어(`--out_suffix`).
  - train QA는 항상 논문 · 팀원 방식(이미지별로 합침, `no` 제외, "질문? 답.")이다. 옛 방식(`pair`)은 제거했고 `--trainqa paper`는 기존 명령과의 호환용으로 받기만 한다.
  - `--pathvqa`: PathVQA train QA(HF `flaviagiammarino/path-vqa`, 고정 커밋)를 train QA에 더한다(논문 KB 구성). 같은 사진은 이미지 내용으로 묶는다. 출력 파일 이름에 `_pathvqa`가 붙어 기존 파일과 섞이지 않으며, 기본 동작은 그대로다.
  - 통일 방식(변형 A): 옵션 없이 `--out_suffix _v2`. 변형 B(이미지→텍스트 + train QA): `--retrieval cross --out_suffix _v2cross`. 변형 C(train QA 없음): `--kb peir,roco --out_suffix _v2`(자동으로 cross).
  - 입력은 `data_common/vqarad_img/{trainset,testset}.json`과 VQA-RAD 이미지 폴더다.
  - 알려진 한계: union에서는 train QA 항목이 항상 top-1이 되어 PEIR · ROCO가 쓰이지 않는다(VQA-RAD 100%, SLAKE 99.9%). VQA-RAD는 top-1의 90%가 같은 이미지의 train QA다.
  - 입력 파일(PEIR 캡션, 키워드)은 저자 배포본이며 이 레포에 없다.

- `highlights.py` : 하이라이트와 정답 주입 규칙. **주의:** `inject_highlights(answer, answer_type, fallback)`는 closed의 하이라이트로 `fallback`(질문과 겹치는 캡션 문장, 팀원 규칙)을 쓰는데, Expert-CFG 레포의 02 노트북은 정답 주입 행의 closed에 검색 파일의 n-gram 하이라이트를 쓴다. 두 구현이 다르고, 정답 주입 행의 closed 점수에 영향을 줄 수 있다(검증 안 함). 02의 결과를 이 모듈로 옮길 때 하나로 정한다.
- `gating.py` : 불확실성 집계(식 4), 상위 5% 선택(동점이면 번호가 작은 문항), AUC.

두 파일(`vqa_scoring.py`, `build_rag_testjson.py`)은 `Expert-CFG` 레포에서 가져왔고 거기서 계속 수정되고 있다. 이 레포로 옮긴 뒤에는 이 레포가 기준이다.
