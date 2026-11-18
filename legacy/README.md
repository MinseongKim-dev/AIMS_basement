# legacy

`aims_v1/` 은 이전 AIMS 프로젝트(VQA-RAD yes/no 분류, BiomedCLIP/ViT, FAISS 하이브리드 RAG, MC Dropout/ECE)의 코드다.
참고용이며 현재 레포의 코드와 연결되어 있지 않다. 이전 레포의 `wandb/` 기록, `__pycache__`, FAISS 인덱스와 `.pkl`, 결과물은 개인 정보와 불필요한 이진 파일이 들어 있어서 가져오지 않았다.

재사용 후보: `aims_v1/aims/uncertainty/metrics.py`의 ECE 계산. 논문 Table 6의 ECE 정의(엔트로피를 신뢰도로 바꾸는 방법)를 확인한 뒤에만 쓴다.
