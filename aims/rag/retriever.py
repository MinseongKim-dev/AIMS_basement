"""Hybrid Retriever: FAISS dense + BM25 sparse.

두 검색 결과를 Reciprocal Rank Fusion(RRF)으로 합칩니다.
RRF는 두 점수의 스케일이 달라도 rank 기반으로 합산하기 때문에
별도 정규화 없이 dense + sparse를 결합할 수 있습니다.

"""

from typing import List, Tuple

from aims.rag.embed import Document, EmbedIndexer


class BM25Retriever:
    """
    BM25 기반 sparse retriever.

    dense(의미 검색)와 달리 키워드 매칭 방식입니다.
    "pleural effusion"이라는 단어가 정확히 있으면 BM25가 잘 찾고,
    "fluid in the lungs" 같은 표현은 dense가 잘 찾습니다.
    둘을 합치는 이유가 여기 있습니다.
    """

    def __init__(self, documents: List[Document]):
        from rank_bm25 import BM25Okapi

        self.documents = documents
        # 공백 기준 토크나이징 (단순하지만 영어 의료 텍스트에 충분)
        tokenized = [doc.text.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Returns:
            List of (Document, bm25_score), score 높은 순
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_k_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        return [(self.documents[i], float(scores[i])) for i in top_k_idx]


class HybridRetriever:
    """
    FAISS dense + BM25 sparse를 RRF로 결합한 Hybrid Retriever.

    RRF score = Σ 1 / (rrf_k + rank_i)
        - rank_i: 각 retriever에서의 순위 (0-based)
        - rrf_k: 60 (논문 권장값, 상위 랭크의 영향력을 완화)

    Args:
        indexer: 구축된 EmbedIndexer
        rrf_k:   RRF 파라미터 (기본값 60)

    Usage:
        retriever = HybridRetriever(indexer)
        docs = retriever.retrieve("Is there pleural effusion?", k=3)
    """

    def __init__(self, indexer: EmbedIndexer, rrf_k: int = 60):
        self.indexer = indexer
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(indexer.documents)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        hybrid 검색 후 상위 k개 Document 반환.

        내부 동작:
            1. FAISS로 dense 후보 2k개
            2. BM25로 sparse 후보 2k개
            3. RRF score로 재순위화 → 상위 k개
        """
        results, _ = self._hybrid_search(query, k)
        seen_questions=set()
        unique_results=[]
        for doc in results:
            if doc.question not in seen_questions:
                unique_results.append(doc)
                seen_questions.add(doc.question)
            if len(unique_results) >= k:
                break
        return unique_results

    def retrieve_with_scores(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """RRF 점수도 함께 반환 (디버깅/분석용)."""
        return zip(*self._hybrid_search(query, k))

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _hybrid_search(
        self, query: str, k: int
    ) -> Tuple[List[Document], List[float]]:
        """RRF 계산 공통 로직."""
        candidate_k = k * 2

        # 1. Dense
        dense_docs = self.indexer.search(query, k=candidate_k)
        dense_ranks = {doc.document_id: rank for rank, doc in enumerate(dense_docs)}

        # 2. Sparse
        sparse_results = self.bm25.retrieve(query, k=candidate_k)
        sparse_ranks = {
            doc.document_id: rank
            for rank, (doc, _) in enumerate(sparse_results)
        }

        # 3. RRF
        all_ids = set(dense_ranks) | set(sparse_ranks)
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        # 4. 상위 k개
        id_to_doc = {doc.document_id: doc for doc in self.indexer.documents}
        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        docs = [id_to_doc[i] for i in top_ids if i in id_to_doc]
        scores = [rrf_scores[i] for i in top_ids if i in id_to_doc]
        return docs, scores


if __name__ == "__main__":
    from aims.data.dataset import load_vqarad
    from aims.rag.embed import EmbedIndexer

    # 1. 인덱스 로드 (embed.py 먼저 실행해서 저장해둬야 함)
    try:
        indexer = EmbedIndexer.load("data/faiss_index")
    except Exception:
        print("인덱스 없음 → 새로 구축")
        train_data, _ = load_vqarad(only_yes_no=True)
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)

    # 2. HybridRetriever 생성
    retriever = HybridRetriever(indexer)

    # 3. 검색 테스트
    queries = [
        "Is there pleural effusion?",
        "Is the heart size normal?",
        "Are there any fractures?",
    ]

    for query in queries:
        print(f"\n[Query] {query}")
        docs = retriever.retrieve(query, k=3)
        for i, doc in enumerate(docs):
            print(f"  {i+1}. Q: {doc.question}")
            print(f"     A: {doc.answer}")

    # 4. Dense vs Sparse vs Hybrid 비교
    query = "Is there pleural effusion?"
    print(f"\n{'='*50}")
    print(f"[Dense only]")
    for doc in indexer.search(query, k=3):
        print(f"  Q: {doc.question} | A: {doc.answer}")

    print(f"\n[BM25 only]")
    bm25 = BM25Retriever(indexer.documents)
    for doc, score in bm25.retrieve(query, k=3):
        print(f"  Q: {doc.question} | A: {doc.answer} | score: {score:.3f}")

    print(f"\n[Hybrid (RRF)]")
    for doc in retriever.retrieve(query, k=3):
        print(f"  Q: {doc.question} | A: {doc.answer}")
