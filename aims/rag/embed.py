"""Embedding and FAISS scaffold module."""

from dataclasses import dataclass
from operator import index
from typing import List
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import os
import pickle

@dataclass
class Document:
    document_id: str
    text: str
    question: str
    answer: str
    source: str = "VQA-RAD"

class EmbedIndexer:
    """Placeholder for embedding + FAISS index logic."""
    """
    텍스트를 임베딩하고 FAISS 인덱스를 구축합니다.

    사용법:
        # 최초 1회 구축
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)  # load_vqarad() 결과
        indexer.save("data/faiss_index")

        # 이후 실험에서는 로드만
        indexer = EmbedIndexer.load("data/faiss_index")
        docs = indexer.search("Is there pleural effusion?", k=3)
    """

    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM = 384  # Dimension of the embedding model

    def __init__(self):
        self.model = SentenceTransformer(self.EMBED_MODEL_NAME)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.documents: List[Document] = []

    def build_from_hf(self, hf_dataset) -> None:
        """Build the FAISS index from a Hugging Face dataset."""
        self.documents = self.hf_to_document(hf_dataset)
        print(f"[EmbedIndexer] 코퍼스: {len(self.documents)}개 문서")

        texts = [doc.text for doc in self.documents]
        embeddings = self.encode(texts)
        self.index=self.build_faiss(embeddings)
        print(f"[EmbedIndexer] FAISS index built with {len(self.documents)} documents.")
        
        

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if self.model is None:
            print(f"Loading embedding model: {self.EMBED_MODEL_NAME}")
            self.model = SentenceTransformer(self.EMBED_MODEL_NAME)

        embedings= self.model.encode(
            texts, batch_size=64, show_progress_bar=len(texts)>100, normalize_embeddings=True
        )
        return embedings.astype(np.float32)
    

    def search(self, query: str, k: int = 5) -> List[Document]:
        # Placeholder for search logic
        assert self.index is not None, "FAISS index is not built. Call build() first."
        query_embedding = self.encode([query])
        distances, indices = self.index.search(query_embedding, k)

        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]

    def save(self, index_path: str) -> None: 
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_path, "faiss.index"))

        with open(os.path.join(index_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f) #인덱스와 문서 리스트를 함께 저장

        print(f"[EmbedIndexer] 저장 완료: {index_path}")

    @classmethod
    def load(cls, index_path: str) -> "EmbedIndexer":
        indexer = cls()
        indexer.index = faiss.read_index(os.path.join(index_path, "faiss.index"))

        with open(os.path.join(index_path, "documents.pkl"), "rb") as f:
            indexer.documents = pickle.load(f)

        print(f"[EmbedIndexer] 로드 완료: {index_path}")
        return indexer

    def hf_to_document(self, hf_dataset) -> List[Document]:
        """Convert Hugging Face dataset to a list of Document dataclasses."""
        documents = []
        seen = set()  # To avoid duplicates
        for i, sample in enumerate(hf_dataset):
            question = str(sample["question"]).strip()
            answer = str(sample["answer"]).strip()
            if not question or not answer:
                continue  # Skip if question or answer is empty
            key=(question.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)
            
            doc = Document(
                document_id=str(i),
                text=f"Q: {question} A: {answer}",
                question=question,
                answer=answer,
            )
            documents.append(doc)
        return documents

    def build_faiss(self, embeddings: np.ndarray):
        """Build the FAISS index from embeddings."""
        index = faiss.IndexFlatIP(self.EMBED_DIM)
        index.add(embeddings)
        return index
    
if __name__ == "__main__":
    from aims.data.dataset import load_vqarad

    # 1. 데이터 로드
    train_data, _ = load_vqarad(only_yes_no=True)
    print(f"train 샘플 수: {len(train_data)}")

    # 2. 인덱스 구축
    indexer = EmbedIndexer()
    indexer.build_from_hf(train_data)

    # 3. 검색 테스트
    query = "Is there pleural effusion?"
    results = indexer.search(query, k=3)
    print(f"\n[Query] {query}")
    for i, doc in enumerate(results):
        print(f"  {i+1}. Q: {doc.question}")
        print(f"     A: {doc.answer}")

    # 4. 저장 → 로드 → 재검색 동일한지 확인
    indexer.save("data/faiss_index")
    indexer2 = EmbedIndexer.load("data/faiss_index")
    results2 = indexer2.search(query, k=3)
    ids1 = [d.document_id for d in results]
    ids2 = [d.document_id for d in results2]
    print(f"\n저장/로드 후 결과 동일: {ids1 == ids2}")