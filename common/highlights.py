"""하이라이트와 전문가 주석 규칙 (순수 텍스트 로직, GPU 불필요).

규준: 논문 §5.2 와 팀원 phi3_urp(retrieval.py)의 규칙.
  - open 문항: 정답을 전문가 주석으로 쓴다(규칙 완화). 검색 캡션에 정답이 없으면 캡션 뒤에 덧붙이고 정답을 하이라이트.
  - closed 문항: 질문과 내용어가 가장 많이 겹치는 검색 캡션 문장(최대 2개)을 하이라이트. 겹치는 문장이 없으면 하이라이트 없음
    (→ 가이던스 없이 RAG 와 같은 입력으로 생성).
"""
import re

STOP = set('''a an the is are was were be been of in on at to for from by with and or not no yes this that these those there here it its their
what which where when how why who whom does do did can could would should shown show shows seen see image images picture photo scan visible present
any some into than as about type kind part side'''.split())


def words(text):
    return re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', str(text).lower())


def content_words(text):
    return {w for w in words(text) if w not in STOP and len(w) > 1}


def split_sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', str(text)) if s.strip()]


def closed_sentence_highlights(question, caption, max_n=2):
    """질문과 내용어가 가장 많이 겹치는 문장(들). 같은 최대 겹침이 여럿이면 앞에서부터 max_n 개."""
    q = content_words(question)
    scored = [(len(q & content_words(s)), s) for s in split_sentences(caption)]
    best = max((n for n, _ in scored), default=0)
    return [s.strip() for n, s in scored if n == best and n > 0][:max_n]


def inject_caption(caption, answer, answer_type):
    """전문가 주석 주입 (규칙 완화): open 은 정답이 캡션에 없으면 뒤에 덧붙인다. closed 는 그대로."""
    if str(answer_type).strip().upper() != 'OPEN':
        return caption
    gt = str(answer).strip()
    return caption if gt.lower() in str(caption).lower() else f'{caption} {gt}.'


def inject_highlights(answer, answer_type, fallback=()):
    """주입 행의 하이라이트: open 은 정답 전체, closed 는 fallback(= closed 문장 하이라이트)."""
    return [str(answer).strip()] if str(answer_type).strip().upper() == 'OPEN' else list(fallback)
