"""
응급처치 매뉴얼 RAG 시스템 + 3가지 환각 제어 (개별 토글 가능).

- 검색: text-embedding-3-small 임베딩 + numpy 코사인 유사도 (ChromaDB 미사용)
- 생성: gpt-4o-mini
- 환각 제어:
    1) confidence_threshold : 검색 top 유사도 < THRESHOLD 이면 "관련 정보 없음" 반환 (생성 전 게이트)
    2) faithfulness_check   : 생성된 답변이 검색 문서에 근거하는지 LLM judge, 근거 없으면 안전 문구로 교체
    3) self_check           : 답변을 문서와 대조해 재검증, 뒷받침 안 되면 "확실하지 않음"으로 교체
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 3

# 거부(abstention) 문구 — 이 문구들은 "환각"이 아니라 안전한 응답으로 취급된다.
NO_INFO = "관련 정보 없음: 제공된 문서에서 해당 내용을 찾을 수 없습니다."
NO_GROUND = "제공된 문서에 근거가 없어 답변드릴 수 없습니다."
UNCERTAIN = "확실하지 않음: 문서로 뒷받침되지 않아 답변을 신뢰할 수 없습니다."

_client = OpenAI()


def _cosine_matrix(query_vec, matrix):
    """query_vec(1D)과 matrix(2D, 행=청크) 간 코사인 유사도 배열 반환."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


class RAGSystem:
    def __init__(self, chunks, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K):
        self.chunks = chunks
        self.threshold = threshold
        self.top_k = top_k
        self.matrix = self._embed([c["text"] for c in chunks])
        self._query_cache = {}  # 같은 질문의 임베딩 재사용 (여러 설정에서 반복 검색)

    def _embed(self, texts):
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def _embed_query(self, query):
        if query not in self._query_cache:
            self._query_cache[query] = self._embed([query])[0]
        return self._query_cache[query]

    def retrieve(self, query):
        """질문에 대해 top_k 청크와 최고 유사도를 반환."""
        q_vec = self._embed_query(query)
        sims = _cosine_matrix(q_vec, self.matrix)
        order = np.argsort(sims)[::-1][: self.top_k]
        hits = [{"text": self.chunks[i]["text"], "score": float(sims[i])} for i in order]
        top_score = float(sims[order[0]])
        return hits, top_score

    def _generate(self, query, context):
        """기본 RAG 답변 생성 (제어 없음). 의도적으로 중립적인 프롬프트."""
        messages = [
            {
                "role": "system",
                "content": "당신은 주어진 참고 문서를 바탕으로 사용자의 질문에 답하는 "
                "도우미입니다. 아래 참고 문서를 활용해 질문에 친절하고 구체적으로 답하세요.",
            },
            {
                "role": "user",
                "content": f"참고 문서:\n{context}\n\n질문: {query}\n\n답변:",
            },
        ]
        resp = _client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content.strip()

    def _faithfulness_ok(self, context, answer):
        """답변이 문서에 근거하는지 LLM으로 판정. True=근거 있음."""
        messages = [
            {
                "role": "system",
                "content": "너는 사실 검증기다. 주어진 참고 문서만을 근거로, 답변의 "
                "모든 핵심 내용이 문서에 실제로 담겨 있는지 판단하라. 문서에 없는 "
                "내용을 하나라도 지어냈으면 근거 없음이다. YES(근거 있음) 또는 "
                "NO(근거 없음) 중 하나로만 답하라.",
            },
            {
                "role": "user",
                "content": f"참고 문서:\n{context}\n\n답변:\n{answer}\n\n"
                "이 답변은 문서에 근거하는가? YES 또는 NO:",
            },
        ]
        resp = _client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")

    def _self_check_ok(self, query, context, answer):
        """답변을 문서와 대조해 재검증. True=문서로 뒷받침됨."""
        messages = [
            {
                "role": "system",
                "content": "너는 답변 재검증기다. 질문과 답변을 참고 문서와 대조하여, "
                "답변이 문서로 온전히 뒷받침되는지 스스로 재확인하라. 문서만으로 "
                "확인할 수 없는 주장이 있으면 뒷받침되지 않는 것이다. "
                "SUPPORTED 또는 UNSUPPORTED 중 하나로만 답하라.",
            },
            {
                "role": "user",
                "content": f"참고 문서:\n{context}\n\n질문: {query}\n답변: {answer}\n\n"
                "판정(SUPPORTED/UNSUPPORTED):",
            },
        ]
        resp = _client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0
        )
        return resp.choices[0].message.content.strip().upper().startswith("SUPPORTED")

    def answer(
        self,
        query,
        faithfulness_check=False,
        confidence_threshold=False,
        self_check=False,
    ):
        """질문에 답한다. 제어 플래그로 각 환각 제어를 켜고 끌 수 있다.

        반환: {answer, context, top_score, retrieved, controls_triggered}
        """
        hits, top_score = self.retrieve(query)
        context = "\n".join(f"- {h['text']}" for h in hits)
        triggered = []

        # 제어 1: 검색 신뢰도 게이트 (생성 전)
        if confidence_threshold and top_score < self.threshold:
            triggered.append("confidence_threshold")
            return {
                "answer": NO_INFO,
                "context": context,
                "top_score": top_score,
                "retrieved": hits,
                "controls_triggered": triggered,
            }

        # 기본 생성
        answer = self._generate(query, context)

        # 제어 2: 충실성(faithfulness) 검증
        if faithfulness_check and not self._faithfulness_ok(context, answer):
            triggered.append("faithfulness_check")
            answer = NO_GROUND

        # 제어 3: 자기 재검증(self-check)
        if self_check and not answer.startswith(("관련 정보 없음", "제공된 문서에 근거", "확실하지 않음")):
            if not self._self_check_ok(query, context, answer):
                triggered.append("self_check")
                answer = UNCERTAIN

        return {
            "answer": answer,
            "context": context,
            "top_score": top_score,
            "retrieved": hits,
            "controls_triggered": triggered,
        }


if __name__ == "__main__":
    # 간단한 동작 확인용 인라인 코퍼스 (실제 실험 코퍼스는 halubench.load_experiment 사용).
    demo_chunks = [
        {"doc_id": "d1", "title": "geo", "text": "The capital of France is Paris."},
        {"doc_id": "d2", "title": "geo", "text": "The Nile is the longest river in Africa."},
        {"doc_id": "d3", "title": "sci", "text": "Water boils at 100 degrees Celsius at sea level."},
    ]
    rag = RAGSystem(demo_chunks)
    print(f"인덱스 구축 완료: 청크 {len(rag.chunks)}개, 임베딩 차원 {rag.matrix.shape}\n")

    for q in ["What is the capital of France?", "Who was the first president of the moon?"]:
        print(f"Q: {q}")
        hits, top = rag.retrieve(q)
        print(f"  top 유사도: {top:.3f}")
        for h in hits:
            print(f"   {h['score']:.3f}  {h['text'][:45]}...")
        out = rag.answer(q)
        print(f"  [기본 RAG] {out['answer'][:120]}...\n")
