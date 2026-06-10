"""
환각 판정기 (LLM judge).

GPT-4o-mini가 답변을 참고 문서와 대조하여 GROUNDED / HALLUCINATED로 판정한다.
"정보 없음/확실하지 않음" 같은 거부(abstention)는 지어낸 내용이 아니므로 GROUNDED로 본다.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
_client = OpenAI()

# 안전한 거부 응답을 나타내는 신호 문구들 — 이런 답변은 환각이 아니다.
_ABSTENTION_MARKERS = (
    "관련 정보 없음",
    "제공된 문서에 근거",
    "확실하지 않음",
    "답변드릴 수 없",
    "답변할 수 없",
    "정보를 찾을 수 없",
    "정보가 없",
    "알 수 없",
)


def is_abstention(answer):
    """답변이 '모른다/정보 없음' 류의 안전한 거부인지 판단."""
    return any(m in answer for m in _ABSTENTION_MARKERS)


def judge(context, question, answer):
    """답변을 GROUNDED / HALLUCINATED 로 판정하여 문자열로 반환."""
    # 거부 응답은 지어낸 내용이 없으므로 판정 없이 GROUNDED 처리.
    if is_abstention(answer):
        return "GROUNDED"

    messages = [
        {
            "role": "system",
            "content": "너는 답변의 환각 여부를 판정하는 평가자다. "
            "주어진 참고 문서(passage)를 기준으로, 이 답변이 문서 내용에 근거하는지, "
            "아니면 문서에 없거나 문서와 모순되는 내용을 담고 있는지 판단하라. "
            "답변이 '모르겠다/관련 정보 없음/확실하지 않음'처럼 정보 부재를 "
            "정직하게 밝힌 경우는 GROUNDED로 본다. 문서만으로 뒷받침되지 않는 "
            "사실 주장이 하나라도 있으면 HALLUCINATED다. "
            "GROUNDED 또는 HALLUCINATED 중 하나로만 답하라.",
        },
        {
            "role": "user",
            "content": f"참고 문서:\n{context}\n\n질문: {question}\n답변: {answer}\n\n"
            "판정(GROUNDED/HALLUCINATED):",
        },
    ]
    resp = _client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return "HALLUCINATED" if "HALLUCINATED" in verdict else "GROUNDED"


if __name__ == "__main__":
    ctx = "- [심폐소생술(CPR) 절차] 5단계: 성인 기준 약 5cm 깊이로, 분당 100~120회 속도로 압박한다."
    samples = [
        ("성인 CPR 가슴 압박 깊이는?", "약 5cm 깊이로 압박합니다.", "GROUNDED 예상"),
        ("골절 응급처치는?", "부목을 대고 움직이지 않게 고정한 뒤 병원으로 이송합니다.", "HALLUCINATED 예상"),
        ("골절 응급처치는?", "관련 정보 없음: 문서에서 찾을 수 없습니다.", "GROUNDED(거부) 예상"),
    ]
    for q, a, expect in samples:
        v = judge(ctx, q, a)
        print(f"[{v:12}] ({expect})  Q={q}  A={a[:30]}")
