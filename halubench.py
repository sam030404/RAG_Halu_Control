"""
HaluBench (PatronusAI/HaluBench) 로더.

두 가지 용도로 샘플을 제공한다.
- load_validation_sample : Step 1 (judge 검증) — PASS/FAIL 균형 샘플
    각 행: passage(근거 문서), question, answer(후보 답변), label(PASS/FAIL), source_ds
- load_experiment        : Step 2 (환각 제어 실험) — 검색 코퍼스 + answerable/unanswerable 질문
    일부 passage를 코퍼스에서 제외(held-out)해 "문서에 없는" 질문을 만든다 (SQuAD 2.0식 abstention 평가).
"""

import random

from datasets import load_dataset

DATASET = "PatronusAI/HaluBench"

_raw = None


def _load_raw():
    """HaluBench test split을 한 번만 로드해 캐시."""
    global _raw
    if _raw is None:
        ds = load_dataset(DATASET, split="test")
        _raw = [dict(r) for r in ds]
    return _raw


def load_validation_sample(n=100, seed=42, source=None):
    """Step 1: judge 검증용. PASS/FAIL 을 절반씩 균형 있게 뽑는다."""
    rows = _load_raw()
    if source:
        rows = [r for r in rows if r["source_ds"] == source]
    passes = [r for r in rows if r["label"] == "PASS"]
    fails = [r for r in rows if r["label"] == "FAIL"]
    rng = random.Random(seed)
    rng.shuffle(passes)
    rng.shuffle(fails)
    half = n // 2
    sample = passes[:half] + fails[:half]
    rng.shuffle(sample)
    return sample


def load_experiment(n_corpus=50, n_heldout=50, seed=42, source=None):
    """Step 2: 검색 코퍼스 + 질문 세트를 만든다.

    - 코퍼스: n_corpus개 passage (해당 질문은 answerable, in_doc=True)
    - held-out: n_heldout개 passage는 코퍼스에서 제외 (해당 질문은 unanswerable, in_doc=False)
    반환: (chunks, questions)
      chunks    = [{doc_id, title, text}]  (RAGSystem 입력 형식)
      questions = [{q, in_doc}]
    """
    rows = _load_raw()
    if source:
        rows = [r for r in rows if r["source_ds"] == source]

    # passage 중복 제거 (같은 passage가 코퍼스와 held-out에 동시에 들어가는 것 방지)
    seen, uniq = set(), []
    for r in rows:
        if r["passage"] in seen:
            continue
        seen.add(r["passage"])
        uniq.append(r)

    rng = random.Random(seed)
    rng.shuffle(uniq)
    corpus_rows = uniq[:n_corpus]
    heldout_rows = uniq[n_corpus:n_corpus + n_heldout]

    chunks = [
        {"doc_id": r["id"], "title": r["source_ds"], "text": r["passage"]}
        for r in corpus_rows
    ]
    questions = (
        [{"q": r["question"], "in_doc": True} for r in corpus_rows]
        + [{"q": r["question"], "in_doc": False} for r in heldout_rows]
    )
    rng.shuffle(questions)
    return chunks, questions


if __name__ == "__main__":
    from collections import Counter

    rows = _load_raw()
    print(f"HaluBench test 총 {len(rows)}행")
    print("label 분포 :", dict(Counter(r["label"] for r in rows)))
    print("source_ds  :", dict(Counter(r["source_ds"] for r in rows)))

    val = load_validation_sample(100)
    print(f"\n[Step1] 검증 샘플 {len(val)}개, "
          f"label 분포 {dict(Counter(r['label'] for r in val))}")

    chunks, qs = load_experiment(50, 50)
    n_in = sum(1 for q in qs if q["in_doc"])
    print(f"[Step2] 코퍼스 {len(chunks)}개, 질문 {len(qs)}개 "
          f"(answerable {n_in} / unanswerable {len(qs) - n_in})")
    print("  예시 질문:", qs[0]["q"][:60], "| in_doc =", qs[0]["in_doc"])
