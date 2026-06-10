"""
Step 2 — HaluBench 기반 RAG 환각 제어 실험.

HaluBench passage들을 검색 코퍼스로, question을 질문으로 사용해
4가지 설정으로 답변을 생성하고, (Step 1에서 검증된) judge로 환각을 판정한다.
일부 passage는 코퍼스에서 제외(held-out)해 "문서에 없는" 질문을 만든다.

  설정1 Baseline / 설정2 +Faithfulness / 설정3 +Threshold / 설정4 +All

저장: results_halubench.csv (요약), details_halubench.csv (문항별), results_halubench.png
"""

import os

from halubench import load_experiment
from rag import RAGSystem
from run import CONFIGS, run_config, save_chart, save_details, save_summary, summarize

TITLE = "Hallucination Control on HaluBench-RAG (100-sample pilot)"


def main():
    if os.environ.get("SMOKE"):
        chunks, questions = load_experiment(n_corpus=4, n_heldout=4)
        print(f"[SMOKE MODE] 코퍼스 {len(chunks)}개, 질문 {len(questions)}개\n")
    else:
        chunks, questions = load_experiment(n_corpus=50, n_heldout=50)

    rag = RAGSystem(chunks)  # threshold=0.5 기본
    n_in = sum(1 for q in questions if q["in_doc"])
    print(f"코퍼스 {len(rag.chunks)}개 passage 인덱싱 완료. "
          f"질문 {len(questions)}개 (answerable {n_in} / unanswerable {len(questions)-n_in}) "
          f"× 설정 {len(CONFIGS)}개\n")

    all_rows, summaries = [], []
    for label, flags in CONFIGS:
        rows = run_config(rag, label, flags, questions)
        all_rows.extend(rows)
        s = summarize(label, rows)
        summaries.append(s)
        print(f"  {label:14} 환각 {s['hallucinated']:2}/{s['total']} "
              f"({s['hallucination_rate']:.1f}%)  거부 {s['abstained']:2}  "
              f"| unanswerable 환각 {s['out_of_doc_hallucination_rate']:.1f}%  "
              f"answerable 정답률 {s['in_doc_answer_rate']:.1f}%")

    save_details(all_rows, "details_halubench.csv")
    save_summary(summaries, "results_halubench.csv")
    save_chart(summaries, "results_halubench.png", title=TITLE)

    print("\n=== 요약 (환각 비율) ===")
    base = summaries[0]["hallucination_rate"]
    for s in summaries:
        print(f"  {s['config']:14} {s['hallucination_rate']:5.1f}%  "
              f"(Δ {s['hallucination_rate']-base:+.1f}p)")
    print("\n저장 완료: results_halubench.csv, details_halubench.csv, results_halubench.png")


if __name__ == "__main__":
    main()
