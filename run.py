"""
환각 제어 실험 실행기.

4가지 설정으로 40개 질문을 돌려 환각 비율을 측정하고,
results.csv (요약), details.csv (문항별), results.png (막대그래프)를 저장한다.

  설정1 Baseline      : 기본 RAG (제어 없음)
  설정2 +Faithfulness : + faithfulness_check
  설정3 +Threshold    : + confidence_threshold
  설정4 +All          : 세 가지 제어 모두 적용
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_chunks
from judge import is_abstention, judge
from questions import QUESTIONS
from rag import RAGSystem

# 라벨(영어) → RAG 제어 플래그
CONFIGS = [
    ("Baseline", dict()),
    ("+Faithfulness", dict(faithfulness_check=True)),
    ("+Threshold", dict(confidence_threshold=True)),
    ("+All", dict(faithfulness_check=True, confidence_threshold=True, self_check=True)),
]

MAX_WORKERS = 8


def run_one(rag, item, flags):
    """질문 1개를 주어진 설정으로 실행하고 판정까지 수행."""
    out = rag.answer(item["q"], **flags)
    verdict = judge(out["context"], item["q"], out["answer"])
    return {
        "config": None,  # 호출부에서 채움
        "question": item["q"],
        "in_doc": item["in_doc"],
        "top_score": round(out["top_score"], 3),
        "controls_triggered": "|".join(out["controls_triggered"]),
        "abstained": is_abstention(out["answer"]),
        "verdict": verdict,
        "answer": out["answer"].replace("\n", " ").strip(),
    }


def run_config(rag, label, flags, questions):
    """한 설정으로 전체 질문을 병렬 실행."""
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one, rag, item, flags) for item in questions]
        for fut in futures:
            row = fut.result()
            row["config"] = label
            rows.append(row)
    return rows


def summarize(label, rows):
    total = len(rows)
    hall = sum(1 for r in rows if r["verdict"] == "HALLUCINATED")
    abst = sum(1 for r in rows if r["abstained"])
    out_rows = [r for r in rows if not r["in_doc"]]
    in_rows = [r for r in rows if r["in_doc"]]
    out_hall = sum(1 for r in out_rows if r["verdict"] == "HALLUCINATED")
    in_hall = sum(1 for r in in_rows if r["verdict"] == "HALLUCINATED")
    # in-doc 질문인데 거부한 경우 = 과잉 거부(false refusal)
    in_refused = sum(1 for r in in_rows if r["abstained"])
    in_answered = len(in_rows) - in_refused - in_hall  # 정상적으로 답한 in-doc
    n_in = max(len(in_rows), 1)
    return {
        "config": label,
        "total": total,
        "hallucinated": hall,
        "abstained": abst,
        "hallucination_rate": round(100 * hall / total, 1),
        "out_of_doc_hallucination_rate": round(100 * out_hall / max(len(out_rows), 1), 1),
        "in_doc_hallucination_rate": round(100 * in_hall / n_in, 1),
        "in_doc_answered": in_answered,
        "in_doc_answer_rate": round(100 * in_answered / n_in, 1),
        "in_doc_false_refusal_rate": round(100 * in_refused / n_in, 1),
    }


def save_details(all_rows, path="details.csv"):
    fields = [
        "config", "question", "in_doc", "top_score",
        "controls_triggered", "abstained", "verdict", "answer",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)


def save_summary(summaries, path="results.csv"):
    fields = [
        "config", "total", "hallucinated", "abstained",
        "hallucination_rate", "out_of_doc_hallucination_rate", "in_doc_hallucination_rate",
        "in_doc_answered", "in_doc_answer_rate", "in_doc_false_refusal_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)


def save_chart(summaries, path="results.png",
               title="Hallucination Control in First-Aid RAG (40-sample pilot)"):
    labels = [s["config"] for s in summaries]
    overall = [s["hallucination_rate"] for s in summaries]
    out_only = [s["out_of_doc_hallucination_rate"] for s in summaries]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar([i - width / 2 for i in x], overall, width,
                label="Overall (40 Q)", color="#d9534f")
    b2 = ax.bar([i + width / 2 for i in x], out_only, width,
                label="Out-of-doc only (20 Q)", color="#f0ad4e")

    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(overall), max(out_only)) * 1.25 + 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"그래프 저장: {path}")


def main():
    # SMOKE=1 이면 2 in-doc + 2 out-of-doc 만으로 빠르게 파이프라인 점검
    if os.environ.get("SMOKE"):
        in_q = [q for q in QUESTIONS if q["in_doc"]][:2]
        out_q = [q for q in QUESTIONS if not q["in_doc"]][:2]
        questions = in_q + out_q
        print(f"[SMOKE MODE] 질문 {len(questions)}개로 점검\n")
    else:
        questions = QUESTIONS

    rag = RAGSystem(get_chunks())
    print(f"인덱스 구축 완료 (청크 {len(rag.chunks)}개). 질문 {len(questions)}개 × 설정 {len(CONFIGS)}개 실행\n")

    all_rows, summaries = [], []
    for label, flags in CONFIGS:
        rows = run_config(rag, label, flags, questions)
        all_rows.extend(rows)
        s = summarize(label, rows)
        summaries.append(s)
        print(f"  {label:14} 환각 {s['hallucinated']:2}/{s['total']} "
              f"({s['hallucination_rate']:.1f}%)  거부 {s['abstained']:2}  "
              f"| out-of-doc 환각 {s['out_of_doc_hallucination_rate']:.1f}%")

    save_details(all_rows)
    save_summary(summaries)
    save_chart(summaries)

    print("\n=== 요약 (환각 비율) ===")
    base = summaries[0]["hallucination_rate"]
    for s in summaries:
        delta = s["hallucination_rate"] - base
        print(f"  {s['config']:14} {s['hallucination_rate']:5.1f}%  (Δ {delta:+.1f}p)")
    print("\n저장 완료: results.csv, details.csv, results.png")


if __name__ == "__main__":
    main()
