"""
Step 2 — HaluBench 기반 RAG 환각 제어 실험 (본 실험).

Step 1(validate_judge.py)에서 judge 신뢰도를 검증한 뒤, 그 검증된 judge로
HaluBench passage를 검색 코퍼스, question을 질문으로 사용해 4가지 설정으로
답변을 생성하고 환각을 판정한다. 일부 passage는 코퍼스에서 제외(held-out)해
"문서에 없는"(unanswerable) 질문을 만든다.

  설정1 Baseline / 설정2 +Faithfulness / 설정3 +Threshold / 설정4 +All

저장: results_halubench.csv (요약), details_halubench.csv (문항별), results_halubench.png

  SMOKE=1 python run_halubench.py  # 8문항으로 파이프라인만 빠르게 점검
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halubench import load_experiment
from judge import is_abstention, judge
from rag import RAGSystem

TITLE = "Hallucination Control on HaluBench-RAG (100-sample pilot)"

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
    # in-doc(answerable) 질문인데 거부한 경우 = 과잉 거부(false refusal)
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


def save_details(all_rows, path="details_halubench.csv"):
    fields = [
        "config", "question", "in_doc", "top_score",
        "controls_triggered", "abstained", "verdict", "answer",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)


def save_summary(summaries, path="results_halubench.csv"):
    fields = [
        "config", "total", "hallucinated", "abstained",
        "hallucination_rate", "out_of_doc_hallucination_rate", "in_doc_hallucination_rate",
        "in_doc_answered", "in_doc_answer_rate", "in_doc_false_refusal_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)


def save_chart(summaries, path="results_halubench.png", title=TITLE, n_out=None):
    labels = [s["config"] for s in summaries]
    overall = [s["hallucination_rate"] for s in summaries]
    out_only = [s["out_of_doc_hallucination_rate"] for s in summaries]

    n_total = summaries[0]["total"] if summaries else 0
    overall_label = f"Overall ({n_total} Q)"
    out_label = f"Out-of-doc only ({n_out} Q)" if n_out else "Out-of-doc only"

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar([i - width / 2 for i in x], overall, width,
                label=overall_label, color="#d9534f")
    b2 = ax.bar([i + width / 2 for i in x], out_only, width,
                label=out_label, color="#f0ad4e")

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
    save_chart(summaries, "results_halubench.png", title=TITLE, n_out=len(questions) - n_in)

    print("\n=== 요약 (환각 비율) ===")
    base = summaries[0]["hallucination_rate"]
    for s in summaries:
        print(f"  {s['config']:14} {s['hallucination_rate']:5.1f}%  "
              f"(Δ {s['hallucination_rate']-base:+.1f}p)")
    print("\n저장 완료: results_halubench.csv, details_halubench.csv, results_halubench.png")


if __name__ == "__main__":
    main()
