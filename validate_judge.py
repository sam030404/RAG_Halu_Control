"""
Step 1 — LLM judge 신뢰도 검증.

HaluBench 정답 라벨(PASS/FAIL)과 우리 judge.py의 판정(GROUNDED/HALLUCINATED)이
얼마나 일치하는지 측정한다. 이것이 뒤이은 환각 제어 실험(Step 2)의 신뢰 기반이 된다.

  라벨 매핑: PASS -> GROUNDED,  FAIL -> HALLUCINATED
  양성(positive) = HALLUCINATED (탐지 대상)

출력: 정확도/정밀도/재현율/F1, 혼동행렬
저장: judge_validation.csv (문항별), judge_validation.png (혼동행렬)
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halubench import load_validation_sample
from judge import judge

MAX_WORKERS = 8


def gold_of(label):
    return "HALLUCINATED" if label == "FAIL" else "GROUNDED"


def eval_one(row):
    verdict = judge(row["passage"], row["question"], str(row["answer"]))
    gold = gold_of(row["label"])
    return {
        "id": row["id"],
        "source_ds": row["source_ds"],
        "question": row["question"].replace("\n", " ")[:200],
        "gold": gold,
        "judge": verdict,
        "correct": verdict == gold,
    }


def metrics(rows):
    # 양성 = HALLUCINATED
    tp = sum(1 for r in rows if r["gold"] == "HALLUCINATED" and r["judge"] == "HALLUCINATED")
    fn = sum(1 for r in rows if r["gold"] == "HALLUCINATED" and r["judge"] == "GROUNDED")
    fp = sum(1 for r in rows if r["gold"] == "GROUNDED" and r["judge"] == "HALLUCINATED")
    tn = sum(1 for r in rows if r["gold"] == "GROUNDED" and r["judge"] == "GROUNDED")
    n = len(rows)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_confusion(m, path="judge_validation.png"):
    mat = [[m["tp"], m["fn"]], [m["fp"], m["tn"]]]  # rows=gold, cols=pred
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred: HALLUCINATED", "pred: GROUNDED"])
    ax.set_yticks([0, 1], labels=["gold: HALLUCINATED", "gold: GROUNDED"])
    labels = [["TP", "FN (missed halluc.)"], ["FP (false alarm)", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{mat[i][j]}",
                    ha="center", va="center",
                    color="white" if mat[i][j] > max(max(mat)) / 2 else "black",
                    fontsize=11)
    ax.set_title(f"Judge vs HaluBench labels (n={m['n']})\n"
                 f"acc={m['accuracy']*100:.1f}%  F1={m['f1']*100:.1f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"혼동행렬 저장: {path}")


def save_details(rows, path="judge_validation.csv"):
    fields = ["id", "source_ds", "question", "gold", "judge", "correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    n = int(os.environ.get("VAL_N", "100"))
    sample = load_validation_sample(n=n)
    print(f"Step 1: judge 검증 — HaluBench 샘플 {len(sample)}개 (PASS/FAIL 균형)\n")

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        rows = list(ex.map(eval_one, sample))

    m = metrics(rows)
    save_details(rows)
    save_confusion(m)

    print(f"  정확도(Accuracy)  : {m['accuracy']*100:.1f}%  ({m['tp']+m['tn']}/{m['n']})")
    print(f"  정밀도(Precision) : {m['precision']*100:.1f}%   (환각이라 판정한 것 중 실제 환각)")
    print(f"  재현율(Recall)    : {m['recall']*100:.1f}%   (실제 환각을 잡아낸 비율)")
    print(f"  F1               : {m['f1']*100:.1f}%")
    print(f"  혼동행렬 TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")

    # 출처별 정확도
    print("\n  [출처(source_ds)별 정확도]")
    srcs = sorted(set(r["source_ds"] for r in rows))
    for s in srcs:
        sub = [r for r in rows if r["source_ds"] == s]
        acc = sum(1 for r in sub if r["correct"]) / len(sub)
        print(f"    {s:14} {acc*100:5.1f}%  (n={len(sub)})")

    print("\n저장 완료: judge_validation.csv, judge_validation.png")


if __name__ == "__main__":
    main()
