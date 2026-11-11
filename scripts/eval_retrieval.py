"""检索效果评测：不依赖大模型，只看“检索对不对 + 阈值分流对不对”。

用法：
    python scripts/eval_retrieval.py                 # 评测全部用例
    python scripts/eval_retrieval.py -k 5            # 调整 top_k
    python scripts/eval_retrieval.py --sweep         # 扫一遍阈值，看分流边界

输出的核心指标：
    Hit@1 / Hit@k   —— 首个/前 k 个结果是否命中标注文档
    MRR             —— 正确文档排名倒数的均值
    Refuse-Acc      —— 知识库外的提问被正确拒答的比例（衡量“不乱答”的能力）
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import RAGPipeline, Route, load_settings  # noqa: E402

# (问题, 期望命中的文档；None 表示知识库里没有，应当拒答/追问)
CASES: list[tuple[str, str | None]] = [
    ("本科生国家奖学金一年多少钱？", "奖助学金评定办法.md"),
    ("国家助学金分几个档次？", "奖助学金评定办法.md"),
    ("奖学金评定结果什么时候公示？", "奖助学金评定办法.md"),
    ("图书馆一次最多能借几本书？", "图书馆借阅与座位管理办法.md"),
    ("图书馆的座位怎么预约？", "图书馆借阅与座位管理办法.md"),
    ("宿舍晚上几点熄灯断网？", "学生宿舍管理规定.md"),
    ("宿舍里能用大功率电器吗？", "学生宿舍管理规定.md"),
    ("校园卡丢了该怎么补办？", "校园卡使用与补办指南.md"),
    ("宿舍门禁卡和校园卡是同一张吗？", "校园卡使用与补办指南.md"),
    ("期末考试的补退选什么时候截止？", "选课与考试管理办法.md"),
    ("考试作弊会怎么处理？", "选课与考试管理办法.md"),
    ("重修需要重新交学费吗？", "选课与考试管理办法.md"),
    ("实验室里浓硫酸溅到手上怎么办？", "实验室安全守则.md"),
    ("危险化学品是怎么管理的？", "实验室安全守则.md"),
    ("实验室起火应该打什么电话？", "实验室安全守则.md"),
    # ---- 知识库外的提问：应当被阈值挡下 ----
    ("请帮我写一段快速排序的 Python 代码", None),
    ("今天北京的天气怎么样？", None),
    ("推荐几部好看的科幻电影", None),
    ("如何评价量子纠缠现象？", None),
]


def _load_pipeline(threshold: float | None, top_k: int) -> RAGPipeline:
    settings = load_settings()
    overrides = {"top_k": top_k}
    if threshold is not None:
        overrides["similarity_threshold"] = threshold
    return RAGPipeline(replace(settings, **overrides))


def evaluate(pipeline: RAGPipeline, top_k: int) -> dict:
    hits_at_1 = hits_at_k = 0
    rr_total = 0.0
    questions = 0
    refusals = correct_refusals = 0
    rows = []

    for question, expected in CASES:
        result = pipeline.search(question, top_k=top_k)
        names = [chunk.source for chunk in result.candidates]

        if expected is None:
            refusals += 1
            ok = result.route is not Route.ANSWER
            correct_refusals += int(ok)
            rows.append((question, "-", result.route.value, result.candidates[0].score if result.candidates else 0.0, "OK" if ok else "NG 误答"))
            continue

        questions += 1
        rank = names.index(expected) + 1 if expected in names else None
        hits_at_1 += int(rank == 1)
        hits_at_k += int(rank is not None)
        rr_total += 1.0 / rank if rank else 0.0
        answered = result.route is Route.ANSWER
        mark = "OK" if (rank == 1 and answered) else ("PART" if rank else "NG")
        rows.append((question, expected.replace(".md", ""), result.route.value, result.candidates[0].score if result.candidates else 0.0, mark))

    width = max(len(q) for q, *_ in rows) + 2
    print(f"{'问题':<{width}}{'期望文档':<22}{'分流':<9}{'top1分':>7}  判定")
    print("-" * (width + 45))
    for question, doc, route, score, mark in rows:
        print(f"{question:<{width}}{doc:<22}{route:<9}{score:>7.3f}  {mark}")

    return {
        "hit@1": hits_at_1 / questions if questions else 0.0,
        "hit@k": hits_at_k / questions if questions else 0.0,
        "mrr": rr_total / questions if questions else 0.0,
        "refuse_acc": correct_refusals / refusals if refusals else 0.0,
        "n": questions,
        "n_ood": refusals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测检索与阈值分流效果")
    parser.add_argument("-k", "--top-k", type=int, default=4, choices=range(1, 11), metavar="K")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖默认阈值（默认 0.35）")
    parser.add_argument("--sweep", action="store_true", help="扫描阈值，观察 ANSWER 比例变化")
    args = parser.parse_args(argv)

    pipeline = _load_pipeline(args.threshold, args.top_k)
    if pipeline.store.count() == 0:
        print("知识库为空，请先运行：python scripts/build_index.py")
        return 1

    print(f"向量模型：{pipeline.embedder.status}｜库内片段：{pipeline.store.count()}｜top_k={args.top_k}\n")
    metrics = evaluate(pipeline, args.top_k)
    print(
        f"\nHit@1 = {metrics['hit@1']:.1%}   Hit@{args.top_k} = {metrics['hit@k']:.1%}   "
        f"MRR = {metrics['mrr']:.3f}   拒答准确率 = {metrics['refuse_acc']:.1%}"
        f"（{metrics['n']} 条知识库内问题，{metrics['n_ood']} 条库外问题）"
    )

    if args.sweep:
        print("\n阈值扫描（同一批问题在不同阈值下如何分流）：")
        print(f"{'阈值':>6}{'answer':>9}{'clarify':>9}{'refuse':>9}")
        for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            counter = Counter(
                pipeline.search(question, top_k=args.top_k, threshold=threshold).route.value
                for question, _ in CASES
            )
            print(f"{threshold:>6.2f}{counter['answer']:>9}{counter['clarify']:>9}{counter['refuse']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
