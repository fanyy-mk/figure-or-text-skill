"""
#0 · before 基线跑批（基于 v1）
=====================================
用 v1 的判断逻辑跑一遍现有 24 条用例，存档 before 数字，供 v2 改造后做 A/B 对比。

用法：
    pip install openai python-dotenv
    # 方式一：建 .env 文件填 Key（推荐，.gitignore 已忽略 .env）
    # 方式二：命令行 set LLM_API_KEY=xxx && set LLM_MODEL=ep-xxx
    python scripts/run_baseline_v1.py              # 正式跑
    python scripts/run_baseline_v1.py --dry-run    # 不调 API，验证统计逻辑
    python scripts/run_baseline_v1.py --limit 3    # 先试跑前 3 条

统计口径：
    - 主准确率只算 16 条「明确图 / 明确字」用例，有唯一正确答案
    - 8 条边界用例单独列出，不计入准确率（它们没有唯一答案）
    - 否定拦截率单独算含否定的用例：#9 #13 #17 #22
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re
import urllib.request

# 允许用 .env 存 Key；没装 dotenv 也能跑（走系统环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v1结果")

# ============================================================
# 24 条用例 —— 抽自 references/test_cases.md
# expect: image / text / boundary（boundary 有倾向但不计分）
# ============================================================

CASES = [
    # 一、明确图（8）
    (1,  "出租屋 15 平怎么放床和衣柜？",                      "image", "空间关系"),
    (2,  "番茄炒蛋怎么做？",                                  "image", "流程步骤"),
    (3,  "iPhone 16 Pro、华为 Mate 70、小米 15 Ultra 怎么选？", "image", "多维对比"),
    (4,  "这个系统的整体架构是怎样的？",                       "image", "空间关系"),
    (5,  "注册登录的完整流程是什么？",                         "image", "流程步骤"),
    (6,  "这三款洗发水的成分对比怎么样？",                     "image", "多维对比"),
    (7,  "客厅刷成灰色是什么效果？",                           "image", "效果对比"),
    (8,  "水流从水泵到喷头的整个链路是怎样的？",                "image", "链路+流程"),
    # 二、明确字（8）
    (9,  "为什么不建议年轻人裸辞？",                           "text",  "否定命题"),
    (10, "什么是递归？请用大白话解释",                         "text",  "抽象定义"),
    (11, "怎么看待职场内卷这个现象？",                         "text",  "价值判断"),
    (12, "牛奶能空腹喝吗？",                                   "text",  "一句话问答"),
    (13, "为什么这个方案是错的？",                             "text",  "否定命题"),
    (14, "推荐三部适合周末看的电影",                           "text",  "信息罗列"),
    (15, "怎么理解「沉没成本」这个概念？",                      "text",  "概念辨析"),
    (16, "今天大庆天气怎么样？",                               "text",  "一句话问答"),
    # 三、边界（8）—— 不计入主准确率
    (17, "三个方案，A 不能选因为预算超了，B 不能选因为时间不够，所以选 C", "boundary", "否定优先 vs 对比（倾向字）"),
    (18, "这个品牌的发展历程是怎样的？",                       "boundary", "时间线（倾向图）"),
    (19, "简单科普一下光合作用",                               "boundary", "答法不同结论不同（摇摆）"),
    (20, "这个数学公式怎么推导？",                             "boundary", "抽象推导（倾向字）"),
    (21, "洗衣机的内部结构是什么样的？",                       "boundary", "空间结构（倾向图）"),
    (22, "这个设计为什么不好看？",                             "boundary", "否定+审美（倾向字）"),
    (23, "从北京到哈尔滨自驾怎么走最合理？",                   "boundary", "空间路径（倾向图）"),
    (24, "怎么判断自己是不是适合考研？",                       "boundary", "抽象决策（倾向字）"),
]

NEGATION_IDS = {9, 13, 17, 22}   # 含否定命题的用例


def safe_name(s: str) -> str:
    """把问题变成安全文件名"""
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "", s).strip()
    return (s[:24] or "untitled").rstrip("。？?，, ")


def save_image(idx: int, question: str, url: str) -> str:
    """下载生图结果到 v1结果/，返回相对路径；失败返回空串"""
    if not url:
        return ""
    ext = ".png" if ".png" in url.split("?")[0] else ".jpg"
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{idx:02d}_{safe_name(question)}{ext}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        return os.path.basename(path)
    except Exception as e:
        print(f"    图片下载失败: {type(e).__name__}")
        return ""


def decide(question: str, dry_run: bool) -> dict:
    """调 v1 的判断。失败返回 None，不计分。"""
    if dry_run:
        # 模拟：按问题长度奇偶返回，只为验证统计逻辑
        time.sleep(0.02)
        return {"need_image": len(question) % 2 == 0,
                "judge_reason": "dry-run 模拟",
                "dominant_structure": "模拟"}
    try:
        from figure_or_text import analyze_and_answer
        r = analyze_and_answer(question)
        return {"need_image": bool(r.get("need_image")),
                "judge_reason": r.get("judge_reason", ""),
                "dominant_structure": r.get("dominant_structure", ""),
                "draw_guide": r.get("draw_guide", ""),
                "draw_type": r.get("draw_type", "")}
    except Exception as e:
        print(f"    调用失败: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不调 API，验证统计逻辑")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    ap.add_argument("--sleep", type=float, default=1.0, help="每条之间间隔秒，防限流（并发模式下不使用）")
    ap.add_argument("--workers", type=int, default=4, help="并发数，默认 4（单条 60-90s，串行要半小时）")
    ap.add_argument("--with-image", action="store_true",
                    help="对判成图的用例真的调用生图并下载到 v1结果/（要花钱，默认关）")
    ap.add_argument("--only-ids", nargs="+", type=int, default=None,
                    help="只跑指定题号，如 --only-ids 1 2 3 7（用于只给判成图的题补图）")
    args = ap.parse_args()

    if not args.dry_run:
        missing = [k for k in ("LLM_API_KEY", "LLM_MODEL") if not os.getenv(k)]
        if missing:
            print("=" * 50)
            print("缺少环境变量:", ", ".join(missing))
            print()
            print("两种配法：")
            print("  1) 在项目根目录建 .env 文件（已在 .gitignore 里，不会泄露）")
            print("     LLM_API_KEY=你的Key")
            print("     LLM_MODEL=ep-xxxxx        # 必须是 ep- 开头的接入点 ID")
            print("  2) 命令行临时设置：")
            print("     set LLM_API_KEY=xxx && set LLM_MODEL=ep-xxx")
            print()
            print("只想先看流程：python scripts/run_baseline_v1.py --dry-run")
            print("=" * 50)
            return 1

    if args.only_ids:
        wanted = set(args.only_ids)
        cases = [c for c in CASES if c[0] in wanted]
    elif args.limit:
        cases = CASES[:args.limit]
    else:
        cases = CASES
    real = not args.dry_run
    workers = max(1, args.workers) if real else 1
    print(f"\n开始跑 {'DRY-RUN' if args.dry_run else '正式'}，共 {len(cases)} 条，并发 {workers}\n")

    def one(case):
        idx, q, expect, note = case
        t0 = time.time()
        try:
            r = decide(q, args.dry_run)
        except Exception as e:      # 兜底：单条炸了不能拖垮整批
            r = None
        cost = round(time.time() - t0, 1)
        if r is None:
            return {"id": idx, "question": q, "expect": expect, "note": note,
                    "actual": None, "hit": None, "reason": "调用失败/超时",
                    "structure": "", "elapsed": cost}
        actual = "image" if r["need_image"] else "text"
        hit = None if expect == "boundary" else (actual == expect)

        # 真正生图（可选，默认关——生图要花钱）
        img_file, img_url, img_cost = "", "", 0.0
        if args.with_image and actual == "image":
            try:
                from figure_or_text import generate_image
                t1 = time.time()
                img_url = generate_image(r.get("draw_guide", "") or q)
                img_cost = round(time.time() - t1, 1)
                img_file = save_image(idx, q, img_url)
            except Exception as e:
                print(f"    #{idx:02d} 生图失败: {type(e).__name__} {e}")

        return {"id": idx, "question": q, "expect": expect, "note": note,
                "actual": actual, "hit": hit,
                "reason": r["judge_reason"],
                "structure": r["dominant_structure"], "elapsed": cost,
                "draw_type": r.get("draw_type", ""),
                "image_file": img_file, "image_url": img_url,
                "image_elapsed": img_cost}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, c): c[0] for c in cases}
        for fu in as_completed(futs):
            row = fu.result()
            rows.append(row)
            flag = "" if row["hit"] is None else ("对" if row["hit"] else "错")
            a = row["actual"] or "失败"
            if row.get("image_file"):
                img = f"  [图已存 {row['image_file']}]"
            elif args.with_image and row["actual"] == "image":
                img = "  [生图失败]"
            else:
                img = ""
            print(f"  [{row['id']:02d}] 期望={row['expect']:<8} 实际={a:<6} {flag:<2} "
                  f"{row['elapsed']:>5}s | {row['question'][:20]}{img}")
            # 增量落盘：中途崩了也不丢已完成的部分
            os.makedirs(OUT_DIR, exist_ok=True)
            tmp = os.path.join(OUT_DIR, "_partial_baseline.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sorted(rows, key=lambda x: x["id"]), f, ensure_ascii=False, indent=2)

    rows.sort(key=lambda x: x["id"])

    # ---------------- 统计 ----------------
    scoped = [r for r in rows if r["expect"] in ("image", "text") and r["hit"] is not None]
    n = len(scoped)
    correct = sum(1 for r in scoped if r["hit"])

    truth_img = [r for r in scoped if r["expect"] == "image"]
    pred_img = [r for r in scoped if r["actual"] == "image"]
    recall = sum(1 for r in truth_img if r["actual"] == "image") / len(truth_img) if truth_img else 0
    precision = sum(1 for r in pred_img if r["expect"] == "image") / len(pred_img) if pred_img else 0

    neg = [r for r in rows if r["id"] in NEGATION_IDS and r["actual"] is not None]
    neg_block = sum(1 for r in neg if r["actual"] == "text") / len(neg) if neg else 0

    boundary = [r for r in rows if r["expect"] == "boundary"]

    metrics = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模式": "dry-run" if args.dry_run else "v1-before-baseline",
        "主准确率": round(correct / n, 4) if n else 0,
        "统计条数": n,
        "正确条数": correct,
        "召回率_该图判图": round(recall, 4),
        "精确率_判图中真该图": round(precision, 4),
        "过度生成率": round(1 - precision, 4),
        "否定拦截率": round(neg_block, 4),
        "否定用例数": len(neg),
    }

    print("=" * 52)
    print("#0 BEFORE 基线" + ("（DRY-RUN 模拟数据，不要存档使用）" if args.dry_run else ""))
    print("=" * 52)
    if args.only_ids or args.limit:
        print(f"  ⚠ 只跑了 {len(rows)}/{len(CASES)} 条，下面的指标不代表整体水平\n")
    print(f"  主准确率（{n} 条明确用例）   {metrics['主准确率']:.1%}   ({correct}/{n})")
    print(f"  召回率  该出图的判出图      {recall:.1%}   ({sum(1 for r in truth_img if r['actual']=='image')}/{len(truth_img)})")
    print(f"  精确率  判出图的真该图      {precision:.1%}")
    print(f"  过度生成率                 {1-precision:.1%}")
    print(f"  否定拦截率（#9/13/17/22）  {neg_block:.1%}   ({sum(1 for r in neg if r['actual']=='text')}/{len(neg)})")
    print("-" * 52)
    print("  边界用例（不计分，看倾向是否一致）：")
    for r in boundary:
        a = r["actual"] or "失败"
        print(f"    #{r['id']:02d} {r['question'][:26]:<28} 实际={a:<6} 倾向={r['note']}")
    print("=" * 52)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "dryrun_" if args.dry_run else "before_baseline_"
    out = os.path.join(OUT_DIR, f"{prefix}{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n已存档: {os.path.relpath(out, os.path.dirname(OUT_DIR))}")
    print("这份 JSON 就是 before 基线，v2 改完后用同一批用例再跑一次做对比。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
