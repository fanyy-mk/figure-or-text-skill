"""
#0 · after 基线跑批（基于 v2） + v1/v2 A/B 对比
=================================================
用 v2 的形态路由跑同一批 24 条用例，跟 before 基线做 A/B。

用法：
    pip install fastapi uvicorn openai python-dotenv
    python scripts/run_baseline_v2.py --dry-run        # 不调 API，验证统计逻辑
    python scripts/run_baseline_v2.py --limit 3        # 先试跑前 3 条
    python scripts/run_baseline_v2.py                  # 正式跑全量
    python scripts/run_baseline_v2.py --workers 6      # 调并发

统计口径（必须和 v1 一致，否则数字不可比）：
    - 主准确率只算 16 条「明确图 / 明确字」用例；8 条边界用例单独列出，不计分
    - 否定拦截率：#9 #13 #17 #22
    - v2 是 6 分类（text/table/diagram/sketch/media/widget），
      做 A/B 时统一二值化：text=无形态产物，其余=有形态产物
    - 分层看两个层面：
        route   路由层 —— decide_form 判成什么形态
        e2e     端到端 —— 最终是否真的产出了可渲染的图（可能被降级）
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT_DIR = os.path.join(ROOT, "v1结果")

from run_baseline_v1 import CASES                     # 复用同一批用例，保证两边完全一致
import figure_or_text_v2 as v2

# v2 六分类 -> 是否与「配图」等价
NON_TEXT = {"table", "diagram", "sketch", "media", "widget"}


def parse_sse(chunk: str):
    """把 answer_stream 产出的 SSE 片段解析成 (event, data)。"""
    out = []
    for block in chunk.strip().split("\n\n"):
        if not block.strip():
            continue
        ev, data = "", {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    data = {}
        if ev:
            out.append((ev, data))
    return out


async def run_one_async(question: str, dry_run: bool) -> dict:
    """跑单条 v2 完整链路，记录分层延迟与产物。"""
    if dry_run:
        fake_forms = ["diagram", "text", "table", "sketch", "media", "widget", "text", "text"]
        form = fake_forms[abs(hash(question)) % len(fake_forms)]
        return {
            "form": form,
            "form_label": v2.FORM_LABELS.get(form, form),
            "reason": "DRY-RUN 模拟",
            "has_figure": form in ("diagram", "table", "widget"),
            "figure_kind": ("mermaid" if form != "table" else "html") if form in NON_TEXT else None,
            "degraded": form in ("sketch", "media"),
            "note": "这张图没能按时生成出来，以上都是文字说明。" if form in ("sketch", "media") else "",
            "answer_chars": 120,
            "code_blocks_in_answer": ["mermaid"] if form in ("diagram", "table", "widget") else [],
            "t_route": 1.2, "t_first": 1.8, "t_total": 6.5,
        }

    t0 = time.time()
    res = {
        "form": v2.FORM_TEXT, "form_label": "", "reason": "",
        "has_figure": False, "figure_kind": None, "degraded": False, "note": "",
        "answer_chars": 0, "code_blocks_in_answer": [],
        "t_route": None, "t_first": None, "t_total": None,
    }
    text_parts = []

    async def collect():
        async for chunk in v2.answer_stream(question):
            for ev, data in parse_sse(chunk):
                now = time.time() - t0
                if ev == "meta":
                    res["form"] = data.get("form", v2.FORM_TEXT)
                    res["form_label"] = data.get("form_label", "")
                    res["reason"] = data.get("reason", "")
                    if res["t_route"] is None:
                        res["t_route"] = round(now, 2)
                elif ev == "delta":
                    text_parts.append(data.get("text", ""))
                    if res["t_first"] is None:
                        res["t_first"] = round(now, 2)
                elif ev == "figure":
                    res["has_figure"] = True
                    res["figure_kind"] = data.get("kind")
                elif ev == "note":
                    res["degraded"] = True
                    res["note"] = data.get("message", "")
                elif ev == "warn":
                    res["note"] = (res["note"] + " / " if res["note"] else "") + data.get("message", "")
                elif ev == "done":
                    res["t_total"] = round(now, 2)

    try:
        await asyncio.wait_for(collect(), timeout=v2.T_TOTAL + 20)
    except asyncio.TimeoutError:
        res["degraded"] = True
        res["note"] = "整体超时，强制中断"
    except Exception as e:
        res["degraded"] = True
        res["note"] = f"链路异常：{e}"

    full = "".join(text_parts)
    res["answer_chars"] = len(full)
    res["code_blocks_in_answer"] = [k for k, _ in v2.extract_code_blocks(full)]
    if res["t_total"] is None:
        res["t_total"] = round(time.time() - t0, 2)
    if res["t_first"] is None:
        res["t_first"] = res["t_total"]
    return res


def run_one(idx, question, expect, note, dry_run) -> dict:
    try:
        r = asyncio.run(run_one_async(question, dry_run))
    except Exception as e:
        r = {"form": v2.FORM_TEXT, "form_label": "", "reason": f"调用失败：{e}",
             "has_figure": False, "figure_kind": None, "degraded": True,
             "note": "调用失败", "answer_chars": 0, "code_blocks_in_answer": [],
             "t_route": None, "t_first": None, "t_total": None}

    route_figure = r["form"] in NON_TEXT          # 路由层：想要图
    e2e_figure = r["has_figure"]                  # 端到端：真的给到了图
    hit = None if expect == "boundary" else (route_figure == (expect == "image"))
    return {
        "id": idx, "question": question, "expect": expect, "note": note,
        "form": r["form"], "form_label": r["form_label"], "reason": r["reason"],
        "route_figure": route_figure, "e2e_figure": e2e_figure,
        "figure_kind": r["figure_kind"], "degraded": r["degraded"], "note_msg": r["note"],
        "answer_chars": r["answer_chars"],
        "code_blocks": r["code_blocks_in_answer"],
        "t_route": r["t_route"], "t_first": r["t_first"], "t_total": r["t_total"],
        "hit": hit,
    }


# ============================================================
# 指标
# ============================================================

def metrics(rows, key="route_figure"):
    def pick(pid):
        return [r for r in rows if r["id"] == pid]

    scored = [r for r in rows if r["expect"] != "boundary"]
    n = len(scored)
    correct = sum(1 for r in scored if r[key] == (r["expect"] == "image"))

    should_img = [r for r in scored if r["expect"] == "image"]
    pred_img = [r for r in scored if r[key]]
    recall = sum(1 for r in should_img if r[key]) / len(should_img) if should_img else 0
    prec = sum(1 for r in pred_img if r["expect"] == "image") / len(pred_img) if pred_img else 0
    over = 1 - prec

    neg_ids = [9, 13, 17, 22]
    neg = [r for r in rows if r["id"] in neg_ids]
    neg_block = sum(1 for r in neg if not r[key]) / len(neg) if neg else 0

    return {
        "主准确率": correct / n if n else 0, "correct": correct, "n": n,
        "召回率": recall, "过度生成率": over, "精确率": prec,
        "否定拦截率": neg_block,
        "pred_img_n": len(pred_img),
    }


def pct(x):
    return f"{x:.1%}"


def secs(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return None
    return min(vals), statistics.median(vals), max(vals)


def fmt_s(t):
    return "—" if t is None else f"{t[0]:.1f} / {t[1]:.1f} / {t[2]:.1f}"


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不调 API，验证统计逻辑")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    ap.add_argument("--workers", type=int, default=4, help="并发数，默认 4")
    ap.add_argument("--only-ids", nargs="*", type=int, default=None, help="只跑指定题号")
    ap.add_argument("--before", default=None, help="指定 before 基线 JSON；默认取最新一份")
    args = ap.parse_args()

    if not args.dry_run and not v2.LLM_API_KEY:
        print("❌ 没读到 LLM_API_KEY。请在项目根目录建 .env：")
        print("     LLM_API_KEY=你的Key")
        print("     LLM_MODEL=ep-xxxxx        # 必须是 ep- 开头的接入点 ID")
        sys.exit(1)

    cases = CASES
    if args.only_ids:
        cases = [c for c in CASES if c[0] in set(args.only_ids)]
    elif args.limit:
        cases = CASES[:args.limit]

    tag = "DRY-RUN 模拟" if args.dry_run else "正式"
    print(f"\n开始跑 v2 after 基线（{tag}），共 {len(cases)} 条，并发 {args.workers}\n")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, i, q, e, n, args.dry_run): i for i, q, e, n in cases}
        for fu in as_completed(futs):
            row = fu.result()
            rows.append(row)
            flag = "" if row["hit"] is None else ("对" if row["hit"] else "错")
            tt = row["t_total"]
            print(f"  [{row['id']:02d}] {row['form']:<8} 图={'有' if row['e2e_figure'] else '无'} "
                  f"{flag:<2} {'' if tt is None else f'{tt:>5.1f}s'} | {row['question'][:24]}")
    rows.sort(key=lambda r: r["id"])

    # ---------- after 指标 ----------
    m_route = metrics(rows, "route_figure")
    m_e2e = metrics(rows, "e2e_figure")

    print("\n" + "=" * 60)
    print("v2 · AFTER 基线" + ("（DRY-RUN 模拟数据，不要存档）" if args.dry_run else ""))
    print("=" * 60)
    print(f"  主准确率（16 条明确）   {pct(m_route['主准确率'])}   ({m_route['correct']}/{m_route['n']})  ← 路由层")
    print(f"                        {pct(m_e2e['主准确率'])}   ({m_e2e['correct']}/{m_e2e['n']})  ← 端到端")
    print(f"  召回率                 {pct(m_route['召回率'])} → {pct(m_e2e['召回率'])}")
    print(f"  过度生成率             {pct(m_route['过度生成率'])} → {pct(m_e2e['过度生成率'])}")
    print(f"  否定拦截率             {pct(m_route['否定拦截率'])}")
    print(f"  降级率（判图但没出图）  {pct(sum(1 for r in rows if r['form'] in NON_TEXT and not r['e2e_figure']) / max(1, sum(1 for r in rows if r['form'] in NON_TEXT)))}")

    t = secs(rows, "t_total")
    f = secs(rows, "t_first")
    rt = secs(rows, "t_route")
    print(f"\n  首字延迟   min/中位/max   {fmt_s(f)} s")
    print(f"  单次总耗时 min/中位/max   {fmt_s(t)} s")
    print(f"  其中判断   min/中位/max   {fmt_s(rt)} s")

    # ---------- 形态分布 ----------
    print("\n--- 形态分布 ---")
    dist = {}
    for r in rows:
        dist[r["form"]] = dist.get(r["form"], 0) + 1
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v} 条")

    # 隐藏缺陷：判成 text 但答案里其实有代码块
    wasted = [r["id"] for r in rows if r["form"] == v2.FORM_TEXT and r["code_blocks"]]
    if wasted:
        print(f"\n  ⚠️ 判为纯文字、但答案里其实生成了代码块的用例：{wasted}")
        print("     （v2 当前实现：form=text 时不去抽代码块，这些图被白白丢掉了）")

    # ---------- 边界用例 ----------
    print("\n--- 边界 8 条（不计分）---")
    for r in rows:
        if r["expect"] == "boundary":
            print(f"  #{r['id']:02d} {r['form']:<8} 倾向={r['note'][:12]:<14} | {r['question'][:26]}")

    # ---------- 存档 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "dryrun_" if args.dry_run else "after_baseline_"
    out = os.path.join(OUT_DIR, f"{prefix}{ts}.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"ts": ts, "version": "v2", "dry_run": args.dry_run,
                   "rows": rows, "metrics_route": m_route, "metrics_e2e": m_e2e},
                  fp, ensure_ascii=False, indent=2)
    print(f"\n已存档 → {os.path.relpath(out, ROOT)}")

    # ---------- A/B 对比 ----------
    if args.dry_run:
        print("\n（DRY-RUN：下面的 A/B 数字是模拟的，只看格式不看得数）")

    before_path = args.before
    if not before_path:
        # 必须挑「用例全」的那份：_1824 只有 7 条（补图时才跑的），
        # 拿它当基准会算出假数字。优先取 rows 最多的 before_baseline_*
        cands = [f for f in os.listdir(OUT_DIR)
                 if f.startswith("before_baseline_") and f.endswith(".json")]
        best, best_n = None, -1
        for f in cands:
            try:
                with open(os.path.join(OUT_DIR, f), encoding="utf-8") as fp:
                    n = len(json.load(fp).get("rows", []))
            except Exception:
                continue
            if n > best_n:
                best, best_n = f, n
        before_path = os.path.join(OUT_DIR, best) if best else None
        if best:
            print(f"（自动选中基准 {best}，共 {best_n} 条用例）")
    if not before_path or not os.path.exists(before_path):
        print("\n没找到 before 基线，跳过 A/B 对比")
        return

    with open(before_path, encoding="utf-8") as fp:
        before = json.load(fp)
    b = {r["id"]: r for r in before["rows"]}
    bm = before.get("metrics", {})

    print("\n" + "=" * 60)
    print(f"A/B 对比   before = {os.path.basename(before_path)}")
    print("=" * 60)
    print(f"{'指标':<20}{'v1 (before)':>14}{'v2 (after)':>14}{'变化':>12}")
    print("-" * 60)

    def bval(*names, default=None):
        """v1 存档里召回率叫「召回率_该图判图」，兼容两种写法。"""
        for n in names:
            v = bm.get(n)
            if v is not None:
                return v
        return default

    rows_ab = [
        ("主准确率(路由层)", bval("主准确率"), m_route["主准确率"]),
        ("主准确率(端到端)", bval("主准确率"), m_e2e["主准确率"]),
        ("召回率", bval("召回率", "召回率_该图判图"), m_route["召回率"]),
        ("过度生成率", bval("过度生成率"), m_route["过度生成率"]),
        ("否定拦截率", bval("否定拦截率"), m_route["否定拦截率"]),
    ]
    for name, bv, av in rows_ab:
        if bv is None:
            print(f"{name:<20}{'—':>14}{pct(av):>14}{'—':>12}")
        else:
            d = av - bv
            print(f"{name:<20}{pct(bv):>14}{pct(av):>14}{d * 100:>+10.1f}pp")

    # 延迟对比
    print("-" * 60)
    bt = secs([{"x": r.get("elapsed")} for r in before["rows"] if r.get("elapsed")], "x")
    print(f"{'单次耗时 中位':<20}{'—' if not bt else f'{bt[1]:.0f}s':>14}"
          f"{'—' if not t else f'{t[1]:.0f}s':>14}{'' if (bt and t) else '':>12}")

    # 逐条对比
    print("\n--- 逐条：两边判断不一致的用例 ---")
    diff = []
    for r in rows:
        bb = b.get(r["id"])
        if not bb or bb.get("actual") is None:
            continue
        b_fig = bb["actual"] == "image"
        if b_fig != r["route_figure"]:
            diff.append((r, bb, b_fig))
    if not diff:
        print("  无（两次判断完全一致）")
    else:
        for r, bb, b_fig in sorted(diff, key=lambda x: x[0]["id"]):
            print(f"  #{r['id']:02d} {r['question'][:24]}")
            print(f"       v1: {'图' if b_fig else '字'} | {bb.get('reason','')[:30]}")
            print(f"       v2: {r['form']:<8} | {r['reason'][:30]}")

    print()


if __name__ == "__main__":
    main()
