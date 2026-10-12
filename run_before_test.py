"""
跑 before 基线测试：批量调用 /api/decide，统计准确率
结果保存到 v1结果 文件夹
"""
import requests
import json
import time
import os

API_URL = "http://localhost:8000/api/decide"
OUTPUT_DIR = "v1结果"

# 测试用例
test_cases = [
    # 一、明确图场景（应该 need_image = true）
    {"id": 1, "question": "出租屋15平怎么放床和衣柜？", "expected": True, "group": "图场景", "label": "空间关系"},
    {"id": 2, "question": "番茄炒蛋怎么做？", "expected": True, "group": "图场景", "label": "状态转换"},
    {"id": 3, "question": "iPhone 16 Pro、华为Mate 70、小米15 Ultra怎么选？", "expected": True, "group": "图场景", "label": "多维对比"},
    {"id": 4, "question": "这个系统的整体架构是怎样的？", "expected": True, "group": "图场景", "label": "空间关系"},
    {"id": 5, "question": "注册登录的完整流程是什么？", "expected": True, "group": "图场景", "label": "状态转换"},
    {"id": 6, "question": "这三款洗发水的成分对比怎么样？", "expected": True, "group": "图场景", "label": "多维对比"},
    {"id": 7, "question": "客厅刷成灰色是什么效果？", "expected": True, "group": "图场景", "label": "量的变化"},
    {"id": 8, "question": "水流从水泵到喷头的整个链路是怎样的？", "expected": True, "group": "图场景", "label": "空间+状态"},

    # 二、明确字场景（应该 need_image = false）
    {"id": 9, "question": "为什么不建议年轻人裸辞？", "expected": False, "group": "字场景", "label": "否定命题"},
    {"id": 10, "question": "什么是递归？请用大白话解释", "expected": False, "group": "字场景", "label": "抽象定义"},
    {"id": 11, "question": "怎么看待职场内卷这个现象？", "expected": False, "group": "字场景", "label": "抽象思辨"},
    {"id": 12, "question": "牛奶能空腹喝吗？", "expected": False, "group": "字场景", "label": "简单常识"},
    {"id": 13, "question": "为什么这个方案是错的？", "expected": False, "group": "字场景", "label": "否定命题"},
    {"id": 14, "question": "推荐三部适合周末看的电影", "expected": False, "group": "字场景", "label": "信息罗列"},
    {"id": 15, "question": "怎么理解\"沉没成本\"这个概念？", "expected": False, "group": "字场景", "label": "抽象定义"},
    {"id": 16, "question": "今天大庆天气怎么样？", "expected": False, "group": "字场景", "label": "简单问答"},

    # 三、边界场景
    {"id": 17, "question": "三个方案，A不能选因为预算超了，B不能选因为时间不够，所以选C", "expected": False, "group": "边界", "label": "否定优先"},
    {"id": 18, "question": "这个品牌的发展历程是怎样的？", "expected": True, "group": "边界", "label": "时间线"},
    {"id": 19, "question": "简单科普一下光合作用", "expected": False, "group": "边界", "label": "一句话科普"},
    {"id": 20, "question": "这个数学公式怎么推导？", "expected": False, "group": "边界", "label": "抽象推导"},
    {"id": 21, "question": "洗衣机的内部结构是什么样的？", "expected": True, "group": "边界", "label": "空间结构"},
    {"id": 22, "question": "这个设计为什么不好看？", "expected": False, "group": "边界", "label": "否定+审美"},
    {"id": 23, "question": "从北京到哈尔滨自驾怎么走最合理？", "expected": True, "group": "边界", "label": "路径规划"},
    {"id": 24, "question": "怎么判断自己是不是适合考研？", "expected": False, "group": "边界", "label": "决策分析"},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

print("=" * 80)
print("开始跑 before 基线测试...")
print(f"共 {len(test_cases)} 条用例，每条约 80 秒，预计耗时 {len(test_cases)*80//60} 分钟")
print("=" * 80)

for i, case in enumerate(test_cases, 1):
    t0 = time.time()
    try:
        resp = requests.post(API_URL, json={"question": case["question"]}, timeout=120)
        data = resp.json()
        predicted = data.get("need_image", None)
        judge_reason = data.get("judge_reason", "")
        draw_type = data.get("draw_type", "")

        correct = (predicted == case["expected"])
        status = "✅" if correct else "❌"
        elapsed = time.time() - t0

        print(f"{status} [{i:2d}/24] #{case['id']:2d} {case['question'][:25]:<25} | 预期:{str(case['expected']):<5} 实际:{str(predicted):<5} | {elapsed:.0f}s")

        results.append({
            **case,
            "predicted": predicted,
            "correct": correct,
            "judge_reason": judge_reason,
            "draw_type": draw_type,
            "elapsed_seconds": round(elapsed, 1),
        })

        # 每条之间等 1 秒，避免限流
        time.sleep(1)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ [{i:2d}/24] #{case['id']:2d} 请求失败: {str(e)[:50]}")
        results.append({**case, "predicted": None, "correct": False, "judge_reason": str(e), "elapsed_seconds": round(elapsed, 1)})

# ========== 统计 ==========
print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

total = len(results)
valid = [r for r in results if r["predicted"] is not None]
correct_count = sum(1 for r in valid if r["correct"])
print(f"总准确率: {correct_count}/{len(valid)} = {correct_count/len(valid)*100:.1f}% (失败 {total - len(valid)} 条)")

# 分组统计
for group in ["图场景", "字场景", "边界"]:
    group_results = [r for r in valid if r["group"] == group]
    group_correct = sum(1 for r in group_results if r["correct"])
    if group_results:
        print(f"  {group}: {group_correct}/{len(group_results)} = {group_correct/len(group_results)*100:.1f}%")

# 召回率
should_image = [r for r in valid if r["expected"] == True]
recall = sum(1 for r in should_image if r["predicted"] == True) / len(should_image) * 100
print(f"\n召回率（该出图的判对率）: {recall:.1f}% (目标 ≥ 85%)")

# 精确率
predicted_true = [r for r in valid if r["predicted"] == True]
if predicted_true:
    precision = sum(1 for r in predicted_true if r["expected"] == True) / len(predicted_true) * 100
    print(f"精确率（判出图的正确率）: {precision:.1f}% (目标 ≥ 80%)")

# 否定场景拦截率
negative_cases = [r for r in valid if r["label"] in ["否定命题", "否定优先", "否定+审美"]]
if negative_cases:
    neg_correct = sum(1 for r in negative_cases if r["predicted"] == False)
    neg_rate = neg_correct / len(negative_cases) * 100
    print(f"否定场景拦截率: {neg_rate:.1f}% (目标 ≥ 95%)")

# ========== 保存结果 ==========
# 1. JSON 原始数据
json_path = os.path.join(OUTPUT_DIR, "before_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n原始数据已保存: {json_path}")

# 2. Markdown 报告
md_lines = [
    "# Before 基线测试报告",
    f"\n> 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"> 用例总数: {len(test_cases)} 条",
    f"> 成功: {len(valid)} 条",
    f"> 失败: {total - len(valid)} 条",
    "\n---\n",
    "## 核心指标\n",
    f"| 指标 | 数值 | 目标 |",
    f"|------|------|------|",
    f"| 总准确率 | {correct_count/len(valid)*100:.1f}% | - |",
    f"| 召回率 | {recall:.1f}% | ≥ 85% |",
    f"| 精确率 | {precision:.1f}% | ≥ 80% |",
    f"| 否定拦截率 | {neg_rate:.1f}% | ≥ 95% |",
    "\n---\n",
    "## 分组表现\n",
    "| 分组 | 准确率 |",
    "|------|--------|",
]

for group in ["图场景", "字场景", "边界"]:
    group_results = [r for r in valid if r["group"] == group]
    group_correct = sum(1 for r in group_results if r["correct"])
    if group_results:
        md_lines.append(f"| {group} | {group_correct/len(group_results)*100:.1f}% ({group_correct}/{len(group_results)}) |")

md_lines.extend([
    "\n---\n",
    "## Bad Case（判断错误的）\n",
])

bad_cases = [r for r in valid if not r["correct"]]
if bad_cases:
    md_lines.append("| # | 问题 | 预期 | 实际 | 理由 |")
    md_lines.append("|---|------|------|------|------|")
    for r in bad_cases:
        md_lines.append(f"| {r['id']} | {r['question'][:30]} | {r['expected']} | {r['predicted']} | {r['judge_reason'][:50]} |")
else:
    md_lines.append("无 Bad Case 🎉")

md_lines.extend([
    "\n---\n",
    "## 全部用例明细\n",
    "| # | 分组 | 问题 | 预期 | 实际 | 对错 | 耗时 |",
    "|---|------|------|------|------|------|------|",
])

for r in results:
    status = "✅" if r["correct"] else "❌"
    q = r["question"][:25]
    md_lines.append(f"| {r['id']} | {r['group']} | {q} | {r['expected']} | {r['predicted']} | {status} | {r.get('elapsed_seconds', '?')}s |")

md_path = os.path.join(OUTPUT_DIR, "before_report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"测试报告已保存: {md_path}")
