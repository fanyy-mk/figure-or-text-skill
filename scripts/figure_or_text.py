"""
图字择优 · 智能回答 后端服务
=====================================
FastAPI 实现，提供完整的「判断→回答→配图」链路。

启动方式：
    pip install fastapi uvicorn openai
    python figure_or_text.py

接口：
    POST /api/answer           完整接口：判断 + 回答 + 可选配图
    POST /api/decide           纯判断接口：只返回 need_image 结果
    GET  /health               健康检查
"""

import os
import json
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# 配置区
# ⚠️ Key 一律写在项目根目录的 .env 里，不要写进代码 —— 代码要上 GitHub，
#    .env 已在 .gitignore 中，不会被提交。
# ============================================================

def _load_dotenv():
    """可选加载 .env；没装 python-dotenv 也能跑（走系统环境变量）"""
    try:
        from dotenv import load_dotenv
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(root, ".env"))
    except Exception:
        pass


_load_dotenv()

# 大模型 API 配置（兼容 OpenAI 格式，豆包/通义/DeepSeek 都能接）
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# 生图 API 配置（兼容 OpenAI 格式）
IMAGE_API_BASE = os.getenv("IMAGE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")

if not LLM_API_KEY:
    print("=" * 56)
    print("未检测到 LLM_API_KEY")
    print("请在项目根目录建 .env 文件（内容见 .env.example）：")
    print("    LLM_API_KEY=你的方舟Key")
    print("    LLM_MODEL=ep-xxxxx      # 必须是 ep- 开头的接入点 ID")
    print("=" * 56)

# ============================================================
# Prompt 模板
# ============================================================

MERGED_PROMPT = """你是一个智能回答助手。请同时完成两件事：
1. 先判断这个问题是否需要配图辅助回答
2. 给出完整的文字答案

## 判断规则（按顺序执行）

### 第一层：字优先（命中任意一条，直接判定无需配图）
1. 答案核心是否定命题（反驳、不该、错误、不能、为什么不）→ 不配图
2. 答案核心是纯抽象逻辑（定义、辨析、推导、概念、价值判断）→ 不配图
3. 一句话就能说清的简单事实问答 → 不配图

### 第二层：图场景（命中即用图）
检查答案核心是否涉及以下四种结构之一，且配图能明显提升理解效率：
a. 多维对比：多方案 × 多指标的横向比较 → 对比矩阵/雷达图
b. 空间关系：架构、布局、组成结构、拆解关系 → 平面图/架构图
c. 流程步骤：多步骤操作、动作顺序、操作流程 → 步骤流程图
d. 效果对比：前后状态变化、效果预览 → 前后对比图

### 第三层：默认
都不命中 → 不配图。字是默认选项。

## 补充规则
1. 混合场景：同时命中字优先和图场景，以字优先。
2. 不为了"显得内容丰富"而配图。
3. 否定命题和抽象逻辑场景，绝对不配图。

## 强制规则（必须遵守）
- 如果 need_image = true，draw_type 和 draw_guide 不能为空，必须填全。
- 如果 need_image = false，draw_type 和 draw_guide 必须为空字符串。
- 不能出现 need_image = true 但 draw_type 为空的情况。

## 输出格式（严格 JSON，无额外文字）
{{
  "need_image": true/false,
  "judge_reason": "一句话说明判断依据",
  "dominant_structure": "对比/空间/流程/效果/否定/抽象/默认",
  "answer_text": "完整的文字答案，结构清晰，分点说明，用大白话解释",
  "draw_type": "图类型名称（need_image=true时必填），不需要图则为空字符串",
  "draw_guide": "绘图核心要求：画面主体、必要元素、标注文字、风格要求（need_image=true时必填）；不需要图则为空字符串"
}}"""

ANSWER_PROMPT = """请回答用户的问题，给出完整、准确、有条理的文字答案。

要求：
1. 答案结构清晰，分点说明
2. 语言简洁直白，用大白话解释
3. 如果涉及多方案对比、流程步骤、结构关系，在文字里清晰描述出来
4. 不要主动说"建议配图"，你只需要输出文字答案即可

用户问题：{question}"""

IMAGE_PROMPT_TEMPLATE = """{draw_guide}

风格要求：简洁扁平化示意图，白底黑线条，标注清晰，教学图解风格，不要3D效果，不要艺术渲染，不要多余装饰元素。"""

# ============================================================
# 数据模型
# ============================================================

class QueryRequest(BaseModel):
    question: str

class DecisionResponse(BaseModel):
    need_image: bool
    judge_reason: str
    dominant_structure: str
    draw_type: str = ""
    draw_guide: str = ""

class FullResponse(BaseModel):
    need_image: bool
    judge_reason: str
    answer_text: str
    image_url: str = ""
    image_type: str = ""

# ============================================================
# 大模型调用（兼容 OpenAI 格式）
# ============================================================

def call_llm(prompt: str, system_prompt: str = "", response_format: str = "text") -> str:
    """调用大模型，返回文本结果"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": LLM_MODEL, "messages": messages}
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM 调用失败] {e}")
        return ""

def call_image_gen(prompt: str) -> str:
    """调用生图 API，返回图片 URL"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=IMAGE_API_BASE, api_key=IMAGE_API_KEY)
        resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        return resp.data[0].url
    except Exception as e:
        print(f"[生图失败] {e}")
        return ""

# ============================================================
# 核心逻辑
# ============================================================

def analyze_and_answer(question: str) -> dict:
    """一次调用完成：判断要不要配图 + 生成文字答案"""
    raw = call_llm(
        prompt=question,
        system_prompt=MERGED_PROMPT,
        response_format="json"
    )
    if not raw:
        return {
            "need_image": False,
            "judge_reason": "❌ 大模型调用失败",
            "dominant_structure": "错误",
            "answer_text": "抱歉，回答生成失败。",
            "draw_type": "",
            "draw_guide": ""
        }
    try:
        result = json.loads(raw)
        return {
            "need_image": result.get("need_image", False),
            "judge_reason": result.get("judge_reason", ""),
            "dominant_structure": result.get("dominant_structure", "默认"),
            "answer_text": result.get("answer_text", ""),
            "draw_type": result.get("draw_type", ""),
            "draw_guide": result.get("draw_guide", "")
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "need_image": False,
            "judge_reason": "❌ JSON 解析失败",
            "dominant_structure": "错误",
            "answer_text": raw[:500],
            "draw_type": "",
            "draw_guide": ""
        }

def generate_image(draw_guide: str) -> str:
    """生成配图"""
    prompt = IMAGE_PROMPT_TEMPLATE.format(draw_guide=draw_guide)
    return call_image_gen(prompt)

# ============================================================
# FastAPI 路由
# ============================================================

app = FastAPI(title="图字择优 · 智能回答", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "skill": "figure-or-text", "version": "1.0.0"}

@app.post("/api/decide", response_model=DecisionResponse)
def api_decide(req: QueryRequest):
    """纯判断接口：只返回是否需要配图的判断结果"""
    result = analyze_and_answer(req.question)
    return DecisionResponse(
        need_image=result["need_image"],
        judge_reason=result["judge_reason"],
        dominant_structure=result["dominant_structure"],
        draw_type=result["draw_type"],
        draw_guide=result["draw_guide"]
    )

@app.post("/api/answer", response_model=FullResponse)
def api_answer(req: QueryRequest):
    """完整接口：一次调用判断+回答，可选配图"""
    # 第一步：一次调用搞定 判断 + 文字答案
    result = analyze_and_answer(req.question)

    # 第二步：如果需要图，调用生图
    image_url = ""
    image_type = ""
    if result["need_image"]:
        image_url = generate_image(result["draw_guide"])
        image_type = result["draw_type"]

    return FullResponse(
        need_image=result["need_image"],
        judge_reason=result["judge_reason"],
        answer_text=result["answer_text"],
        image_url=image_url,
        image_type=image_type
    )

# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("图字择优 · 智能回答")
    print("接口地址: http://localhost:8000")
    print("文档地址: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
