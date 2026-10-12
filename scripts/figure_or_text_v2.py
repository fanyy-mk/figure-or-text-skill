"""
图字择优 v2 · 回答形态路由
=====================================
v1 (figure_or_text.py) 保留用于存档 before 基线，不要删。

v2 相对 v1 的改动：
1. 文字流式返回，图后补 —— 用户感知延迟 ≈ 文字延迟，不等图
2. 形态从「要不要图」扩展为六分类路由（预留）
3. 结构化内容走代码块生成（SVG/Mermaid/HTML），服务端渲染校验，坏了就降级
4. 检索优先于生成（可插拔 provider，默认关闭）
5. 分层超时 + 服务端超时，兜底必有返回值

启动：
    pip install fastapi uvicorn openai
    MOCK=1 python figure_or_text_v2.py      # 无 API Key 也能跑通全流程演示
    python figure_or_text_v2.py             # 真实调用，需配好环境变量

接口：
    POST /api/answer/stream   SSE 流式（推荐）
    POST /api/answer          非流式（兼容 v1）
    GET  /health
    GET  /demo/stream?q=      浏览器直接看流式效果
"""

import os
import re
import json
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("figure_or_text_v2")

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

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

IMAGE_API_BASE = os.getenv("IMAGE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")

# MOCK=1 时不调用任何真实 API，用假数据跑通全链路（用来演示/自测链路）
MOCK = os.getenv("MOCK", "0") == "1"

# 是否启用「生图模型」兜底（v2 推荐关掉：结构化内容已由代码块覆盖）
ENABLE_IMAGE_GEN = os.getenv("ENABLE_IMAGE_GEN", "0") == "1"

# ============================================================
# 分层超时（秒）—— 每一层都必须有返回值
# ============================================================

T_DECIDE = 15.0        # 形态判断
T_ANSWER = 30.0        # 流式回答整体
T_RETRIEVE = 3.0       # 检索真实图/视频
T_CODEGEN = 45.0       # 代码块生成
T_CHECK = 5.0          # 渲染校验
T_IMAGE = 45.0         # 生图兜底
T_TOTAL = 60.0         # 整体硬超时，到点必须返回

# ============================================================
# 形态枚举（六分类）
# ============================================================

FORM_TEXT = "text"        # 纯文字
FORM_TABLE = "table"      # 表格
FORM_DIAGRAM = "diagram"  # 结构化图 —— 代码生成
FORM_SKETCH = "sketch"    # 示意图   —— 生图模型
FORM_MEDIA = "media"      # 真实图/视频 —— 检索（正确性最高）
FORM_WIDGET = "widget"    # 交互组件 —— 代码生成

FORM_LABELS = {
    FORM_TEXT: "纯文字",
    FORM_TABLE: "表格",
    FORM_DIAGRAM: "结构化图",
    FORM_SKETCH: "示意图",
    FORM_MEDIA: "真实图/视频",
    FORM_WIDGET: "交互组件",
}

# ============================================================
# Prompt（草稿版，第 4 步会精调）
# ============================================================

ROUTER_PROMPT = """你是回答形态路由专家。判断这个答案用哪种形态呈现，用户最快懂。

候选形态：
- text   纯文字：抽象概念、否定命题、价值判断、一句话能答完
- table  表格：离散维度的多维对比
- diagram 结构化图：流程步骤、结构层级、架构关系 —— 必须输出代码块
- sketch 示意图：真实世界外观/效果预览，且检索不到合适的现成内容
- media  真实图片或视频：描述真实世界外观，且能检索到现成内容（动作演示优先）
- widget 交互组件：需要用户自己试算或试操作

硬规则：
1. 检索优先于生成：能用现成的真实内容就用，别生成
2. 结构化内容（流程/结构/架构）必须给代码块，不能用文字描述代替
3. 代码块必须能被直接渲染，SVG/Mermaid/HTML 三选一
4. 安全性高的领域（动作演示、医疗、操作安全）优先 media，不用 sketch

严格输出 JSON，不要额外文字：
{"form": "text|table|diagram|sketch|media|widget", "reason": "一句话理由", "draw_guide": "需要看图时的核心要求，否则空字符串"}"""

ANSWER_SYSTEM = """你是回答助手。给出完整、准确、有条理的答案，分点说明，用大白话。

重要：当答案涉及流程步骤、结构层级、架构关系或多维对比，且用户说不清楚时，你必须在答案中直接嵌入一个代码块，三选一：
- ```mermaid   流程图/状态图
- ```svg      结构图/示意图（必须是合法 XML）
- ```html     表格或交互组件

代码块要求：
- 必须是能直接渲染的完整代码，不能有省略号或占位符
- 中文标注，字体清晰
- 不要 3D、不要艺术效果、不要多余装饰
不要说"建议配图"，直接把代码块写出来。"""

# ============================================================
# 数据模型
# ============================================================

class QueryRequest(BaseModel):
    question: str

# ============================================================
# 大模型调用
# ============================================================

def _llm_client():
    from openai import OpenAI
    return OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)


def call_llm(prompt: str, system_prompt: str = "", timeout: float = T_DECIDE) -> str:
    """非流式调用，带超时。失败返回空字符串，绝不抛出。"""
    if MOCK:
        return json.dumps({"form": FORM_DIAGRAM, "reason": "MOCK", "draw_guide": "MOCK"}, ensure_ascii=False)
    try:
        client = _llm_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        log.warning(f"LLM 调用失败: {e}")
        return ""


async def stream_llm(prompt: str, system_prompt: str, deadline: Optional[float] = None):
    """异步生成 token。产出 ('delta', str) / ('error', str)，结束时不产出。"""
    loop = asyncio.get_event_loop()

    if MOCK:
        fake = [
            "这是 MOCK 模式的回答。\n\n",
            "第一步：准备工作。\n",
            "第二步：执行操作。\n\n",
            "```mermaid\nflowchart TD\n  A[开始] --> B[处理] --> C[结束]\n```\n",
        ]
        for piece in fake:
            await asyncio.sleep(0.15)
            yield ("delta", piece)
        return

    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            client = _llm_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
                stream=True,
                timeout=T_ANSWER,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    asyncio.run_coroutine_threadsafe(queue.put(("delta", delta)), loop)
        except Exception as e:
            log.warning(f"流式调用失败: {e}")
            try:
                asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
            except Exception:
                pass
        finally:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception:
                pass

    loop.run_in_executor(None, worker)

    while True:
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            yield ("error", "超时")
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            yield ("error", "超时")
            break
        if item is None:
            break
        yield item

# ============================================================
# 代码块提取与渲染校验
# ============================================================

CODE_BLOCK_RE = re.compile(r"```(mermaid|svg|html)\s*\n(.*?)```", re.S | re.I)

VALID_CODE_KINDS = {"mermaid", "svg", "html"}
MERMAID_HEADS = ("graph", "flowchart", "sequencediagram", "classdiagram",
                 "statediagram", "erdiagram", "gantt", "pie", "mindmap",
                 "timeline", "journey", "gitgraph")


def extract_code_blocks(text: str) -> list:
    """提取代码块。返回 [(kind, code), ...]"""
    return [(m.group(1).lower(), m.group(2).strip()) for m in CODE_BLOCK_RE.finditer(text)]


def strip_code_blocks(text: str) -> str:
    """去掉代码块后的正文（图单独返回，正文不重复）"""
    return CODE_BLOCK_RE.sub("", text).strip()


def validate_code(kind: str, code: str) -> tuple:
    """渲染校验。返回 (是否可渲染, 失败原因)"""
    if not code.strip():
        return False, "代码块为空"
    try:
        if kind == "svg":
            import xml.etree.ElementTree as ET
            ET.fromstring(code)
            if "<svg" not in code.lower():
                return False, "不是合法 svg 根元素"
            return True, ""
        if kind == "mermaid":
            head = code.strip().lower().split("\n")[0].strip()
            if not any(head.startswith(h) for h in MERMAID_HEADS):
                return False, f"mermaid 首行关键字不识别: {head[:20]}"
            return True, ""
        if kind == "html":
            low = code.lower()
            if "<div" not in low and "<table" not in low and "<svg" not in low:
                return False, "html 缺少可渲染的根元素"
            return True, ""
    except Exception as e:
        return False, f"校验异常: {e}"
    return False, f"不支持的类型: {kind}"

# ============================================================
# 检索 provider —— 可插拔，默认关闭
# ============================================================

def retrieve_media(query: str) -> Optional[dict]:
    """
    检索真实图/视频。默认返回 None（未接入）。

    接入方式：替换这个函数体，返回一个 dict：
        {"type": "image"/"video", "url": "...", "source": "...", "title": "..."}
    找不到就返回 None，链路会自动降级到生成。
    """
    return None

# ============================================================
# 生图兜底（默认关闭）
# ============================================================

def generate_image(draw_guide: str) -> str:
    if not ENABLE_IMAGE_GEN:
        return ""
    if MOCK:
        return "https://example.com/mock.png"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=IMAGE_API_BASE, api_key=IMAGE_API_KEY)
        prompt = f"{draw_guide}\n\n风格要求：简洁扁平化示意图，白底，线条清晰，中文标注，不要3D不要艺术渲染。"
        resp = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1,
                                      timeout=T_IMAGE)
        return resp.data[0].url or ""
    except Exception as e:
        log.warning(f"生图失败: {e}")
        return ""

# ============================================================
# 形态路由 + 降级链
# ============================================================

async def decide_form(question: str) -> dict:
    """形态判断。任何失败都降级到 text —— 字是永远不会错的兜底。"""
    default = {"form": FORM_TEXT, "reason": "默认纯文字", "draw_guide": ""}
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(call_llm, question, ROUTER_PROMPT, T_DECIDE),
            timeout=T_DECIDE,
        )
        if not raw:
            return {**default, "reason": "判断服务无响应，降级为纯文字"}
        data = json.loads(raw)
        form = data.get("form", FORM_TEXT)
        if form not in FORM_LABELS:
            form = FORM_TEXT
        return {
            "form": form,
            "reason": data.get("reason", ""),
            "draw_guide": data.get("draw_guide", ""),
        }
    except asyncio.TimeoutError:
        log.warning("形态判断超时")
        return {**default, "reason": "判断超时，降级为纯文字"}
    except Exception as e:
        log.warning(f"形态判断失败: {e}")
        return {**default, "reason": "判断失败，降级为纯文字"}


async def build_figure(form: str, draw_guide: str, answer_text: str) -> dict:
    """
    构造图形产物。每一层失败都必须返回明确的东西，不能打转。
    返回 {"figure": {...} | None, "degraded": bool, "message": str}
    """
    # 1) 检索优先：真实内容正确性最高、成本最低
    if form in (FORM_MEDIA, FORM_SKETCH):
        try:
            hit = await asyncio.wait_for(
                asyncio.to_thread(retrieve_media, draw_guide or answer_text[:80]),
                timeout=T_RETRIEVE,
            )
            if hit:
                return {"figure": {"kind": "media", **hit}, "degraded": False, "message": ""}
        except asyncio.TimeoutError:
            log.warning("检索超时，降级")
        except Exception as e:
            log.warning(f"检索失败: {e}")

    # 2) 结构化内容：从答案里抽代码块 + 渲染校验
    blocks = extract_code_blocks(answer_text)
    for kind, code in blocks:
        ok, why = await asyncio.wait_for(
            asyncio.to_thread(validate_code, kind, code), timeout=T_CHECK
        )
        if ok:
            return {
                "figure": {"kind": kind, "code": code},
                "degraded": False,
                "message": "",
            }
        log.warning(f"代码块不可渲染({kind}): {why}")

    # 3) 生图兜底（默认关闭）
    if form in (FORM_SKETCH, FORM_MEDIA) and draw_guide:
        url = await asyncio.wait_for(
            asyncio.to_thread(generate_image, draw_guide), timeout=T_IMAGE
        )
        if url:
            return {"figure": {"kind": "image", "url": url}, "degraded": False, "message": ""}

    # 4) 到底了，明确返回降级说明
    if form in (FORM_DIAGRAM, FORM_TABLE, FORM_WIDGET, FORM_SKETCH, FORM_MEDIA):
        return {
            "figure": None,
            "degraded": True,
            "message": "这张图没能按时生成出来，以上都是文字说明。",
        }
    return {"figure": None, "degraded": False, "message": ""}

# ============================================================
# SSE 流式主链路
# ============================================================

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_stream(question: str):
    started = asyncio.get_event_loop().time()
    deadline = started + T_TOTAL

    def budget() -> float:
        return max(1.0, deadline - asyncio.get_event_loop().time())

    decision = await decide_form(question)
    form = decision["form"]
    yield sse("meta", {"form": form, "form_label": FORM_LABELS[form],
                       "reason": decision["reason"]})

    full_text = ""
    pending: Optional[str] = None   # 代码块缓冲
    buffer = ""
    try:
        async for kind, payload in stream_llm(question, ANSWER_SYSTEM, deadline=deadline):
            if kind == "error":
                yield sse("warn", {"message": "回答生成中断，已返回已有内容"})
                break
            full_text += payload
            buffer += payload

            # 代码块进入缓冲，不直接吐给用户
            while True:
                if pending is None:
                    m = re.search(r"```(mermaid|svg|html)\s*\n", buffer, re.I)
                    if not m:
                        break
                    if m.start() > 0:
                        yield sse("delta", {"text": buffer[:m.start()]})
                    pending = m.group(1).lower()
                    buffer = buffer[m.end():]
                    continue
                end = buffer.find("```")
                if end == -1:
                    break
                buffer = buffer[end + 3:]
                pending = None
                continue

            if pending is None and buffer and "`" not in buffer[-3:]:
                yield sse("delta", {"text": buffer})
                buffer = ""
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning(f"主链路异常: {e}")
        yield sse("warn", {"message": "回答生成异常，已返回已有内容"})

    if pending is None and buffer:
        yield sse("delta", {"text": buffer})

    if not full_text.strip():
        yield sse("delta", {"text": "抱歉，这次没能生成回答，请重试。"})

    # 图后补，失败必有明确降级说明
    needs_figure = form not in (FORM_TEXT,)
    if needs_figure:
        try:
            fig = await asyncio.wait_for(
                build_figure(form, decision["draw_guide"], full_text),
                timeout=min(T_CODEGEN, budget()),
            )
        except asyncio.TimeoutError:
            fig = {"figure": None, "degraded": True, "message": "这张图没能按时生成出来。"}
        except Exception as e:
            log.warning(f"建图失败: {e}")
            fig = {"figure": None, "degraded": True, "message": "这张图没能生成出来。"}

        if fig["figure"]:
            yield sse("figure", fig["figure"])
        elif fig["degraded"]:
            yield sse("note", {"message": fig["message"], "retryable": True})

    yield sse("done", {"degraded": False})

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="图字择优 v2 · 回答形态路由", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "skill": "figure-or-text", "version": "2.0.0", "mock": MOCK}


@app.post("/api/answer/stream")
def api_answer_stream(req: QueryRequest):
    return StreamingResponse(
        answer_stream(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/demo/stream")
def demo_stream(q: str = "西红柿牛腩怎么做"):
    """浏览器直接访问看流式效果"""
    return StreamingResponse(
        answer_stream(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/answer")
async def api_answer(req: QueryRequest):
    """非流式（兼容 v1），内部走同一链路后合并返回"""
    async def collect():
        text, figure, note, meta = "", None, "", {}
        async for chunk in answer_stream(req.question):
            for line in chunk.strip().split("\n"):
                if line.startswith("event: "):
                    ev = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if ev == "meta":
                        meta = data
                    elif ev == "delta":
                        text += data.get("text", "")
                    elif ev == "figure":
                        figure = data
                    elif ev == "note":
                        note = data.get("message", "")
        return {
            "form": meta.get("form", FORM_TEXT),
            "form_label": meta.get("form_label", ""),
            "reason": meta.get("reason", ""),
            "answer_text": text.strip(),
            "figure": figure,
            "note": note,
        }

    try:
        return await asyncio.wait_for(collect(), timeout=T_TOTAL)
    except asyncio.TimeoutError:
        return {
            "form": FORM_TEXT, "form_label": FORM_LABELS[FORM_TEXT],
            "reason": "整体超时", "answer_text": "抱歉，这次回答超时了，请重试。",
            "figure": None, "note": "请求超时",
        }


if __name__ == "__main__":
    import uvicorn
    print("=" * 52)
    print("图字择优 v2 · 回答形态路由")
    print(f"MOCK 模式: {'开' if MOCK else '关'}   生图兜底: {'开' if ENABLE_IMAGE_GEN else '关'}")
    print("流式: http://localhost:8000/demo/stream?q=西红柿牛腩怎么做")
    print("文档: http://localhost:8000/docs")
    print("=" * 52)
    uvicorn.run(app, host="0.0.0.0", port=8000)
