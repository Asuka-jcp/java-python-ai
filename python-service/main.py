from datetime import datetime, timezone
from typing import List, Literal
import json
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AI Special Service")

SUPPORTED_PLATFORMS = ["微博", "知乎", "抖音", "小红书", "B站", "X", "Reddit"]


class HotTopicRequest(BaseModel):
    platform: str = Field(min_length=1)


class HotTopicItem(BaseModel):
    title: str
    url: str


class HotTopicResponse(BaseModel):
    platform: str
    date: str
    topics: List[HotTopicItem]
    note: str | None = None


class RewriteRequest(BaseModel):
    text: str = Field(min_length=50, max_length=30000)


class RewriteResponse(BaseModel):
    rewrittenText: str
    note: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    intent: Literal["hot_topics", "rewrite", "qa"]
    answer: str
    note: str | None = None


def _seed_topics(platform: str) -> list[tuple[str, str]]:
    seeds = {
        "微博": [("微博热搜榜", "https://s.weibo.com/top/summary"), ("微博话题页", "https://s.weibo.com")],
        "知乎": [("知乎热榜", "https://www.zhihu.com/hot"), ("知乎发现", "https://www.zhihu.com/explore")],
        "抖音": [("抖音热点", "https://www.douyin.com/hot"), ("抖音发现", "https://www.douyin.com/discover")],
        "小红书": [("小红书探索", "https://www.xiaohongshu.com/explore"), ("小红书发现", "https://www.xiaohongshu.com")],
        "B站": [("B站热门", "https://www.bilibili.com/v/popular/all"), ("B站排行榜", "https://www.bilibili.com/ranking")],
        "X": [("X Explore", "https://x.com/explore/tabs/trending"), ("X For You", "https://x.com/home")],
        "Reddit": [("Reddit Popular", "https://www.reddit.com/r/popular/"), ("Reddit All", "https://www.reddit.com/r/all/")],
    }
    return seeds[platform]


def _call_openai(messages: list[dict], temperature: float = 0.7) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        if resp.status_code >= 300:
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def _openai_rewrite(text: str) -> str | None:
    return _call_openai(
        [
            {"role": "system", "content": "你是改写助手，在保持事实和观点基本一致前提下，降低措辞重复率。"},
            {"role": "user", "content": f"请改写以下文本，保持原意但表达不同:\n{text}"},
        ],
        temperature=0.8,
    )


def _local_rewrite(text: str) -> str:
    sentences = re.split(r"([。！？!?])", text)
    merged = []
    for i in range(0, len(sentences), 2):
        s = sentences[i].strip()
        punc = sentences[i + 1] if i + 1 < len(sentences) else ""
        if not s:
            continue
        s = s.replace("我们", "本文").replace("因为", "由于").replace("所以", "因此")
        merged.append(f"{s}{punc}")
    return "\n".join(merged)


def _extract_platform(message: str) -> str | None:
    for p in SUPPORTED_PLATFORMS:
        if p.lower() in message.lower():
            return p
    aliases = {
        "bilibili": "B站",
        "b站": "B站",
        "weibo": "微博",
        "zhihu": "知乎",
        "douyin": "抖音",
        "xiaohongshu": "小红书",
        "reddit": "Reddit",
        "twitter": "X",
        "x": "X",
    }
    msg = message.lower()
    for alias, target in aliases.items():
        if alias in msg:
            return target
    return None


def _fallback_route(message: str) -> dict:
    msg = message.strip()
    if len(msg) >= 50 and ("改写" in msg or "重写" in msg):
        return {"intent": "rewrite", "text": msg}

    platform = _extract_platform(msg)
    if platform and ("热点" in msg or "热搜" in msg or "trending" in msg.lower()):
        return {"intent": "hot_topics", "platform": platform}

    return {"intent": "qa", "answer": "你可以让我做两件事：1）查询某个平台今日热点；2）改写一段文章。"}


def _llm_route(message: str) -> dict:
    content = _call_openai(
        [
            {
                "role": "system",
                "content": (
                    "你是意图路由器。请把用户请求识别成 JSON，且只能输出 JSON，不要输出其他文本。"
                    "可选 intent: hot_topics, rewrite, qa。"
                    "hot_topics 需要 platform 字段(微博/知乎/抖音/小红书/B站/X/Reddit之一)。"
                    "rewrite 需要 text 字段(待改写文本)。"
                    "qa 需要 answer 字段(对用户的自然语言回复)。"
                ),
            },
            {"role": "user", "content": message},
        ],
        temperature=0.2,
    )

    if not content:
        return _fallback_route(message)

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("intent") in {"hot_topics", "rewrite", "qa"}:
            return parsed
    except json.JSONDecodeError:
        pass

    return _fallback_route(message)


def _to_natural_hot_topics(platform: str, topics: list[HotTopicItem], note: str | None) -> str:
    lines = [f"已为你整理 {platform} 今日热点："]
    for idx, t in enumerate(topics, start=1):
        lines.append(f"{idx}. {t.title} - {t.url}")
    if note:
        lines.append(f"说明：{note}")
    return "\n".join(lines)


@app.post("/api/ai/hot-topics", response_model=HotTopicResponse)
def hot_topics(req: HotTopicRequest):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if req.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(status_code=400, detail=f"不支持平台: {req.platform}")

    topics = [HotTopicItem(title=t, url=u) for t, u in _seed_topics(req.platform)]
    return HotTopicResponse(platform=req.platform, date=today, topics=topics, note="当前为兜底热点源，可扩展AI抓取")


@app.post("/api/ai/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    ai_text = _openai_rewrite(req.text)
    if ai_text:
        return RewriteResponse(rewrittenText=ai_text, note="由大模型改写")
    return RewriteResponse(rewrittenText=_local_rewrite(req.text), note="未配置OPENAI_API_KEY，使用本地改写兜底")


@app.post("/api/ai/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    route = _llm_route(req.message)
    intent = route.get("intent", "qa")

    if intent == "hot_topics":
        platform = route.get("platform") or _extract_platform(req.message)
        if not platform:
            platform = "微博"
        result = hot_topics(HotTopicRequest(platform=platform))
        return ChatResponse(
            intent="hot_topics",
            answer=_to_natural_hot_topics(result.platform, result.topics, result.note),
            note="由大模型识别意图并自动调用热点接口",
        )

    if intent == "rewrite":
        text = route.get("text") or req.message
        if len(text.strip()) < 50:
            return ChatResponse(
                intent="qa",
                answer="你是想让我改写文章吗？请粘贴至少50字原文，我会在保持原意的前提下重写。",
                note="改写文本不足50字，未调用改写接口",
            )
        result = rewrite(RewriteRequest(text=text))
        return ChatResponse(
            intent="rewrite",
            answer=f"改写完成：\n{result.rewrittenText}",
            note="由大模型识别意图并自动调用改写接口",
        )

    answer = route.get("answer") or "你可以让我查询热点或改写文章。"
    return ChatResponse(intent="qa", answer=answer, note="对话模式")
