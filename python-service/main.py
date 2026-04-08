from datetime import datetime, timezone
from typing import List
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AI Special Service")


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


def _openai_rewrite(text: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是改写助手，在保持事实和观点基本一致前提下，降低措辞重复率。"},
            {"role": "user", "content": f"请改写以下文本，保持原意但表达不同:\n{text}"}
        ],
        "temperature": 0.8,
    }

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        if resp.status_code >= 300:
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


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


@app.post("/api/ai/hot-topics", response_model=HotTopicResponse)
def hot_topics(req: HotTopicRequest):
    # 这里模拟“python专项能力(可接入AI)”: 先返回可点击链接，后续可扩展真实爬取+AI总结。
    # 为避免无密钥环境不可用，这里使用稳定兜底数据。
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seeds = {
        "微博": [("微博热搜榜", "https://s.weibo.com/top/summary"), ("微博话题页", "https://s.weibo.com")],
        "知乎": [("知乎热榜", "https://www.zhihu.com/hot"), ("知乎发现", "https://www.zhihu.com/explore")],
        "抖音": [("抖音热点", "https://www.douyin.com/hot"), ("抖音发现", "https://www.douyin.com/discover")],
        "小红书": [("小红书探索", "https://www.xiaohongshu.com/explore"), ("小红书发现", "https://www.xiaohongshu.com")],
        "B站": [("B站热门", "https://www.bilibili.com/v/popular/all"), ("B站排行榜", "https://www.bilibili.com/ranking")],
        "X": [("X Explore", "https://x.com/explore/tabs/trending"), ("X For You", "https://x.com/home")],
        "Reddit": [("Reddit Popular", "https://www.reddit.com/r/popular/"), ("Reddit All", "https://www.reddit.com/r/all/")],
    }
    if req.platform not in seeds:
        raise HTTPException(status_code=400, detail=f"不支持平台: {req.platform}")

    topics = [HotTopicItem(title=t, url=u) for t, u in seeds[req.platform]]
    return HotTopicResponse(platform=req.platform, date=today, topics=topics, note="当前为兜底热点源，可扩展AI抓取")


@app.post("/api/ai/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    ai_text = _openai_rewrite(req.text)
    if ai_text:
        return RewriteResponse(rewrittenText=ai_text, note="由大模型改写")
    return RewriteResponse(rewrittenText=_local_rewrite(req.text), note="未配置OPENAI_API_KEY，使用本地改写兜底")
