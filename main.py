import os
import re
import json
import sqlite3
from datetime import datetime
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Load env ---
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from openai import OpenAI  # openai>=1.0 style
    _openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

# LLM prompt helpers

def summarizer_tool_schema() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "summarize_and_extract_metadata",
            "description": (
                "Return a 1–2 sentence summary and structured metadata for the provided text. "
                "The caller will compute 'keywords' locally, so return an empty list for metadata.keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise 1–2 sentence summary grounded in the provided text."
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": ["string", "null"],
                                "description": "Explicit title if present; otherwise null."
                            },
                            "topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "Exactly 3 short, distinct key topics."
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"],
                                "description": "Overall tone label."
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                                "maxItems": 0,
                                "description": "Leave empty; caller computes locally."
                            }
                        },
                        "required": ["title", "topics", "sentiment", "keywords"]
                    }
                },
                "required": ["summary", "metadata"]
            }
        }
    }

def build_messages(
    text: str,
    language: Optional[str] = None,
    max_summary_sentences: int = 2
) -> list:
    lang_line = f"- Input language hint: {language}\n" if language else ""
    sys = (
        "You analyze ONE block of unstructured text (article/blog/update) and MUST call the "
        "`summarize_and_extract_metadata` tool with the following arguments:\n"
        "  - summary: 1–2 sentences, grounded in the text (no bullets/quotes).\n"
        "  - metadata:\n"
        "      title: explicit title if present; else null.\n"
        "      topics: EXACTLY 3 short, non-redundant key topics (nouns/noun phrases, ≤3 words each).\n"
        "      sentiment: one of {positive, neutral, negative}.\n"
        "      keywords: []  (leave EMPTY; caller computes locally).\n"
        "\n"
        "Constraints:\n"
        "- Do NOT invent facts or titles; ground strictly in the provided text.\n"
        "- Keep topics compact and distinct (avoid synonyms/overlap).\n"
        "- Sentiment must be a single label from {positive, neutral, negative}.\n"
        f"- Max summary sentences: {max_summary_sentences}\n"
        f"{lang_line}"
        "Return ONLY via a tool call to `summarize_and_extract_metadata`."
    )
    user = (
        "Source text:\n"
        "```text\n"
        f"{text}\n"
        "```"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]

_STOPWORDS = set("""
a an the and or but if then else when while of to in on at by for with without from as is are was were be been being have has had do does did not no nor this that these those it its their them they he she you we i me my our your his her who whom which what where why how can could should would may might will just than so such very also into over under up down out about across after before during between through per via than ever more most less least each other same own
""".split())

_VERBISH = set("""
be am is are was were been being do does did done doing have has had having get gets got getting make makes made making go goes went going take takes took taking see sees saw seen seeing say says said saying use uses used using need needs needed needing want wants wanted wanting like likes liked liking write writes wrote written writing
""".split())

_TIMEWORDS = set("""
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
today yesterday tomorrow week month year quarter q1 q2 q3 q4
""".split())

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text)

def extract_top_nouns(text: str, k: int = 3) -> List[str]:
    tokens = [w.lower() for w in _tokenize_words(text)]
    filtered = [
        w for w in tokens
        if len(w) > 2
        and w not in _STOPWORDS
        and w not in _VERBISH
        and w not in _TIMEWORDS
    ]
    # Boost for Capitalized words in original text (treated as possible nouns)
    caps_lower = {c.lower() for c in re.findall(r"\b([A-Z][a-zA-Z]+)\b", text)}
    counts = Counter(filtered)
    for w in caps_lower:
        if w in counts:
            counts[w] += 1
    return [w for w, _ in counts.most_common(k)]


def call_llm(text: str, language: Optional[str], max_summary_sentences: int) -> Tuple[str, Dict[str, Any]]:
    """
    Calls OpenAI with tool choice to get summary + metadata.
    Returns (summary, metadata_dict_without_keywords).
    Raises RuntimeError on failure.
    """
    if _openai_client is None:
        raise RuntimeError("OPENAI_API_KEY not set or OpenAI client unavailable.")

    messages = build_messages(text, language, max_summary_sentences)

    try:
        completion = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=[summarizer_tool_schema()],
            tool_choice={"type": "function", "function": {"name": "summarize_and_extract_metadata"}},
            temperature=0.2,
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")

    summary = ""
    meta = {"title": None, "topics": [], "sentiment": "neutral", "keywords": []}

    # Parse the first tool call, if present
    try:
        if completion.choices:
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if tc.type == "function" and tc.function and tc.function.name == "summarize_and_extract_metadata":
                    args_str = tc.function.arguments or "{}"
                    args = json.loads(args_str)
                    summary = (args.get("summary") or "").strip()
                    md = args.get("metadata") or {}
                    meta["title"] = md.get("title", None)
                    meta["topics"] = (md.get("topics") or [])[:3]
                    meta["sentiment"] = md.get("sentiment", "neutral")
                    break
    except Exception as e:
        raise RuntimeError(f"Could not parse LLM tool output: {e}")

    if not summary or not meta["topics"]:
        # Sanity check: if the tool didn't produce expected fields, treat as failure.
        raise RuntimeError("LLM returned incomplete result.")

    return summary, meta


DB_PATH = os.getenv("DB_PATH", "analyzer.db")

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            summary TEXT NOT NULL,
            title TEXT,
            topics TEXT NOT NULL,      -- JSON array of 3 strings
            sentiment TEXT NOT NULL,
            keywords TEXT NOT NULL,    -- JSON array of 3 strings
            created_at TEXT NOT NULL   -- ISO timestamp
        )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_analyses_sentiment ON analyses(sentiment)")
        # For LIKE search across topics/keywords JSON strings (simple approach)
        c.execute("CREATE INDEX IF NOT EXISTS idx_analyses_topics ON analyses(topics)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_analyses_keywords ON analyses(keywords)")

def insert_analysis(
    text: str,
    summary: str,
    title: Optional[str],
    topics: List[str],
    sentiment: str,
    keywords: List[str],
) -> Dict[str, Any]:
    created_at = datetime.utcnow().isoformat()
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO analyses (text, summary, title, topics, sentiment, keywords, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            text,
            summary,
            title,
            json.dumps(topics),
            sentiment,
            json.dumps(keywords),
            created_at
        ))
        _id = cur.lastrowid
        row = {
            "id": _id,
            "text": text,
            "summary": summary,
            "title": title,
            "topics": topics,
            "sentiment": sentiment,
            "keywords": keywords,
            "created_at": created_at
        }
        return row

def search_by_topic_or_keyword(q: str) -> List[Dict[str, Any]]:
    """
    Very lightweight search: string-match the query inside topics/keywords JSON.
    """
    like = f"%{q.lower()}%"
    with _conn() as c:
        cur = c.execute("""
            SELECT id, text, summary, title, topics, sentiment, keywords, created_at
            FROM analyses
            WHERE lower(topics) LIKE ? OR lower(keywords) LIKE ?
            ORDER BY created_at DESC
        """, (like, like))
        results = []
        for r in cur.fetchall():
            results.append({
                "id": r["id"],
                "text": r["text"],
                "summary": r["summary"],
                "title": r["title"],
                "topics": json.loads(r["topics"]),
                "sentiment": r["sentiment"],
                "keywords": json.loads(r["keywords"]),
                "created_at": r["created_at"]
            })
        return results

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Unstructured text to analyze.")
    language: Optional[str] = Field(None, description="Optional language hint (e.g., 'en').")
    max_summary_sentences: int = Field(2, ge=1, le=3, description="Max sentences in summary (1–3).")

class AnalyzeResponse(BaseModel):
    id: int
    summary: str
    metadata: Dict[str, Any]
    created_at: str

class SearchResponseItem(BaseModel):
    id: int
    summary: str
    title: Optional[str]
    topics: List[str]
    sentiment: str
    keywords: List[str]
    created_at: str

# ---------------------------
# Confidence score (bonus, naive)
# ---------------------------

def naive_confidence(summary: str, topics: List[str], keywords: List[str]) -> float:
    """
    A toy heuristic: start at 0.6, add small bumps if we got all 3 topics/keywords and
    the summary length is reasonable. Clamp to [0,1].
    """
    score = 0.6
    if len(topics) == 3:
        score += 0.1
    if len(keywords) == 3:
        score += 0.1
    words = len(summary.split())
    if 15 <= words <= 50:
        score += 0.1
    return max(0.0, min(1.0, round(score, 2)))



app = FastAPI(title="LLM Knowledge Extractor (Jouster)")

@app.on_event("startup")
def _startup():
    init_db()

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' is required and must be non-empty.")

    # 1) Compute keywords locally
    keywords = extract_top_nouns(text, k=3)

    # 2) Call LLM for summary + metadata (title, topics, sentiment)
    try:
        summary, meta = call_llm(text, body.language, body.max_summary_sentences)
    except RuntimeError as e:
        # Robustness: do not crash; return a clean error
        raise HTTPException(status_code=502, detail=str(e))

    # 3) Persist
    title = meta.get("title")
    topics = meta.get("topics") or []
    sentiment = meta.get("sentiment") or "neutral"

    doc = insert_analysis(
        text=text,
        summary=summary,
        title=title,
        topics=topics,
        sentiment=sentiment,
        keywords=keywords,
    )

    # 4) Attach confidence (bonus; not stored to keep schema tiny)
    conf = naive_confidence(summary, topics, keywords)
    doc_out = {
        "id": doc["id"],
        "summary": doc["summary"],
        "metadata": {
            "title": doc["title"],
            "topics": doc["topics"],
            "sentiment": doc["sentiment"],
            "keywords": doc["keywords"],
            "confidence": conf
        },
        "created_at": doc["created_at"],
    }
    return doc_out

@app.get("/search", response_model=List[SearchResponseItem])
def search(topic: str = Query(..., description="Search term matched against topics/keywords")):
    q = (topic or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="'topic' query param is required.")
    rows = search_by_topic_or_keyword(q)
    # Map to response items
    return [
        {
            "id": r["id"],
            "summary": r["summary"],
            "title": r["title"],
            "topics": r["topics"],
            "sentiment": r["sentiment"],
            "keywords": r["keywords"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]

# Optional root for quick sanity
@app.get("/")
def root():
    return {
        "ok": True,
        "endpoints": {
            "POST /analyze": "Analyze and store text",
            "GET /search?topic=xyz": "Search stored analyses by topic/keyword",
        }
    }
