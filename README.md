# LLM Knowledge Extractor (FastAPI + SQLite)

A tiny API that accepts unstructured text and returns a 1–2 sentence summary plus structured metadata:
- `title` (if present)
- `topics` (exactly 3)
- `sentiment` (positive | neutral | negative)
- `keywords` (top 3 nouns computed **locally**, not by the LLM)

It also persists every analysis to SQLite and supports simple search over topics/keywords.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # put your OPENAI_API_KEY in .env
uvicorn main:app --reload
