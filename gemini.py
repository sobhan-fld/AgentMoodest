# gemini.py
import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_TOKEN")

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 45
MAX_RETRIES = 3
BACKOFF = 1.6

def _extract_headings(text: str):
    """
    Detect numbered headings like '1.', '1.1', etc., and ALL-CAPS lines as section headers.
    Used to ask Gemini to keep everything, same order.
    """
    lines = text.splitlines()
    heads = []
    for ln in lines:
        s = ln.strip()
        if re.match(r"^\d+(?:\.\d+)*\s+.+", s):
            heads.append(s)
        elif s.isupper() and 1 < len(s) <= 80 and len(s.split()) <= 10:
            heads.append(s)
    # dedupe preserving order
    seen = set(); ordered = []
    for h in heads:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered[:200]

PROMPT_TEMPLATE = """You are rewriting a policy/HR document **for extractive QA**.
CRITICAL RULES:
- **Do NOT translate.** Use the **same language** as the original document.
- **Do NOT omit content.** Preserve and cover **every section and subsection**, in the **same order**.
- Keep all numbers, dates, thresholds, names, and conditions exactly (normalize formatting if needed).
- You may restructure into cleaner bullets and short paragraphs for retrieval, but **do not shorten** overall content.
- If a section is present in the original, it **must** be present in the output with the **same heading** (and numbering if any).

You MUST include these headings (exact order) and fully cover each:
{headings_block}

OUTPUT:
- Plain text.
- Keep original headings (and numbering) intact.
- Rephrase for clarity only; keep length comparable or longer than the original.

Rewrite now, following all rules:

---
{original}
---
"""

def _payload(text: str, headings):
    headings_block = "\n".join(f"- {h}" for h in headings) if headings else "- (No headings detected; still cover ALL content in the same order.)"
    prompt = PROMPT_TEMPLATE.format(headings_block=headings_block, original=text)
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 50,
            "topP": 0.9,
            # Very high budget. The API still enforces an internal cap, but we won't limit it here.
            "maxOutputTokens": 8192
        },
    }

def _call_gemini(payload: dict) -> str:
    resp = requests.post(
        API_URL,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    cands = data.get("candidates") or []
    parts = cands[0].get("content", {}).get("parts", []) if cands else []
    out = (parts[0].get("text") if parts and isinstance(parts[0], dict) else "").strip()
    return out

def improve_agent_details(text: str) -> str:
    """
    Rewrite the document for extractive QA with:
    - same language as original,
    - full section coverage (no omissions),
    - same order and headings,
    - not intentionally shorter.
    Returns plain text; falls back to original on persistent failure.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")
    if not api_key:
        raise RuntimeError("GEMINI_TOKEN is not set.")

    headings = _extract_headings(text)
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = _call_gemini(_payload(text, headings))
            if not out:
                raise RuntimeError("No valid content returned from Gemini.")

            # Normalize spacing
            out = out.replace("\r\n", "\n")
            out = "\n\n".join([s.strip() for s in out.split("\n\n") if s.strip()])

            # If output looks significantly shorter, request expansion once.
            if attempt == 1 and len(out) < int(0.95 * len(text)):
                expand_note = (
                    "\n\n[Note to model: Your previous rewrite was too short. "
                    "Expand to fully cover every bullet, rule, and heading. "
                    "Keep the SAME language and headings; do not omit anything.]\n"
                )
                out2 = _call_gemini(_payload(text + expand_note, headings))
                if out2 and len(out2) > len(out):
                    out = out2

            return out
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF ** attempt)
            else:
                return text  # graceful fallback to keep the app usable

    raise RuntimeError(f"Gemini request failed: {last_err}")
