# qamodel.py
# Multilingual extractive QA (EN/ES) with retrieval, section matching, and "I don't know" thresholding.
# Optional: Gemini generative fallback using retrieved context (toggle with USE_GEMINI_GENERATION=1).

import os
import re
import threading
from typing import List, Tuple, Optional

# ----------------------------- Config -----------------------------

MODEL_NAME = "deepset/xlm-roberta-base-squad2"  # Multilingual SQuAD2
MAX_LEN = 384
STRIDE = 128
TOP_K_CHUNKS = 6
MAX_ANSWER_LEN = 48

# Confidence controls
NO_ANSWER_MARGIN = 1.5       # span_score - null_score must exceed this
MIN_RETRIEVAL_SCORE = 2      # minimal keyword overlap to consider relevant

# Optional generative fallback with Gemini (answers ONLY from provided context)
USE_GEMINI_GENERATION = os.getenv("USE_GEMINI_GENERATION", "0") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")

# Keep tokenizer quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------- Globals -----------------------------

model = None
tokenizer = None
model_ready = False
model_loading = False

# Lazy heavy imports
torch = None
AutoTokenizer = None
AutoModelForQuestionAnswering = None

# ----------------------------- Loading -----------------------------

def load_model():
    """Load QA model/tokenizer with lazy imports."""
    global model, tokenizer, model_ready, model_loading
    global torch, AutoTokenizer, AutoModelForQuestionAnswering

    if model_loading or model_ready:
        return

    model_loading = True
    try:
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer, AutoModelForQuestionAnswering as _AutoModel

        torch = _torch
        AutoTokenizer = _AutoTokenizer
        AutoModelForQuestionAnswering = _AutoModel

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAME, torch_dtype=_torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        model_ready = True
        print(f"✅ QA model ready: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_ready = False
    finally:
        model_loading = False


def start_model_loading():
    """Start loading in a background thread."""
    if not model_ready and not model_loading:
        threading.Thread(target=load_model, daemon=True).start()

# Autostart; if you prefer to control from app.py, comment this out.
start_model_loading()

# ----------------------------- Text utils -----------------------------

HEADING_RX = re.compile(r"^\s*(\d+(?:\.\d+)*\s+.+|[A-Z][A-Z0-9\s\-&,\/]{3,})\s*$")

# Tiny query expansion to help classification-type questions
QUERY_SYNONYMS = {
    "kind": ["type", "category", "classification", "classes"],
    "kinds": ["types", "categories", "classifications", "classes"],
    "employees": ["staff", "workers", "personnel"],
    "policy": ["rule", "guideline", "standard"],
}

def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _expand_query_terms(q: str) -> List[str]:
    base = _norm(q).split()
    expanded = set(base)
    for t in list(base):
        for syn in QUERY_SYNONYMS.get(t, []):
            expanded.add(syn)
    return [t for t in expanded if len(t) > 2]

def _split_paragraphs(doc: str) -> List[str]:
    paras = re.split(r"\n{2,}|\r\n{2,}", doc.strip())
    chunks = []
    for p in paras:
        p = p.strip()
        while len(p) > 1200:
            cut = p.rfind(" ", 0, 1200)
            if cut < 400:
                cut = 1200
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    return [c for c in chunks if c]

def _extract_sections(doc: str) -> List[Tuple[str, str]]:
    """Return list of (title, content) for numbered or ALL-CAPS headings."""
    lines = doc.splitlines()
    idxs = []
    for i, ln in enumerate(lines):
        if HEADING_RX.match(ln.strip()):
            idxs.append(i)
    if not idxs:
        return [("FULL_DOCUMENT", doc.strip())]

    idxs.append(len(lines))
    sections = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        title = lines[a].strip()
        body = "\n".join(lines[a + 1 : b]).strip()
        if body:
            sections.append((title, body))
    return sections

def _score_query_vs_title(query: str, title: str) -> float:
    """Simple overlap+substring score to match queries to section titles."""
    q = set(_norm(query).split())
    t = set(_norm(title).split())
    if not q or not t:
        return 0.0
    jacc = len(q & t) / max(1, len(q | t))
    sub = 0.25 if _norm(query) in _norm(title) else 0.0
    return min(1.0, jacc + sub)

def _find_best_section(query: str, doc: str) -> Optional[Tuple[str, str, float]]:
    best = None
    best_s = 0.0
    for title, body in _extract_sections(doc):
        sc = _score_query_vs_title(query, title)
        if sc > best_s:
            best_s = sc
            best = (title, body, sc)
    return best

def _rank_chunks(question: str, doc: str, top_k: int = TOP_K_CHUNKS) -> List[Tuple[int, str]]:
    """Very light retrieval: count query-term overlaps (+bonus for exact substring)."""
    q_terms = _expand_query_terms(question)
    paras = _split_paragraphs(doc)
    if not q_terms:
        return [(0, p) for p in paras[:top_k]]

    scored = []
    q_norm = _norm(question)
    for p in paras:
        terms = _norm(p).split()
        score = sum(terms.count(t) for t in q_terms)
        if q_norm in _norm(p):
            score += 3
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def _trim(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind(".", 0, max_chars)
    if cut < 200:
        cut = max_chars
    return text[:cut].rstrip() + "…"

# ----------------------------- QA core -----------------------------

def _best_span_with_null(input_ids_1d, outputs) -> Tuple[Optional[str], float, float]:
    """
    Given 1-D input_ids tensor and model outputs, return:
    (best_text, best_span_score, null_score).
    For SQuAD2 models, null score is approximated by start/end at CLS (index 0).
    """
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Null score (no-answer) from CLS token (index 0)
    null_score = (start_logits[0] + end_logits[0]).item()

    L = start_logits.size(0)
    best_text = ""
    best_score = float("-inf")

    for s in range(L):
        e_max = min(s + MAX_ANSWER_LEN, L)
        # best end within window
        e_rel = torch.argmax(end_logits[s:e_max]).item()
        e = s + e_rel
        if e >= s:
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                ids = input_ids_1d[s : e + 1]  # slice directly on 1-D tensor
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if text:
                    best_text = text
                    best_score = score

    return (best_text if best_text else None), best_score, null_score

# ----------------------------- Optional Gemini fallback -----------------------------

def _gen_answer_with_gemini(question: str, context_blocks: List[str]) -> Optional[str]:
    """
    Use Gemini to synthesize an answer ONLY from the provided context.
    Returns a string or None if unavailable/failed.
    """
    if not (USE_GEMINI_GENERATION and GEMINI_TOKEN):
        return None

    try:
        import requests
        prompt = (
            "Answer the question using ONLY the provided context. "
            "If the context is insufficient, reply exactly: I don't know the answer.\n\n"
            f"Question: {question}\n\n"
            "Context:\n" + "\n\n---\n\n".join(context_blocks[:5])
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_TOKEN}"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512}}
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        cands = data.get("candidates") or []
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        out = (parts[0].get("text") if parts and isinstance(parts[0], dict) else "").strip()
        return out or None
    except Exception:
        return None

# ----------------------------- Public API -----------------------------

def generate_answer(question: str, context_docs: List[str]) -> str:
    """Retrieval → extractive QA → (optional) Gemini → section fallback → 'I don't know'."""
    if not model_ready:
        return "Model is loading. Please wait a few seconds and try again."
    if not question or not context_docs:
        return "Please provide both a question and context."

    q_text = question.strip()
    doc = " ".join(context_docs).strip()

    # Simple greetings
    q_norm = _norm(q_text)
    greetings = {
        "hi": "Hi! How can I help you?",
        "hello": "Hello! How can I help you?",
        "hey": "Hey! How can I help you?",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "good morning": "Good morning! How can I help you?",
        "good afternoon": "Good afternoon! How can I help you?",
        "good evening": "Good evening! How can I help you?",
    }
    if q_norm in greetings:
        return greetings[q_norm]

    # Short topic/heading queries → return that section
    if "?" not in q_text and len(q_text.split()) <= 6:
        found = _find_best_section(q_text, doc)
        if found and found[2] >= 0.25:
            title, body, _ = found
            return _trim(f"{title}\n\n{body}")
        return "I don't know the answer."

    # Retrieve top paragraphs
    ranked = _rank_chunks(q_text, doc, top_k=TOP_K_CHUNKS)
    if not ranked:
        return "I don't know the answer."
    top_score = ranked[0][0]
    if top_score < MIN_RETRIEVAL_SCORE:
        return "I don't know the answer."

    # Extractive QA over retrieved chunks
    best_answer = ""
    best_margin = float("-inf")
    with torch.no_grad():
        for _, ctx in ranked:
            enc = tokenizer(
                q_text,
                ctx,
                max_length=MAX_LEN,
                truncation="only_second",
                stride=STRIDE,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
            n = enc["input_ids"].size(0)
            for i in range(n):
                inputs = {
                    "input_ids": enc["input_ids"][i].unsqueeze(0),
                    "attention_mask": enc["attention_mask"][i].unsqueeze(0),
                }
                if "token_type_ids" in enc:
                    inputs["token_type_ids"] = enc["token_type_ids"][i].unsqueeze(0)

                outputs = model(**inputs)
                text, span_score, null_score = _best_span_with_null(enc["input_ids"][i], outputs)
                margin = span_score - null_score
                if text and margin > best_margin:
                    best_answer = text
                    best_margin = margin

    # If confident, return extractive answer
    if best_answer and best_margin >= NO_ANSWER_MARGIN:
        return best_answer

    # Optional Gemini fallback (synthesis from the same retrieved context)
    gen = _gen_answer_with_gemini(q_text, [ctx for _, ctx in ranked])
    if gen:
        # Normalize a common "I don't know" string to our policy
        if _norm(gen) in {"i don t know the answer", "i don't know the answer"}:
            pass
        else:
            return gen

    # As a last resort, give the closest section snippet (useful for topic-ish questions)
    found = _find_best_section(q_text, doc)
    if found and found[2] >= 0.25:
        title, body, _ = found
        return _trim(f"{title}\n\n{body}")

    return "I don't know the answer."

def get_model_status() -> dict:
    return {
        "model_ready": model_ready,
        "model_loading": model_loading,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_name": MODEL_NAME,
        "gemini_fallback": USE_GEMINI_GENERATION,
    }
