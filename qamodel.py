# qamodel.py
import os
import re
import threading
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import re

HEADING_RX = re.compile(r"^\s*(\d+(?:\.\d+)*\s+.+|[A-Z][A-Z0-9\s\-&,\/]{3,})\s*$")

def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("-", "-").replace("–", "-")
    s = re.sub(r"[^\w\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_sections(doc: str):
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
        body = "\n".join(lines[a+1:b]).strip()
        if body:
            sections.append((title, body))
    return sections

def _find_best_section(query: str, doc: str):
    q = set(_norm(query).split())
    if not q:
        return None
    best = None
    best_score = 0.0
    for title, body in _extract_sections(doc):
        t = set(_norm(title).split())
        # Jaccard overlap + substring bonus
        inter = len(q & t)
        union = max(1, len(q | t))
        score = inter / union
        if _norm(query) in _norm(title):
            score += 0.25
        if score > best_score:
            best_score = score
            best = (title, body)
    return best if best_score >= 0.25 else None

def _split_paragraphs(doc: str):
    paras = re.split(r"\n{2,}|\r\n{2,}", doc)
    # also soft-split very long blocks
    chunks = []
    for p in paras:
        p = p.strip()
        while len(p) > 1200:
            cut = p.rfind(" ", 0, 1200)
            if cut < 400: cut = 1200
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    return [c for c in chunks if c]

def _rank_chunks(question: str, doc: str, top_k: int = 6):
    q_terms = [t for t in _norm(question).split() if len(t) > 2]
    if not q_terms:
        return _split_paragraphs(doc)[:top_k]
    scored = []
    for p in _split_paragraphs(doc):
        terms = _norm(p).split()
        # simple bag-of-words overlap
        score = sum(terms.count(t) for t in q_terms)
        # small boost if query string appears verbatim
        if _norm(question) in _norm(p):
            score += 3
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for s, p in scored[:top_k] if p]

# --- Config ---
MODEL_NAME = "deepset/xlm-roberta-large-squad2"  # multilingual EN/ES
MAX_LEN = 384
STRIDE = 128
TOP_K_CHUNKS = 5
TOKENIZERS_PARALLELISM = os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Globals ---
model = None
tokenizer = None
model_ready = False
model_loading = False
_document_cache = {}

# --- Simple ES+EN stopwords (tiny set; expand if you want) ---
STOPWORDS = set("""
the a an and or of to for in on with by from at as is are was were be been being
what when where who how why which do does did can could should would may might must
que de la el los las un una y o del al para por con sin sobre entre desde como es son fue fueron ser
""".split())

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\wáéíóúüñçàèìòùäëïöü]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def _tokenize_for_scoring(text: str) -> List[str]:
    return [t for t in _normalize(text).split() if t not in STOPWORDS and len(t) > 2]

def _split_into_paragraphs(doc: str) -> List[str]:
    paras = re.split(r"\n{2,}|\r\n{2,}", doc.strip())
    # also break very long paragraphs roughly every ~1200 chars
    chunks = []
    for p in paras:
        p = p.strip()
        while len(p) > 1200:
            cut = p.rfind(" ", 0, 1200)
            if cut < 400: cut = 1200
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    return [c for c in chunks if c]

def _rank_chunks(question: str, doc: str, top_k: int = TOP_K_CHUNKS) -> List[str]:
    q_terms = _tokenize_for_scoring(question)
    if not q_terms:
        return _split_into_paragraphs(doc)[:top_k]
    paras = _split_into_paragraphs(doc)
    scored = []
    for p in paras:
        terms = _tokenize_for_scoring(p)
        score = sum(terms.count(t) for t in q_terms)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for s, p in scored[:top_k] if p]

def load_model():
    global model, tokenizer, model_ready, model_loading
    if model_loading:
        return
    model_loading = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        model_ready = True
        print(f"QA model ready: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_ready = False
    finally:
        model_loading = False

def start_model_loading():
    if not model_ready and not model_loading:
        threading.Thread(target=load_model, daemon=True).start()

start_model_loading()

def _best_span_from_logits(start_logits, end_logits, max_answer_len: int = 32):
    # Search a valid start<=end window up to max_answer_len
    best_score = float("-inf")
    best = (None, None)
    s = start_logits.squeeze(0)
    e = end_logits.squeeze(0)
    L = s.size(0)
    for i in range(L):
        j_max = min(i + max_answer_len, L)
        # pick best end in window [i, j_max)
        j = torch.argmax(e[i:j_max]).item() + i
        if j >= i:
            score = s[i].item() + e[j].item()
            if score > best_score:
                best_score = score
                best = (i, j)
    return best, best_score

def _qa_over_text(question: str, context: str) -> Optional[str]:
    enc = tokenizer(
        question,
        context,
        max_length=MAX_LEN,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_tensors="pt",
        padding="max_length",
    )
    n = enc["input_ids"].size(0)
    best_text, best_score = "", float("-inf")
    with torch.no_grad():
        for i in range(n):
            inputs = {k: v[i].unsqueeze(0) for k, v in enc.items() if k in ("input_ids","attention_mask","token_type_ids")}
            outputs = model(**inputs)
            (s, e), score = _best_span_from_logits(outputs.start_logits, outputs.end_logits)
            if s is None or e is None:
                continue
            span_ids = enc["input_ids"][i][s:e+1]
            text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
            if text and len(text) > 1 and score > best_score:
                best_text, best_score = text, score
    return best_text or None

# def generate_answer(question: str, context_docs: List[str]) -> str:
#     if not model_ready:
#         return "Model is loading. Try again in a few seconds."
#     if not question or not context_docs:
#         return "Please provide both a question and context."
#
#     full_doc = " ".join(context_docs)
#     # tiny cache
#     key = (_normalize(question), hash(full_doc))
#     if key in _document_cache:
#         return _document_cache[key]
#
#     # 1) retrieve top chunks by keyword overlap
#     candidates = _rank_chunks(question, full_doc, top_k=TOP_K_CHUNKS)
#     # 2) run QA over each candidate and keep best
#     best, score = "", float("-inf")
#     for c in candidates:
#         ans = _qa_over_text(question, c)
#         if ans and len(ans) > 1:
#             # quick heuristic: prefer longer-but-sane answers
#             sc = len(ans)
#             if sc > score:
#                 best, score = ans, sc
#
#     if not best:
#         # friendly fallback
#         q = _normalize(question)
#         if q in ("hi","hello","hey"):
#             return "¡Hola! / Hello! ¿En qué puedo ayudarte?"
#         return "No encontré una respuesta específica en el documento. Intenta reformular la pregunta."
#
#     _document_cache[key] = best
#     return best
def generate_answer(question: str, context_docs: list) -> str:
    if not model_ready:
        return "Model is loading. Try again in a few seconds."
    if not question or not context_docs:
        return "Please provide both a question and context."
    question = question.lower()
    if question == 'hi':
        return 'hello'

    q_text = question.strip()
    doc = " ".join(context_docs).strip()

    # 1) Topic/heading queries (short and no '?') → return that section
    if "?" not in q_text and len(q_text.split()) <= 6:
        sec = _find_best_section(q_text, doc)
        if sec:
            title, body = sec
            # return a trimmed section so Postman isn't flooded
            snippet = body if len(body) < 1500 else body[: body.rfind(".", 0, 1500)] + "…"
            return f"{title}\n\n{snippet}"

    # 2) Retrieve top paragraphs, then run QA on those (not the whole doc)
    candidates = _rank_chunks(q_text, doc, top_k=6)

    best_answer = ""
    best_score = float("-inf")
    max_len = 384
    stride = 128
    max_answer_len = 48

    with torch.no_grad():
        for ctx in candidates:
            enc = tokenizer(
                q_text,
                ctx,
                max_length=max_len,
                truncation="only_second",
                stride=stride,
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

                out = model(**inputs)
                s_logits = out.start_logits[0]
                e_logits = out.end_logits[0]
                # search best valid start<=end with length cap
                local_best = ""
                local_score = float("-inf")
                L = s_logits.size(0)
                for s in range(L):
                    e_max = min(s + max_answer_len, L)
                    e = torch.argmax(e_logits[s:e_max]).item() + s
                    if e >= s:
                        score = s_logits[s].item() + e_logits[e].item()
                        if score > local_score:
                            ids = enc["input_ids"][i][s:e+1]
                            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                            if text:
                                local_best = text
                                local_score = score
                if local_best and local_score > best_score:
                    best_answer = local_best
                    best_score = local_score

    if best_answer:
        return best_answer

    # 3) QA found nothing → return best-matching section content
    sec = _find_best_section(q_text, doc)
    if sec:
        title, body = sec
        snippet = body if len(body) < 1500 else body[: body.rfind(".", 0, 1500)] + "…"
        return f"{title}\n\n{snippet}"

    # 4) Final fallback
    return "I couldn't find a specific answer in the document. Try a more direct question."
