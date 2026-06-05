import faiss
import numpy as np
import requests
import re
import os
import time
from datetime import datetime
from nltk.stem import PorterStemmer

# ==========================================
# FILES
# ==========================================

INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

session = requests.Session()

# ==========================================
# LOAD
# ==========================================

index = None
chunks = []


def _load_db():
    global index, chunks
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            chunks = [c.strip() for c in raw.split("\n---CHUNK---\n") if c.strip()]
            print(f"[RAG] Loaded {len(chunks)} chunks")
        else:
            print("[RAG] Warning: Vector DB not found. Run: python ingest.py first.")
    except Exception as e:
        print(f"[RAG] Load error: {e}")


_load_db()

# ==========================================
# EMBEDDING CACHE
# ==========================================

embedding_cache = {}


def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    res = session.post(
        OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30
    )
    emb = res.json()["embedding"]
    embedding_cache[text] = emb
    return emb


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "in",
    "on",
    "at",
    "by",
    "for",
    "of",
    "to",
    "from",
    "with",
    "and",
    "or",
    "but",
    "not",
    "if",
    "as",
    "it",
    "its",
    "this",
    "that",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "tell",
    "give",
    "me",
    "please",
    "about",
    "need",
    "want",
    "also",
    "more",
    "than",
    "only",
    "any",
    "all",
    "some",
}


stemmer = PorterStemmer()


def keywords(text):

    return [
        stemmer.stem(w)
        for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= 3 and w not in STOPWORDS
    ]


def kw_score(qwords, chunk):
    cl = chunk.lower()
    score = sum(2 for w in qwords if w in cl)
    for i in range(len(qwords) - 1):
        if qwords[i] + " " + qwords[i + 1] in cl:
            score += 3
    return score


def parse_chunk_parts(chunk):
    """
    Returns (header_lower, entry_title_lower, entry_number)
    Handles two title formats:
      - POE:  "N. City Office"
      - FRRO: "N. FRRO CityName"  (all-caps acronym + city)
    """
    header = ""
    hm = re.match(r"^\[(.+?)\]", chunk)
    if hm:
        header = hm.group(1).lower()

    title = ""
    number = None
    # Match "N. Title" — title ends at first bullet/field marker
    tm = re.search(r"(\d+)\.\s+([A-Za-z][^\n\r]{1,80})", chunk)
    if tm:
        number = int(tm.group(1))
        raw = tm.group(2)
        # Cut off at first field label or bullet
        raw = re.split(
            r"\s+[•o]\s+|\bLocation\b|\bIn-Charge\b|\bAddress\b"
            r"|\bPhone\b|\bEmail\b|\bContact\b|\bJurisdiction\b",
            raw,
        )[0]
        title = raw.strip().lower()

    return header, title, number


FIELD_ALIASES = {
    "address": ["address"],
    "phone": ["phone"],
    "email": ["email"],
    "incharge": ["in-charge", "incharge", "in charge"],
    "location": ["location"],
    "contact": ["contact"],
    "jurisdiction": ["jurisdiction"],
}

# Domain-specific acronyms that appear in every chunk of their section
# — treated like stopwords for entry-title discrimination
SECTION_ACRONYMS = {"frro", "poe"}


def detect_field_request(query):
    ql = query.lower()
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", ql):
                return field
    return None


def extract_field_from_chunk(chunk, field):
    """
    Pull only the requested field value from the chunk.
    Handles both:
      • Location: value  (bullet format)
      o Address: value   (o-bullet format)
    """
    aliases = FIELD_ALIASES.get(field, [field])
    label_pat = "|".join(re.escape(a) for a in aliases)

    # Pattern: optional bullet (•/o/|), field label, colon, value
    pattern = re.compile(
        r"(?:[•o\|]\s*)(?:"
        + label_pat
        + r")\s*:\s*(.+?)(?=\s*[•o\|]\s*(?:location|address|phone|email|contact|jurisdiction|in-charge|in charge)\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(chunk)
    if m:
        value = re.sub(r"\s+", " ", m.group(1)).strip()
        return value

    # Fallback: simpler match
    m2 = re.search(r"(?:" + label_pat + r")\s*:\s*([^•\n]{5,})", chunk, re.IGNORECASE)
    if m2:
        return re.sub(r"\s+", " ", m2.group(1)).strip()

    return None


# ==========================================
# EXACT ENTRY MATCH
#
# Scores each chunk:
#   +3 per query word found in [HEADER]
#   +5 per query word found in entry title
#
# Returns the SINGLE best chunk + field requested.
# ==========================================


def find_exact_entry(query):
    """
    Find ONE specific entry (e.g. 'Jaipur Office', 'FRRO Bangalore').

    Scoring:
      +5 per query word that is DISCRIMINATING (not generic/acronym).
      +3 per query word found in the header.

    Generic = appears in >50% of all entry titles.
    Acronyms (frro, poe) = always skipped regardless of frequency.
    """
    qwords = keywords(query)
    if not qwords:
        return None, None

    field = detect_field_request(query)

    # Build set of words that appear in >50% of titles → generic
    all_titles = []
    for chunk in chunks:
        _, title, _ = parse_chunk_parts(chunk)
        if title:
            all_titles.append(set(re.findall(r"[a-z]+", title)))

    if not all_titles:
        return None, None

    word_freq = {}
    for ts in all_titles:
        for w in ts:
            word_freq[w] = word_freq.get(w, 0) + 1
    threshold = len(all_titles) / 2
    generic_words = {w for w, cnt in word_freq.items() if cnt > threshold}
    generic_words |= SECTION_ACRONYMS  # always skip frro, poe

    disc_words = [w for w in qwords if w not in generic_words]

    if not disc_words:
        return None, None

    best_score = 0
    best_chunk = None

    for chunk in chunks:
        # Skip Q&A chunks — they are handled by direct_qa_search, not here.
        # A chunk is a Q/A pair when its body starts with Q:
        # or contains a numbered question like "1.What is...?\nAns:"
        body_start = re.search(r"^\[.+?\]\s*", chunk)
        body_text = chunk[body_start.end() :] if body_start else chunk
        if re.search(
            r"^\s*(?:\d+[\.:]\s*)?Q\s*:", body_text, re.IGNORECASE | re.MULTILINE
        ):
            continue
        if re.search(
            r"^\s*\d+[\.:]\s*\S.*?\n\s*(?:Ans|A)\s*:",
            body_text,
            re.IGNORECASE | re.DOTALL,
        ):
            continue

        header, title, _ = parse_chunk_parts(chunk)
        if not title:
            continue

        disc_hits = sum(1 for w in disc_words if w in title)
        if disc_hits == 0:
            continue

        header_score = sum(3 for w in qwords if w in header)
        title_score = disc_hits * 5
        total = header_score + title_score

        if total > best_score:
            best_score = total
            best_chunk = chunk

    if best_chunk:
        return best_chunk, field
    return None, None


# ==========================================
# FORMAT OUTPUT
# Strips [HEADER] and leading number,
# returns clean entry or specific field value.
# ==========================================


def format_entry(chunk, field=None):
    """
    Format a matched chunk for display.

    Rules:
    - If specific field requested (phone, email, address), return only that.
    - If Q:/A: format, return answer only.
    - If FAQ format (Question? Answer...), return answer only.
    - Otherwise return formatted directory/contact entry.
    """

    # Remove [HEADER] prefix
    clean = re.sub(r"^\[.+?\]\s*", "", chunk).strip()

    # Remove leading numbering like "1. "
    clean = re.sub(r"^\d+\.\s+", "", clean).strip()

    # Field-specific extraction
    if field:
        value = extract_field_from_chunk(chunk, field)

        if value:
            label = field.capitalize()
            return f"{label}: {value}"

    # --------------------------------------------------
    # Q:/A: format
    # Example:
    # Q: What is OCI?
    # A: OCI is ...
    # --------------------------------------------------
    a_match = re.search(r"(?:A|Ans)\s*:\s*(.+)", clean, re.DOTALL | re.IGNORECASE)

    q_match = re.search(r"Q\s*:", clean, re.IGNORECASE)

    if q_match and a_match:
        return re.sub(r"\s+", " ", a_match.group(1)).strip()

    # --------------------------------------------------
    # FAQ format
    # Example:
    # How do I submit my OCI application?
    # Applications for registration...
    # --------------------------------------------------
    faq_match = re.match(
        r"^(How|What|Where|When|Why|Who|Can|Do|Does|Is|Are|Should|Will|May)\b.*?\?\s*(.+)$",
        clean,
        re.IGNORECASE | re.DOTALL,
    )

    if faq_match:
        return faq_match.group(2).strip()

    # --------------------------------------------------
    # Restore bullet formatting
    # --------------------------------------------------
    clean = re.sub(r"\s*[•]\s*", "\n• ", clean)

    clean = re.sub(r"\s+o\s+(?=[A-Z])", "\no ", clean)

    clean = clean.strip()

    return clean


# ==========================================
# RETRIEVE  (vector + keyword)
# ==========================================


def retrieve(query, k=8):
    if index is None or not chunks:
        return []

    q_emb = np.array([get_embedding(query)]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    seen = set()
    final = []
    # Only include vector results with a meaningful similarity score (>= 0.25)
    for score, i in zip(D[0], I[0]):
        if i < len(chunks):
            c = chunks[i].strip()
            if len(c) >= 40 and c not in seen and score >= 0.25:
                final.append(c)
                seen.add(c)

    qwords = keywords(query)
    if qwords:
        scored = [(kw_score(qwords, c), c) for c in chunks if kw_score(qwords, c) > 0]
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, c in scored[:4]:
            if c not in seen:
                final.append(c)
                seen.add(c)

    return final[:10]


# ==========================================
# NUMBERED Q&A EXTRACTION
# For  "1. Question?\nAns: Answer" format
# ==========================================


def extract_numbered_qa(query, results):
    qwords = keywords(query)
    if not qwords:
        return None

    best_answer = None
    best_score = 0

    for chunk in results:
        # Normalize: insert newlines before numbered items and Ans:
        normalized = re.sub(r"\s+(?=\d+[\.:]\s)", "\n", chunk)
        normalized = re.sub(r"\s+(?=Ans\s*:)", "\n", normalized)

        items = re.split(r"\n(?=\d+[\.:]\s)", normalized)
        for item in items:
            item = item.strip()
            ans_match = re.search(
                r"Ans\s*:\s*(.+?)(?=\n\d+[\.:]\s|\Z)", item, re.DOTALL | re.IGNORECASE
            )
            if not ans_match:
                continue
            q_part = item[: ans_match.start()].strip()
            q_text = re.sub(r"^\d+[\.:]\s*", "", q_part).strip()
            q_text = re.sub(r"\s+", " ", q_text)
            a_text = re.sub(r"\s+", " ", ans_match.group(1)).strip()

            q_words_in_q = keywords(q_text)
            score = len(set(qwords) & set(q_words_in_q))
            for i in range(len(qwords) - 1):
                if qwords[i] + " " + qwords[i + 1] in q_text.lower():
                    score += 3

            if score > best_score:
                best_score = score
                best_answer = a_text

    min_score = max(2, len(qwords) * 0.5)
    coverage = best_score / max(len(qwords), 1)
    if best_score >= min_score and coverage >= 0.4 and best_answer:
        return best_answer
    return None


# ==========================================
# Q&A EXTRACTION  (Q: / A: format)
# Handles both:
#   - Newline-separated:  Q: text\nA: text
#   - Flat/inline:        Q: text A: text Q: text A: text
# ==========================================


def extract_qa_answer(query, results):
    qwords = keywords(query)
    if not qwords:
        return None

    best_answer = None
    best_score = 0

    for chunk in results:
        # ── Normalize: treat both flat and newline formats ──
        # Insert newline before every Q: and A: so we can split uniformly
        normalized = re.sub(r"\s+(?=Q\s*:)", "\n", chunk)
        normalized = re.sub(r"\s+(?=A\s*:)", "\n", normalized)

        # Split into Q/A pair blocks at each Q:
        pairs = re.split(r"(?=\nQ\s*:)", "\n" + normalized)

        for pair in pairs:
            pair = pair.strip()
            if not re.search(r"Q\s*:", pair, re.IGNORECASE):
                continue

            # Extract question text
            q_match = re.match(
                r"(?:\[.*?\]\s*\n?)?\s*Q\s*:\s*(.+?)(?=\nA\s*:|\nAns\s*:|\Z)",
                pair,
                re.DOTALL | re.IGNORECASE,
            )
            if not q_match:
                continue
            q_text = q_match.group(1).strip().strip('"').strip("'")
            # Collapse internal newlines in q_text
            q_text = re.sub(r"\s+", " ", q_text).strip()

            # Extract answer text
            a_match = re.search(
                r"(?:A\s*:|Ans\s*:)\s*(.+?)(?=\nQ\s*:|\Z)",
                pair,
                re.DOTALL | re.IGNORECASE,
            )
            if not a_match:
                continue
            a_text = re.sub(r"\s+", " ", a_match.group(1)).strip()

            if not a_text or len(a_text) < 5:
                continue

            # Score: keyword overlap between user query and this Q text
            q_words_in_q = keywords(q_text)
            score = len(set(qwords) & set(q_words_in_q))

            # Bigram bonus
            for i in range(len(qwords) - 1):
                if qwords[i] + " " + qwords[i + 1] in q_text.lower():
                    score += 3

            # Overlap ratio bonus (rewards questions that are mostly about the query)
            if q_words_in_q:
                score += score / max(len(qwords), 1)

            if score > best_score:
                best_score = score
                best_answer = a_text

    # Require meaningful overlap: fraction of query words matched + absolute minimum
    min_score = max(2, len(qwords) * 0.5)
    coverage = best_score / max(len(qwords), 1)
    if best_score >= min_score and coverage >= 0.4 and best_answer:
        return best_answer
    return None


# ==========================================
# DIRECT Q&A SEARCH
# Scans ALL chunks for best matching Q/A pair.
# Does NOT rely on vector retrieval — runs on every query
# before falling back to vector search.
# ==========================================


def direct_qa_search(query):

    qwords = keywords(query)

    if not qwords:
        return None

    best_answer = None
    best_score = 0
    best_question = ""
    best_coverage = 0

    query_lower = query.lower().strip()

    for chunk in chunks:

        # Normalize Q/A blocks
        norm = re.sub(r'(?<!\n)\s+(?=Q\s*:)', '\n', chunk)
        norm = re.sub(r'(?<!\n)\s+(?=A\s*:)', '\n', norm)
        norm = re.sub(r'(?<!\n)\s+(?=Ans\s*:)', '\n', norm)

        blocks = re.split(r'(?=\nQ\s*:)', '\n' + norm)

        for block in blocks:

            block = block.strip()

            if not re.search(r'Q\s*:', block, re.IGNORECASE):
                continue

            # -------------------------
            # Question
            # -------------------------

            qm = re.search(
                r'Q\s*:\s*(.+?)(?=\nA\s*:|\nAns\s*:|\Z)',
                block,
                re.DOTALL | re.IGNORECASE
            )

            if not qm:
                continue

            q_text = re.sub(
                r'\s+',
                ' ',
                qm.group(1)
            ).strip()

            # -------------------------
            # Answer
            # -------------------------

            am = re.search(
                r'(?:A\s*:|Ans\s*:)\s*(.+?)(?=\nQ\s*:|\Z)',
                block,
                re.DOTALL | re.IGNORECASE
            )

            if not am:
                continue

            a_text = re.sub(
                r'\s+',
                ' ',
                am.group(1)
            ).strip()

            if len(a_text) < 5:
                continue

            question_lower = q_text.lower().strip()
            answer_lower = a_text.lower().strip()

            # ==================================
            # EXACT QUESTION MATCH
            # ==================================

            if query_lower == question_lower:

                print(
                    f"[EXACT QUESTION MATCH] "
                    f"{q_text[:80]}"
                )

                return a_text

            # ==================================
            # KEYWORDS
            # ==================================

            query_kw = set(qwords)
            q_kw = set(keywords(q_text))
            a_kw = set(keywords(a_text))

            question_hits = len(
                query_kw & q_kw
            )

            answer_hits = len(
                query_kw & a_kw
            )

            question_coverage = (
                question_hits /
                max(len(query_kw), 1)
            )

            answer_coverage = (
                answer_hits /
                max(len(query_kw), 1)
            )

            # ==================================
            # BASE SCORE
            # ==================================

            score = (
                question_hits * 20 +
                answer_hits * 8 +
                question_coverage * 100 +
                answer_coverage * 40
            )

            # ==================================
            # QUERY PHRASE IN QUESTION
            # ==================================

            if query_lower in question_lower:
                score += 200

            # ==================================
            # QUERY PHRASE IN ANSWER
            # ==================================

            if query_lower in answer_lower:
                score += 100

            # ==================================
            # BIGRAM BONUS
            # ==================================

            for i in range(len(qwords) - 1):

                bigram = (
                    qwords[i]
                    + " "
                    + qwords[i + 1]
                )

                if bigram in question_lower:
                    score += 10

                if bigram in answer_lower:
                    score += 5

            # ==================================
            # BEST MATCH
            # ==================================

            coverage = max(
                question_coverage,
                answer_coverage
            )

            if (
                score > best_score
                or (
                    score == best_score
                    and coverage > best_coverage
                )
            ):

                best_score = score
                best_answer = a_text
                best_question = q_text
                best_coverage = coverage

                print(
                    f"[QA] score={score:.2f} "
                    f"coverage={coverage:.2f} "
                    f"question='{q_text[:80]}'"
                )

    # ==================================
    # FINAL CHECK
    # ==================================

    if best_answer and best_coverage >= 0.30:

        print(
            f"[QA-DIRECT] score={best_score:.2f} "
            f"coverage={best_coverage:.2f}"
        )

        return best_answer

    return None


def llm_answer(query, context):
    prompt = f"""You are a strict PDF question-answering assistant.

Rules:
- Answer ONLY using the context below. Do NOT use outside knowledge.
- Give the complete answer without truncating.
- Return ONLY the direct answer — no preamble, no explanation.
- If the context has a table, numbered list, or bullet list relevant to the question, preserve that structure.
- If the answer cannot be found in the context, respond ONLY with: Not found in PDF.
- Do NOT guess or infer answers not clearly stated in the context.

Context:
{context[:7000]}

Question: {query}
Answer:"""
    try:
        res = session.post(
            OLLAMA_GEN_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 600,
                    "top_k": 20,
                    "top_p": 0.9,
                },
            },
            timeout=90,
        )
        answer = res.json().get("response", "").strip()
        answer = re.split(r"\nQ\s*:", answer)[0].strip()

        not_found = [
            "not found in pdf",
            "no data found",
            "not present in",
            "not mentioned in",
        ]
        if any(p in answer.lower() for p in not_found):
            return None
        return answer if answer else None
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None


def find_section_chunks(query):
    """
    If the query matches a section HEADER (not a specific entry),
    return all chunks under that header sorted by entry number.

    Only activates when NO discriminating entry-title word is found
    — that case is handled by find_exact_entry instead.
    """
    qwords = keywords(query)
    if not qwords:
        return []

    # Collect all headers and score them
    header_scores = {}
    for chunk in chunks:
        hm = re.match(r"^\[(.+?)\]", chunk)
        if not hm:
            continue
        h = hm.group(1).lower()
        score = sum(1 for w in qwords if w in h)
        if score > 0:
            header_scores[h] = max(header_scores.get(h, 0), score)

    if not header_scores:
        return []

    best_header = max(header_scores, key=lambda h: header_scores[h])

    # Collect + sort all chunks under that header
    matched = [
        chunk
        for chunk in chunks
        if re.match(r"^\[(.+?)\]", chunk)
        and re.match(r"^\[(.+?)\]", chunk).group(1).lower() == best_header
    ]

    def entry_num(c):
        m = re.search(r"(\d+)\.\s+[A-Z]", c)
        return int(m.group(1)) if m else 9999

    matched.sort(key=entry_num)
    return matched


def find_contact_block(query):
    """
    Handles queries about headquarters / support contacts that are stored
    as plain-text blocks (not Q/A pairs and not numbered directory entries).
    Scores chunks by header + body keyword overlap.
    Returns the single best chunk if keyword overlap is strong enough.
    """
    qwords = keywords(query)
    if not qwords:
        return None

    best_score = 0
    best_chunk = None

    for chunk in chunks:
        hm = re.match(r"^\[(.+?)\]", chunk)
        if not hm:
            continue
        header = hm.group(1).lower()
        body = chunk[hm.end() :].lower()

        # Must not be a numbered-entry chunk (those are handled by find_exact_entry)
        if re.search(r"\n\d+\.\s+[A-Z]", chunk):
            continue
        # Must not be a Q/A chunk
        if re.search(r"\bQ\s*:", chunk):
            continue

        header_hits = sum(1 for w in qwords if w in header)
        body_hits = sum(1 for w in qwords if w in body)
        score = header_hits * 3 + body_hits

        if score > best_score:
            best_score = score
            best_chunk = chunk

    # Require strong overlap: at least half the query words found
    min_needed = max(2, len(qwords) * 0.5)
    if best_score >= min_needed and best_chunk:
        return best_chunk
    return None


# ==========================================
# SAVE CHAT HISTORY
# ==========================================
def save_chat(question, answer, duration_ms):

    try:

        # Auto create folder
        os.makedirs("result", exist_ok=True)

        filename = "result/chat_history.txt"

        content = f"""
==================================================

Chatbot

🧑 You: {question}

🤖 AI:
{answer}

Time Duration: {duration_ms:.2f}ms

Saved Time:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""

        with open(filename, "a", encoding="utf-8") as f:
            f.write(content)

        print(f"[CHAT SAVED] {filename}")

    except Exception as e:

        print(f"[SAVE CHAT ERROR] {e}")


def ask(query):

    try:

        if index is None or not chunks:
            _load_db()

        if index is None or not chunks:
            return "⚠️ No PDFs ingested yet. Please run: python ingest.py"

        query = query.strip()

        if not query:
            return "Please ask a question."

        if query.lower() in {"hi", "hello", "hey"}:
            return "👋 Please ask a question about the PDF."

        print(f"\n[ASK] {query!r}")

        # ==========================================
        # START TIMER
        # ==========================================

        start_time = time.time()

        # ==========================================
        # STEP 0 : EXACT ENTRY MATCH
        # ==========================================
        
        best_chunk, field = find_exact_entry(query)
        
        if best_chunk:
        
            header, title, num = parse_chunk_parts(best_chunk)
        
            print(
                f"[ASK] Exact match → "
                f"header='{header[:35]}' "
                f"title='{title[:30]}' "
                f"field={field}"
            )
        
            answer = format_entry(best_chunk, field)
        
            duration_ms = (time.time() - start_time) * 1000
        
            save_chat(query, answer, duration_ms)
        
            return answer
        
        
        # ==========================================
        # STEP 1 : DIRECT QA SEARCH
        # ==========================================
        
        qa_answer = direct_qa_search(query)
        
        if qa_answer:
        
            duration_ms = (time.time() - start_time) * 1000
        
            save_chat(query, qa_answer, duration_ms)
        
            print(f"[ASK] Direct QA → {qa_answer[:80]}")
        
            return qa_answer

        # ==========================================
        # STEP 1b: SECTION MATCH
        # ==========================================

        section_chunks = find_section_chunks(query)

        if section_chunks:

            dir_chunks = [
                c for c in section_chunks if re.search(r"\n?\d+\.\s+[A-Z]", c)
            ]

            qa_chunks = [c for c in section_chunks if c not in dir_chunks]

            qw_set = set(keywords(query))

            directory_words = {
                "directory",
                "contact",
                "list",
                "all",
                "offices",
                "address",
                "phone",
                "email",
                "location",
                "incharge",
                "charge",
            }

            wants_directory = bool(qw_set & directory_words) or bool(dir_chunks)

            if not wants_directory and qa_chunks:
                target = qa_chunks

            elif dir_chunks:
                target = dir_chunks

            else:
                target = section_chunks

            lines = []

            for chunk in target:

                clean = re.sub(r"^\[.+?\]\s*", "", chunk).strip()

                clean = re.sub(r"\s*[•]\s*", "\n• ", clean)

                clean = re.sub(r"\s+o\s+(?=[A-Z])", "\no ", clean)

                lines.append(clean.strip())

            answer = "\n\n".join(lines)

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, answer, duration_ms)

            print(f"[ASK] Section match → {len(target)} chunks")

            return answer

        # ==========================================
        # STEP 1c: CONTACT BLOCK MATCH
        # ==========================================

        contact_chunk = find_contact_block(query)

        if contact_chunk:

            clean = re.sub(r"^\[.+?\]\s*", "", contact_chunk).strip()

            clean = re.sub(r"\s*[•]\s*", "\n• ", clean)

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, clean, duration_ms)

            print(f"[ASK] Contact block match → {clean[:60]}")

            return clean

        # ==========================================
        # STEP 2: VECTOR RETRIEVAL
        # ==========================================

        results = retrieve(query, k=8)

        print(f"[ASK] {len(results)} chunks retrieved")

        if not results:

            answer = "⚠️ No data found in PDF"

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, answer, duration_ms)

            return answer

        # ==========================================
        # STEP 3: NUMBERED QA EXTRACTION
        # ==========================================

        answer = extract_numbered_qa(query, results)

        if answer:

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, answer, duration_ms)

            print(f"[ASK] Numbered QA → {len(answer)} chars")

            return answer

        # ==========================================
        # STEP 4: QA EXTRACTION
        # ==========================================

        answer = extract_qa_answer(query, results)

        if answer:

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, answer, duration_ms)

            print(f"[ASK] QA extraction → {len(answer)} chars")

            return answer

        # ==========================================
        # STEP 5: LLM FALLBACK
        # ==========================================

        qwords_for_llm = keywords(query)

        top_kw_score = max((kw_score(qwords_for_llm, c) for c in results), default=0)

        if top_kw_score >= 2:

            print("[ASK] LLM fallback")

            context = "\n\n".join(results)

            answer = llm_answer(query, context)

            if answer:

                duration_ms = (time.time() - start_time) * 1000

                save_chat(query, answer, duration_ms)

                print(f"[ASK] LLM answered → " f"{len(answer)} chars")

                return answer

        # ==========================================
        # STEP 6: RETURN BEST CHUNK
        # ==========================================

        print("[ASK] Checking best chunk " "relevance before returning directly")

        qwords = keywords(query)

        if not results:

            answer = "⚠️ No relevant information found " "in the PDF for your question."

            duration_ms = (time.time() - start_time) * 1000

            save_chat(query, answer, duration_ms)

            return answer

        scored = [(kw_score(qwords, c), c) for c in results] if qwords else []

        if scored:

            best_kw_score, best = max(scored, key=lambda x: x[0])

            if best_kw_score >= 2:

                answer = format_entry(best)

                duration_ms = (time.time() - start_time) * 1000

                save_chat(query, answer, duration_ms)

                return answer

        # ==========================================
        # FINAL FALLBACK
        # ==========================================

        answer = "⚠️ No relevant information found " "in the PDF for your question."

        duration_ms = (time.time() - start_time) * 1000

        save_chat(query, answer, duration_ms)

        return answer

    except Exception as e:

        print(f"[ASK ERROR] {e}")

        return "⚠️ Error processing your question"


# ==========================================
# RELOAD DB
# ==========================================


def reload_db():
    global index, chunks

    try:
        print("\n[RAG] Reloading vector database...")

        if os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)

        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                raw = f.read()

            chunks = [c.strip() for c in raw.split("\n---CHUNK---\n") if c.strip()]

        print(f"[RAG] Reload complete → {len(chunks)} chunks")

    except Exception as e:
        print(f"[RAG RELOAD ERROR] {e}")

