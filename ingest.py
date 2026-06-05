import os
import re
import faiss
import requests
import numpy as np
import pdfplumber

# Optional: Vision RAG page rendering
try:
    import fitz  # PyMuPDF — pip install pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("[INGEST] PyMuPDF not installed. Vision RAG disabled. Run: pip install pymupdf")

# ==========================================
# PATHS
# ==========================================

DATA_DIR = "data"
INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"
IMAGE_DIR = "vector_db/images"      # Page images for Vision RAG
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
session = requests.Session()

# ==========================================
# EXTRACT TEXT
# ==========================================

def extract_full_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    clean_row = [str(c).strip() for c in row if c and str(c).strip()]
                    if clean_row:
                        all_text += " | ".join(clean_row) + "\n"
            t = page.extract_text()
            if t:
                all_text += t + "\n"

    # Normalize inline Q:/A: pairs to separate lines
    all_text = re.sub(r'\s+(?=Q\s*:)', '\n', all_text)
    all_text = re.sub(r'\s+(?=A\s*:)', '\n', all_text)
    all_text = re.sub(r'\s+(?=Ans\s*:)', '\n', all_text)

    return all_text


# ==========================================
# VISION RAG: EXTRACT PAGE IMAGES
# ==========================================

def extract_page_images(pdf_path, dpi=150):
    """
    Render each PDF page as a JPEG image and save to IMAGE_DIR.
    Filename encodes the source PDF + page number for scoring.
    DPI=150 is a good balance between quality and file size.

    Requires PyMuPDF (pip install pymupdf).
    """
    if not PYMUPDF_AVAILABLE:
        return

    os.makedirs(IMAGE_DIR, exist_ok=True)

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    # Sanitize filename: replace spaces and special chars
    pdf_stem = re.sub(r"[^\w\-]", "_", pdf_stem).lower()

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 dpi = 1x scale

    saved = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(IMAGE_DIR, f"{pdf_stem}_page_{page_num + 1:04d}.jpg")
        pix.save(out_path)
        saved += 1

    doc.close()
    print(f"[VISION] Saved {saved} page images → {IMAGE_DIR}/")


# ==========================================
# SECTION SPLITTER
# ==========================================

SECTION_HEADER_RE = re.compile(r'^[A-Z][A-Z\s:&/()\-]{8,}$', re.MULTILINE)

FAQ_INLINE_RE = re.compile(
    r'(?:^|\n)(FAQ\s*(?:\([^)]*\))?)\s*\n',
    re.IGNORECASE
)


def split_into_sections(text):
    positions = [(m.start(), m.group().strip()) for m in SECTION_HEADER_RE.finditer(text)]
    if not positions:
        return [("CONTENT", text)]

    raw_sections = []
    for i, (pos, header) in enumerate(positions):
        body_start = pos + len(header)
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            raw_sections.append((header, body))

    sections = []
    for header, body in raw_sections:
        faq_m = FAQ_INLINE_RE.search(body)
        if faq_m:
            before = body[:faq_m.start()].strip()
            faq_body = body[faq_m.end():].strip()
            if before:
                sections.append((header, before))
            if faq_body:
                sections.append(("FAQ", faq_body))
        else:
            sections.append((header, body))

    return sections


# ==========================================
# CHUNKERS
# ==========================================

def fix_malformed_qa(body):
    lines = body.split("\n")
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*Q\s*:", line, re.IGNORECASE):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r"^\s*Q\s*:", lines[j], re.IGNORECASE):
                fixed.append(line)
                fixed.append("A: " + re.sub(r"^\s*Q\s*:\s*", "", lines[j]))
                i = j + 1
                continue
        fixed.append(line)
        i += 1
    return "\n".join(fixed)


def chunk_qa(header, body):
    body = fix_malformed_qa(body)
    body = re.sub(r'\s+(?=Q\s*:)', '\n', body)
    body = re.sub(r'\s+(?=A\s*:)', '\n', body)
    body = re.sub(r'\s+(?=Ans\s*:)', '\n', body)

    parts = re.split(r'(?=\nQ\s*:)', '\n' + body)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) >= 40 and re.search(r'Q\s*:', part, re.IGNORECASE) and \
           re.search(r'(?:A\s*:|Ans\s*:)', part, re.IGNORECASE):
            chunks.append(f"[{header}]\n{part}")
    return chunks


def chunk_numbered_qa(header, body):
    body = re.sub(r'\s+(?=Ans\s*:)', '\n', body)
    body = re.sub(r'\s+(?=A\s*:)', '\n', body)

    parts = re.split(r"(?=\n\d+[\.:]\ *\S)", "\n" + body)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < 20:
            continue
        q_m = re.match(r'^\d+[\.:]\ *(.*?)(?=\n(?:Ans|A)\s*:|\Z)', part, re.DOTALL | re.IGNORECASE)
        a_m = re.search(r'(?:Ans|A)\s*:\s*(.+)', part, re.DOTALL | re.IGNORECASE)
        if q_m and a_m:
            q_text = re.sub(r'\s+', ' ', q_m.group(1)).strip()
            if not q_text.endswith('?'):
                q_text += '?'
            a_text = re.sub(r'\s+', ' ', a_m.group(1)).strip()
            if len(q_text) >= 5 and len(a_text) >= 5:
                chunks.append(f"[{header}]\nQ: {q_text}\nA: {a_text}")
        elif len(part) >= 40:
            chunks.append(f"[{header}]\n{part}")
    return chunks


def chunk_numbered_list(header, body):
    oneline = re.sub(r'\s+', ' ', body).strip()
    positions = [m.start() for m in re.finditer(r'(?=\d+\.\s+[A-Z][a-z])', oneline)]

    if not positions:
        return [f"[{header}]\n{body}"] if len(body) >= 40 else []

    chunks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(oneline)
        entry = oneline[start:end].strip()
        if len(entry) >= 40:
            chunks.append(f"[{header}]\n{entry}")
    return chunks


def chunk_bullet_entries(header, body):
    oneline = re.sub(r'\s+', ' ', body).strip()
    entry_starts = [m.start() for m in re.finditer(r'(?=\d+\.\s+[A-Z][A-Za-z])', oneline)]

    if not entry_starts:
        lines = [l.strip() for l in body.split("\n") if l.strip()]
        chunks = []
        block = [f"[{header}]"]
        for line in lines:
            block.append(line)
            if len(block) >= 9:
                chunks.append("\n".join(block))
                block = [f"[{header}]"]
        if len(block) > 2:
            chunks.append("\n".join(block))
        return chunks

    chunks = []
    for i, start in enumerate(entry_starts):
        end = entry_starts[i + 1] if i + 1 < len(entry_starts) else len(oneline)
        entry = oneline[start:end].strip()
        if len(entry) >= 40:
            chunks.append(f"[{header}]\n{entry}")
    return chunks


def chunk_section(header, body):
    qa_count   = len(re.findall(r"^\s*Q\s*:", body, re.MULTILINE | re.IGNORECASE))
    ans_count  = len(re.findall(r"^\s*(?:A|Ans)\s*:", body, re.MULTILINE | re.IGNORECASE))
    num_entry  = len(re.findall(r"^\d+\.\s+[A-Z][a-z]", body, re.MULTILINE))
    bullet_cnt = len(re.findall(r"^[•]\s", body, re.MULTILINE))
    o_bullet   = len(re.findall(r"^o\s+", body, re.MULTILINE))

    numbered_faq = (
        ans_count >= 2 and num_entry >= 2
        and len(re.findall(r'^\d+[\.:]\s*\w', body, re.MULTILINE)) >= 2
    )
    if numbered_faq and qa_count == 0:
        return chunk_numbered_qa(header, body)

    if (qa_count >= 2 or ans_count >= 2) and (num_entry >= 3 or bullet_cnt >= 3):
        qa_split = re.search(r'\n(?=\s*(?:\d+[\.:]\s*)?Q\s*:)', body, re.IGNORECASE)
        if not qa_split:
            qa_split = re.search(
                r'\n(?=\s*\d+[\.:]\s*(?:What|How|Is|Are|Can|Does|Who|Why|When|Where)\b)',
                body, re.IGNORECASE
            )
        if qa_split:
            dir_body = body[:qa_split.start()].strip()
            qa_body  = body[qa_split.start():].strip()
            result = []
            if dir_body:
                result.extend(chunk_section(header, dir_body))
            if qa_body:
                result.extend(chunk_section(header, qa_body))
            return result

    if qa_count > 0:
        return chunk_qa(header, body)
    if ans_count >= 2:
        return chunk_numbered_qa(header, body)
    if num_entry >= 3 and (bullet_cnt >= 3 or o_bullet >= 3):
        return chunk_bullet_entries(header, body)
    if num_entry >= 3:
        return chunk_numbered_list(header, body)
    if bullet_cnt >= 5:
        return chunk_bullet_entries(header, body)
    if len(body) >= 40:
        return [f"[{header}]\n{body}"]
    return []


# ==========================================
# EMBEDDING
# ==========================================

def get_embedding(text):
    try:
        res = session.post(
            OLLAMA_EMBED_URL,
            json={"model": MODEL, "prompt": text},
            timeout=30
        )
        return res.json().get("embedding")
    except Exception as e:
        print("Embedding error:", e)
        return None


# ==========================================
# PROCESS PDF
# ==========================================

def process_single_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path}")
    os.makedirs("vector_db", exist_ok=True)

    # ── Text extraction and chunking ──
    print("Extracting text...")
    raw_text = extract_full_text(pdf_path)
    print(f"Total chars: {len(raw_text)}")

    print("\nSplitting into sections...")
    sections = split_into_sections(raw_text)
    print(f"Sections: {len(sections)}")

    all_chunks = []
    for header, body in sections:
        sc = chunk_section(header, body)
        all_chunks.extend(sc)
        if sc:
            print(f"  [{header[:55]}] → {len(sc)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("ERROR: No chunks generated")
        return

    print("\n[SAMPLE CHUNKS — verify each is ONE entry]")
    for c in all_chunks[:6]:
        print(f"  >>> {c[:180]}")
        print()

    # ── Embed and save ──
    embeddings, valid_chunks = [], []
    for chunk in all_chunks:
        emb = get_embedding(chunk)
        if emb:
            embeddings.append(emb)
            valid_chunks.append(chunk)

    print(f"Embedded: {len(valid_chunks)}/{len(all_chunks)}")

    if not embeddings:
        print("ERROR: No embeddings — is Ollama running?")
        return

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n---CHUNK---\n".join(valid_chunks))

    print(f"\nDone! {INDEX_FILE}, {CHUNKS_FILE}")

    # ── Vision RAG: extract page images ──
    print("\nExtracting page images for Vision RAG...")
    extract_page_images(pdf_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs in data/ folder")
    else:
        for pdf in pdfs:
            process_single_pdf(os.path.join(DATA_DIR, pdf))

