"""rag.py - retrieval-augmented explanations for Cardio Care AI.

The trained CAD model in app.py (risk / cad_model_predict) still produces the score. RAG never changes it: it retrieves passages
from ./knowledge (guidelines, model card, data dictionary) that match the patient's drivers and has a generator explain them, with
citations. No new dependency is needed (TF-IDF from scikit-learn). Claude is used only if ANTHROPIC_API_KEY is set and the
`anthropic` package is installed; otherwise the answer is the retrieved sources themselves (extractive mode)."""
import os, re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path(__file__).parent / "knowledge"
SUFFIXES = {".md", ".txt", ".pdf"}
SYSTEM = ("You explain the output of a coronary artery disease (CAD) screening model to a patient or caregiver. Use ONLY the numbered "
          "sources and the patient facts given. Cite sources like [1]. Never change, recompute or question the score. Do not diagnose, "
          "do not name drugs or doses, and say plainly when the sources do not cover something. Always say a doctor must confirm. "
          "Plain language, under 250 words.")


# ---------- ingestion ----------
def _read(path):
    if path.suffix.lower() == ".pdf":
        try:  # pypdfium2 is much faster than pdfplumber on long guidelines
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(path))
            pages = []
            for i in range(len(doc)):
                tp = doc[i].get_textpage()
                pages.append(tp.get_text_range())
                tp.close()
            doc.close()
            text = "\n".join(pages)
            if len(text.strip()) > 200:
                return text
        except Exception:
            pass
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text, size=900, overlap=150):
    """Paragraph-based chunks of about `size` characters; very long paragraphs are split at sentence ends."""
    parts = []
    for p in (x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()):
        parts += [p] if len(p) <= size else re.findall(r".{1,%d}(?:[.!?]\s|$)" % size, p, flags=re.S)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = (cur[-overlap:].split(" ", 1)[-1] + " " + p).strip()
        else:
            cur = (cur + "\n\n" + p).strip()
    return out + [cur] if cur else out


def build_index(kb_dir=KB_DIR):
    """Read every .md/.txt/.pdf under kb_dir (README.md is skipped) and index it. Wrap in st.cache_resource in the app."""
    docs = []
    for p in sorted(Path(kb_dir).rglob("*")):
        if p.suffix.lower() in SUFFIXES and p.name.lower() != "readme.md":
            try:
                docs += [{"source": p.name, "text": c} for c in chunk(_read(p))]
            except Exception:
                continue  # unreadable file: skip, do not crash the app
    if not docs:
        return {"docs": [], "vec": None, "X": None}
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return {"docs": docs, "vec": vec, "X": vec.fit_transform([d["text"] for d in docs])}


# ---------- bridge to the model ----------
def patient_context(d, score, tier, contrib, scenario, lab, ref, flaglab):
    """Turn the app's own outputs into a small fact sheet. d = derive(values()); (score, tier, contrib) = risk(d);
    scenario = detect_scenario(values()); lab/ref/flaglab = LAB / REF / FLAGLAB from app.py. The patient's name is left out on purpose."""
    abnormal = []
    for k, (label, unit) in lab.items():
        v = d.get(k)
        lo, hi = ref.get(k, (None, None))
        if v is not None and ((lo is not None and v < lo) or (hi is not None and v > hi)):
            abnormal.append((label, f"{label} {v:g} {unit}"))
    return {"score": score, "tier": tier, "drivers": dict(contrib), "scenario": scenario["name"] if scenario else None,
            "abnormal": abnormal, "history": [l for k, l in flaglab.items() if d.get(k)], "age": d.get("age"), "sex": d.get("sex"),
            "missing": [lab[k][0] for k in lab if d.get(k) is None]}


def build_query(ctx, question=""):
    q = (question or "").strip()  # the question is repeated so it outweighs the patient-context terms
    return " ".join([q, q, q, ctx["tier"], ctx.get("scenario") or "", *ctx["drivers"], *ctx["history"], *(a[0] for a in ctx["abnormal"])])


def _facts(ctx):
    drv = ", ".join(f"{k} ({p} pts)" for k, p in sorted(ctx["drivers"].items(), key=lambda x: -x[1])) or "none stand out"
    return (f"Model score: {ctx['score']}/100 ({ctx['tier']}). Age {ctx.get('age') or 'unknown'}, sex {ctx.get('sex') or 'unknown'}.\n"
            f"Main drivers: {drv}.\nOutside normal range: {'; '.join(a[1] for a in ctx['abnormal']) or 'none recorded'}.\n"
            f"History/findings: {', '.join(ctx['history']) or 'none recorded'}.\nNot available: {len(ctx['missing'])} parameters.")


# ---------- retrieval + generation ----------
def retrieve(index, query, k=4, min_sim=0.05):
    if not index["docs"]:
        return []
    sims = (index["X"] @ index["vec"].transform([query]).T).toarray().ravel()  # TF-IDF rows are L2-normalised: dot = cosine
    return [{**index["docs"][i], "sim": float(sims[i])} for i in np.argsort(-sims)[:k] if sims[i] >= min_sim]


def _extractive(ctx, passages):
    if not passages:
        return "No passage in the knowledge base matches this patient. Add guideline documents to the knowledge folder."
    body = "\n\n".join(f"[{i}] {p['text'][:450].strip()}{'…' if len(p['text']) > 450 else ''}" for i, p in enumerate(passages, 1))
    return f"Score {ctx['score']}/100 ({ctx['tier']}). Most relevant knowledge-base passages:\n\n{body}"


def answer(index, ctx, question="", k=4, model=None):
    """{'text', 'mode' ('claude' | 'extractive'), 'sources', 'note'}. The score in ctx is passed through untouched."""
    passages = retrieve(index, build_query(ctx, question), k)
    note = ""
    if passages and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            src = "\n\n".join(f"[{i}] ({p['source']}) {p['text']}" for i, p in enumerate(passages, 1))
            msg = f"PATIENT FACTS\n{_facts(ctx)}\n\nSOURCES\n{src}\n\nQUESTION\n{question or 'Explain this score and what the patient can do next.'}"
            r = anthropic.Anthropic().messages.create(model=model or os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=700,
                                                      system=SYSTEM, messages=[{"role": "user", "content": msg}])
            return {"text": r.content[0].text, "mode": "claude", "sources": passages, "note": note}
        except Exception as e:
            note = f"Language model unavailable ({type(e).__name__}); showing retrieved passages only."
    return {"text": _extractive(ctx, passages), "mode": "extractive", "sources": passages, "note": note}
