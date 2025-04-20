#!/usr/bin/env python3
"""
prepare_finetune_data.py
────────────────────────
Generate 1–5 Q‑A pairs per .txt Wikipedia page, with strong anti‑hallucination
guards.  Checkpoints every N pages.

Usage:
    python prepare_finetune_data.py                   # full directory
    python prepare_finetune_data.py --file kidney.txt # single file
"""

import asyncio, json, re, glob, argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter

import aiofiles, tiktoken
from tqdm import tqdm
from openai import AsyncOpenAI

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR             = Path("../data/raw/documents/Science/")
OUTPUT_JSON          = Path("../data/finetune/Science/science-qa.json")

OPENAI_MODEL         = "gpt-4o-mini"
TEMPERATURE          = 0.15
TOP_P                = 0.3
MAX_TOKENS_LLM       = 600

CONCURRENCY          = 5
TARGET_TOKENS        = 2500
OVERLAP_SENTENCES    = 2
MAX_QA_TOTAL         = 5
MIN_QA_TOTAL         = 1
CHECKPOINT_INTERVAL  = 10
OVERLAP_THRESHOLD    = 0.80     # 80 % word overlap

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a data‑generation assistant. Produce JSON exactly like:

{ "qa": [ { "question": "...", "answer": "..." }, ... ] }

Rules:
• 1 to 5 items – no other keys, no prose.
• Read the ARTICLE, pick a fact, then write a question *and* its answer.
• If the fact is absent, do NOT invent it; simply generate fewer pairs.
• If a requested fact truly is not in the article, answer with NOT IN ARTICLE.
• Answers must be ≤75 words.
"""

USER_TMPL = 'ARTICLE:\n"""{}"""\n\nGenerate the Q‑A pairs now.'

# ─── Helpers ────────────────────────────────────────────────────────────────
enc       = tiktoken.encoding_for_model(OPENAI_MODEL)
hdr_re    = re.compile(r'^(#{1,6}\s.*|==[^=].*==)\s*$', re.M)
sent_re   = re.compile(r'(?<=[.!?])\s+')
token_re  = re.compile(r"\w+")
client    = AsyncOpenAI()

def split_article(text: str) -> List[str]:
    parts = re.split(hdr_re, text)
    sections = [''.join(p).strip() for p in zip(parts[1::2], parts[2::2])] \
               if len(parts) > 1 else [text]
    chunks, cur, cur_tok = [], [], 0
    for sec in sections:
        tok = len(enc.encode(sec))
        if cur_tok + tok > TARGET_TOKENS and cur:
            chunks.append('\n\n'.join(cur))
            tail = sent_re.split(cur[-1])[-OVERLAP_SENTENCES:]
            cur, cur_tok = [' '.join(tail), sec], len(enc.encode(sec))
        else:
            cur.append(sec); cur_tok += tok
    if cur:
        chunks.append('\n\n'.join(cur))
    return chunks

def word_overlap(a: str, b: str) -> float:
    set_a = set(token_re.findall(a.lower()))
    set_b = set(token_re.findall(b.lower()))
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)

async def call_llm(chunk: str) -> List[Dict[str, str]]:
    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TMPL.format(chunk)}
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS_LLM,
            timeout=40,
        )
        content = resp.choices[0].message.content
        data    = json.loads(content)

        pairs = data if isinstance(data, list) else data.get("qa", [])
        clean = []
        for qa in pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer",   "").strip()
            if not q or not a or a.upper() == "NOT IN ARTICLE":
                continue
            clean.append({"question": q, "answer": a})
        return clean[:MAX_QA_TOTAL]

    except Exception as e:
        print("API error:", e)
        return []

async def process_page(path: Path, sem: asyncio.Semaphore) -> List[Dict[str, str]]:
    if not path.exists():
        print("Missing file:", path)
        return []

    async with aiofiles.open(path, "r", encoding="utf‑8") as f:
        raw = await f.read()
    if not raw.strip():
        return []

    chunks = split_article(raw)
    print(f"Processing {path.name} – {len(chunks)} chunk(s)")
    pairs: List[Dict[str, str]] = []

    async def worker(ch):
        async with sem:
            pairs.extend(await call_llm(ch))

    await asyncio.gather(*[worker(c) for c in chunks])

    # Keep only answers with high overlap to minimise hallucinations
    raw_low = raw.lower()
    final   = []
    for qa in pairs:
        if word_overlap(qa["answer"], raw_low) >= OVERLAP_THRESHOLD:
            final.append(qa)
        if len(final) >= MAX_QA_TOTAL:
            break
    return final if len(final) >= MIN_QA_TOTAL else []

def save_checkpoint(data: List[Dict[str, str]], dest: Path):
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(dest)
    print(f"↳ checkpoint: {len(data):,} pairs saved → {dest}")

# ─── Main ────────────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="single .txt file")
    ap.add_argument("--dir",  help="directory of .txt files")
    ap.add_argument("--out",  help="output JSON path")
    args = ap.parse_args()

    paths = [DATA_DIR / args.file] if args.file else \
            [Path(p) for p in glob.glob(str((Path(args.dir) if args.dir else DATA_DIR) / "*.txt"))]

    out_json = Path(args.out) if args.out else OUTPUT_JSON
    sem      = asyncio.Semaphore(CONCURRENCY)
    all_pairs= []
    for idx, p in enumerate(tqdm(paths, desc="Pages processed"), start=1):
        all_pairs.extend(await process_page(p, sem))
        if idx % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_pairs, out_json)
    save_checkpoint(all_pairs, out_json)
    print(f"\n✅ finished: {len(all_pairs):,} pairs total → {out_json}")

if __name__ == "__main__":
    asyncio.run(main())
