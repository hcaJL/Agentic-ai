"""User feedback style profile — the「角色記錄檔」.

While testing, the user thumbs-up / thumbs-down individual booth lines. A
thumbs-up appends that line (with the moment's context) to a JSONL profile;
retrieval then feeds the best-matching approved lines back into persona_prompt
as few-shot, so the booth drifts toward the voice the user actually likes.

Thumbs-down = "don't collect / remove it" (the user's chosen semantics): it
deletes any matching approved line and never stores a negative example.

JSONL is the source of truth — human-editable, git-diffable, offline-first (same
contract as the rest of memory/). One line per approved example:
  {"verdict","persona","speaker","text","register","tags","san","ts"}

The shape matches personas.PERSONAS[*]["examples"], so an approved line is a
drop-in growth of the hand-written 風格卡 — and the natural corpus for the
stage-2 style RAG (load this JSONL into a ChromaDB collection unchanged).
"""
import json
import os
import random
import threading
import time

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_feedback.jsonl")
_lock = threading.Lock()


def _register_bucket(intensity) -> str:
    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        return "mid"
    return "high" if intensity >= 0.8 else "mid" if intensity >= 0.4 else "low"


def _load() -> list[dict]:
    if not os.path.exists(_PATH):
        return []
    out = []
    with open(_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue                      # skip a corrupt line, keep the rest
    return out


def _write_all(rows: list[dict]) -> None:
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, _PATH)                     # atomic swap


def _key(persona, speaker, text):
    return (persona, speaker, (text or "").strip())


def record(verdict: str, persona: str, speaker: str, text: str,
           intensity=0.2, tags=None, san=None):
    """good -> add the line (dedup on persona+speaker+text);
    bad  -> remove any matching approved line, store nothing.
    Returns the new profile size, or None if the input was unusable."""
    text = (text or "").strip()
    if not text or not persona or not speaker:
        return None
    tags = [str(t) for t in (tags or []) if str(t)]
    with _lock:
        rows = _load()
        # drop any existing line for this exact (persona, speaker, text) — this
        # both dedups a repeated good and implements bad = "remove it".
        rows = [r for r in rows
                if _key(r.get("persona"), r.get("speaker"), r.get("text"))
                != _key(persona, speaker, text)]
        if verdict == "good":
            rows.append({
                "verdict": "good", "persona": persona, "speaker": speaker,
                "text": text, "register": _register_bucket(intensity),
                "tags": tags, "san": san,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        _write_all(rows)
        return len(rows)


def good_examples(persona: str, intensity=0.2, event_tags=None, k: int = 3) -> list[dict]:
    """Top-k approved lines for this persona, scored by tag overlap with the
    moment + a nudge for matching register; ties broken randomly so the same
    line isn't parroted every call. Offline-safe: [] when profile empty/missing."""
    rows = [r for r in _load()
            if r.get("verdict") == "good" and r.get("persona") == persona]
    if not rows:
        return []
    bucket = _register_bucket(intensity)
    tags = set(t.lower() for t in (event_tags or []))
    scored = []
    for r in rows:
        rtags = set(t.lower() for t in r.get("tags", []))
        score = len(tags & rtags) * 2
        if r.get("register") == bucket:
            score += 1
        scored.append((score, random.random(), r))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [{"speaker": r["speaker"], "text": r["text"],
             "register": r.get("register"), "tags": r.get("tags", [])}
            for _, _, r in scored[:k]]
