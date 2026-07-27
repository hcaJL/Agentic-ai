"""Import a broadcaster transcript into the style profile (角色記錄檔).

Real chess-commentary corpus is scarce in Chinese and the Taiwanese chess
channels don't publish subtitles, so the practical route is: yt-dlp pulls the
audio, faster-whisper transcribes it, and this script cleans the transcript
into short style snippets and appends them to personas/style_feedback.jsonl —
the SAME file the 👍/👎 feedback loop writes to, so persona_prompt picks these
lines up as few-shot automatically (see personas/feedback_store.py).

Each imported line is stored verdict="good" (an approved style example) with a
"source" tag for traceability, so you can delete a batch later by source.

Usage:
    python tools/import_style_corpus.py scratch/sample_transcript.txt \
        --persona calm --source "reychess:0tjsVo94YqE" --dry-run
Drop --dry-run to actually write.
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = os.path.join(ROOT, "personas", "style_feedback.jsonl")

# a line that leans explanatory → analyst; otherwise the play-by-play voice
_ANALYST_HINTS = ("因為", "所以", "威脅", "計劃", "計畫", "如果", "接下來",
                  "應該", "最佳", "優勢", "劣勢", "評估", "目的", "策略",
                  "打算", "為了", "弱點", "控制", "其實", "重點")
# coarse event tags from keywords, so retrieval can match by situation
_TAG_RULES = {
    "capture": ("吃", "換子", "拿下"),
    "check": ("將軍", "將軍！", "將死"),
    "sacrifice": ("棄", "犧牲", "送"),
    "castle": ("易位",),
    "opening": ("開局", "開場", "布局", "佈局"),
    "endgame": ("殘局", "終局"),
    "promotion": ("升變", "升后", "升變為"),
    "fork": ("叉", "雙擊", "雙重"),
    "pin": ("牽制", "釘"),
}
# pure filler / noise / channel promo — a broadcaster's sponsor & sign-off
# patter is NOT commentary voice, so it gets dropped
_NOISE = re.compile(
    r"^[\s。，、！？…~—\-]*$|訂閱|按讚|小鈴鐺|下一部影片|謝謝觀看|下期再見|"
    r"千萬別錯過|迫不期待|購買|網課|上分|積極的影片|深入學習|接下來.*影片|"
    r"後裔|乾貨|相信看完|再也不是問題|對大家的")   # + whisper-garbled 後裔 & promo patter
# a square coordinate (F3, c4, g5…) — snippets containing one are dropped by
# default: whisper mangles chess notation ("象g5"→"像G5"), and the move CONTENT
# is worthless to us anyway (the booth generates its own moves). The gold is the
# pure-voice phrasing, so we keep those and throw the notation away.
_COORD = re.compile(r"[a-hA-H][1-8]")
# unambiguous whole-word transcription fixes (safe — no homophone collateral)
# 后翼弃兵这个开局名观众听不懂 → 一律讲成通俗的「這個開局」
_SAFE_FIX = {"黑幫": "黑方", "白幫": "白方",
             "後裔棋兵": "這個開局", "後裔棋": "這個開局", "後翼棋兵": "這個開局",
             "後翼棄兵": "這個開局", "后翼弃兵": "這個開局", "后翼棄兵": "這個開局",
             "白個像": "白格象", "白個象": "白格象", "黑個像": "黑格象"}
_SPLIT = re.compile(r"[，,。.！!？?、；;：:\s]+")
# LoL/esports-specific vocab that would be nonsense in chess commentary — used
# only with --drop-game-terms when importing hype streamer corpus, so we keep
# the喊麥 delivery ("欸不是""這波太扯") and throw away game-content fragments.
# Deliberately NOT listing chess/game shared words (中路/兵線/優勢) — those read
# fine in chess even if they came from a game clip.
_GAME_TERMS = re.compile(
    r"閃現|大招|水晶|防禦塔|防御塔|打野|野區|野区|團戰|团战|開團|开团|買活|买活|"
    r"大龍|大龙|小龍|小龙|復活|复活|越塔|推塔|補兵|补兵|泉水|裝備|裝甲|技能|"
    r"英雄|召喚師|召唤师|峽谷|峡谷|路人|對線|对线|gank|buff|carry|combo|solo|"
    r"上路|下路|中路兵|一血|三殺|四殺|五殺|團滅|团灭|遠古|巴龍")


def _apply_fixes(s: str) -> str:
    for bad, good in _SAFE_FIX.items():
        s = s.replace(bad, good)
    return s


def _to_traditional(text: str) -> str:
    try:
        import opencc
        return opencc.OpenCC("s2twp").convert(text)
    except Exception:
        return text                      # opencc not installed → leave as-is


def _speaker(line: str) -> str:
    return "analyst" if any(h in line for h in _ANALYST_HINTS) else "play_by_play"


def _tags(line: str) -> list[str]:
    return [tag for tag, kws in _TAG_RULES.items() if any(k in line for k in kws)]


def clean(lines, min_len, max_len, keep_notation=False, drop_game=False):
    """Split each transcript line into short sub-sentences and keep the clean
    voice snippets: drop noise, drop notation-bearing fragments (unless
    keep_notation), drop game-vocab fragments (drop_game, for esports corpus),
    apply safe fixes, length-filter, dedup."""
    seen, out = set(), []
    for raw in lines:
        raw = _to_traditional(raw.strip())
        for frag in _SPLIT.split(raw):
            s = _apply_fixes(frag.strip())
            if not s or _NOISE.search(s):
                continue
            if not keep_notation and _COORD.search(s):
                continue
            if drop_game and _GAME_TERMS.search(s):
                continue
            if not (min_len <= len(s) <= max_len):
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def existing_texts() -> set:
    if not os.path.exists(PROFILE):
        return set()
    out = set()
    with open(PROFILE, encoding="utf-8") as f:
        for ln in f:
            try:
                out.add(json.loads(ln).get("text", ""))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="one sentence per line")
    ap.add_argument("--persona", default="realcast",
                    choices=["realcast", "personas2", "excited", "calm", "literary"])
    ap.add_argument("--register", default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--source", default="import", help="traceability tag, e.g. reychess:<id>")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--keep-notation", action="store_true",
                    help="keep fragments with square coordinates (default: drop them)")
    ap.add_argument("--drop-game-terms", action="store_true",
                    help="drop fragments with LoL/esports vocab (for hype streamer corpus)")
    ap.add_argument("--limit", type=int, default=0, help="cap how many lines to import (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="preview only, write nothing")
    a = ap.parse_args()

    with open(a.transcript, encoding="utf-8") as f:
        lines = clean(f.readlines(), a.min_len, a.max_len, a.keep_notation, a.drop_game_terms)
    have = existing_texts()
    rows = []
    for s in lines:
        if s in have:
            continue
        rows.append({
            "verdict": "good", "persona": a.persona, "speaker": _speaker(s),
            "text": s, "register": a.register, "tags": _tags(s), "san": None,
            "source": a.source, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    if a.limit:
        rows = rows[:a.limit]

    n_pbp = sum(r["speaker"] == "play_by_play" for r in rows)
    print(f"cleaned {len(lines)} lines → {len(rows)} new "
          f"(play_by_play {n_pbp} / analyst {len(rows)-n_pbp}), persona={a.persona}")
    for r in rows[:12]:
        print(f"  [{r['speaker'][:3]}] {r['tags']} {r['text']}")
    if len(rows) > 12:
        print(f"  … +{len(rows)-12} more")

    if a.dry_run:
        print("(dry-run: nothing written)")
        return
    with open(PROFILE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(rows)} lines → {PROFILE}")


if __name__ == "__main__":
    sys.exit(main())
