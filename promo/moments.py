"""Moment bank: the 5 strongest promo moments of every episode, with full details.

AI finds the moments (by event ID); code validates them, snaps them to whole lines,
attaches real timestamps, scores them and saves the bank after every episode (resumable).
"""
from __future__ import annotations

import json
from pathlib import Path

from .baseline import intensity
from .bible import THEME_KEYWORDS, bible_block
from .checker import Segment, snap_to_lines
from .parser import Event, Show, fmt_ts
from .promo_maker import story_so_far
from .prompts import MOMENTS_BATCH_NOTE, MOMENTS_SYSTEM, build_moments_batch_prompt, build_moments_prompt

SCORE_KEYS = ("hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_intensity")
SPOILER_PENALTY = {"none": 0.0, "low": 0.3, "high": 1.0}
WEIGHTS = {"hook_strength": 0.35, "cliffhanger_strength": 0.20, "standalone_clarity": 0.25, "emotional_intensity": 0.20}


def moment_score(m: dict) -> float:
    """Promo value of one moment, about 1-5. Weights are a judgement call; tune them here."""
    base = sum(WEIGHTS[k] * m[k] for k in SCORE_KEYS)
    return round(base - SPOILER_PENALTY.get(m.get("spoiler_level", "none"), 0.0), 3)


def _clamp(v, lo=1, hi=5) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return lo


def extract_moments(show: Show, ep_no: int, bible: dict, llm=None, k: int = 5, include_camera: bool = True) -> dict:
    ep = show.episodes[ep_no]
    if llm is None:
        raw = baseline_moments(show, ep_no, k)
    else:
        events = [e for e in ep.events if include_camera or e.kind != "CAMERA"]
        prompt = build_moments_prompt(bible_block(bible), story_so_far(show, ep_no), ep_no, ep.logline,
                                      "\n".join(e.line() for e in events))
        raw = llm.complete_json(MOMENTS_SYSTEM, prompt, tag=f"moments_E{ep_no:02d}")
    moments, notes = validate_moments(raw, show, ep_no, bible, k)
    return {"episode": ep_no, "status": "ok" if moments else "empty", "moments": moments, "notes": notes}


def extract_moments_batch(show: Show, ep_nos: list[int], bible: dict, llm, k: int = 5,
                          include_camera: bool = True, min_moments: int = 3) -> list[dict]:
    """Several episodes in ONE call (saves requests on free tiers with a low daily request limit).
    Episodes that come back with fewer than `min_moments` moments get ONE follow-up call together.
    Episodes missing from the answer get status 'missing' and are not saved, so a rerun redoes them."""

    def ask(nums: list[int], extra: str = "") -> dict:
        blocks = []
        for n in nums:
            ep = show.episodes[n]
            events = [e for e in ep.events if include_camera or e.kind != "CAMERA"]
            blocks.append(f"=== EPISODE {n}: {ep.logline}\nEVENTS:\n" + "\n".join(e.line() for e in events))
        prompt = build_moments_batch_prompt(bible_block(bible), story_so_far(show, nums[0]), "\n\n".join(blocks)) + extra
        raw = llm.complete_json(MOMENTS_SYSTEM + MOMENTS_BATCH_NOTE, prompt,
                                tag=f"moments_E{nums[0]:02d}-E{nums[-1]:02d}{'_retry' if extra else ''}")
        answer = (raw or {}).get("episodes") or {}
        if isinstance(answer, list):  # tolerate [{"episode": 10, "moments": [...]}, ...]
            answer = {str(x.get("episode")): x for x in answer if isinstance(x, dict)}
        out = {}
        for n in nums:
            part = answer.get(str(n)) or answer.get(f"Episode {n}") or answer.get(f"E{n:02d}")
            if part:
                moments, notes = validate_moments(part, show, n, bible, k)
                out[n] = {"episode": n, "status": "ok" if moments else "empty", "moments": moments, "notes": notes}
        return out

    got = ask(ep_nos)
    thin = [n for n in ep_nos if len(got.get(n, {}).get("moments", [])) < min_moments]
    if thin:
        counts = ", ".join(f"episode {n}: {len(got.get(n, {}).get('moments', []))}" for n in thin)
        print(f"  [moments] too few moments ({counts}) -> one follow-up call for episodes {thin}")
        extra = (f"\n\nIMPORTANT: a previous answer returned too few moments ({counts}). "
                 f"Return 5 moments for EACH of these episodes.")
        again = ask(thin, extra)
        for n, rec in again.items():
            if len(rec["moments"]) > len(got.get(n, {}).get("moments", [])):
                rec["notes"].append("from follow-up call (first answer had too few moments)")
                got[n] = rec
    return [got.get(n) or {"episode": n, "status": "missing", "moments": [], "notes": ["episode missing from batch answer"]}
            for n in ep_nos]


def validate_moments(raw: dict, show: Show, ep_no: int, bible: dict, k: int = 5) -> tuple[list[dict], list[str]]:
    ep = show.episodes[ep_no]
    index = {e.id: e for e in ep.events}
    theme_ids = {t["id"] for t in bible.get("themes", [])}
    out: list[dict] = []
    notes: list[str] = []
    for i, m in enumerate((raw or {}).get("moments") or [], 1):
        if not isinstance(m, dict):
            continue
        a = index.get(str(m.get("first_event", "")).strip().upper())
        b = index.get(str(m.get("last_event", "")).strip().upper())
        if a is None or b is None:
            notes.append(f"moment {i}: dropped, event IDs not in episode {ep_no} ({m.get('first_event')}, {m.get('last_event')})")
            continue
        if a.start > b.start:
            a, b = b, a
            notes.append(f"moment {i}: first/last events were reversed; swapped")
        seg = Segment(ep_no, a.start, max(a.end, b.end), a.id, b.id)
        snap_to_lines(seg, ep.events, [], "")
        peak = index.get(str(m.get("peak_event", "")).strip().upper())
        if peak is None or not seg.overlaps(peak):
            peak = max((e for e in ep.events if seg.overlaps(e) and e.kind != "CAMERA"), key=intensity, default=a)
            notes.append(f"moment {i}: peak_event missing or outside the moment; used {peak.id}")
        themes = [t for t in (m.get("themes") or []) if t in theme_ids]
        unknown = [t for t in (m.get("themes") or []) if t not in theme_ids]
        if unknown:
            notes.append(f"moment {i}: unknown theme ids dropped: {unknown}")
        spoiler = str(m.get("spoiler_level", "none")).lower()
        mom = {
            "episode": ep_no,
            "first_event": a.id, "last_event": b.id, "peak_event": peak.id,
            "start": round(seg.start, 2), "end": round(seg.end, 2),
            "start_ts": fmt_ts(seg.start), "end_ts": fmt_ts(seg.end), "seconds": round(seg.duration, 2),
            "type": str(m.get("type", "other")),
            "themes": themes,
            "characters": [str(c) for c in (m.get("characters") or [])],
            "summary": str(m.get("summary", "")),
            "open_question": str(m.get("open_question", "")),
            **{key: _clamp(m.get(key)) for key in SCORE_KEYS},
            "spoiler_level": spoiler if spoiler in SPOILER_PENALTY else "none",
            "why": str(m.get("why", "")),
            "peak_line": peak.line(),
        }
        mom["score"] = moment_score(mom)
        # drop if it mostly overlaps a moment we already kept
        clash = next((o for o in out if min(o["end"], mom["end"]) - max(o["start"], mom["start"])
                      > 0.5 * min(o["seconds"], mom["seconds"])), None)
        if clash:
            notes.append(f"moment {i}: overlaps another moment; dropped")
            continue
        if mom["seconds"] > 40:
            notes.append(f"moment {i}: long ({mom['seconds']}s); kept, the promo maker will trim it")
        out.append(mom)
    out.sort(key=lambda m: -m["score"])
    out = out[:k]
    for rank, m in enumerate(out, 1):
        m["id"] = f"M{ep_no:02d}.{rank}"
    return out, notes


# ------------------------------------------------------------------------------------- rule-based fallback
def baseline_moments(show: Show, ep_no: int, k: int = 5) -> dict:
    """No AI: the k most intense lines (keywords + emotion tags), each with ~6s around it."""
    ep = show.episodes[ep_no]
    peaks: list[Event] = []
    for e in sorted((e for e in ep.events if e.kind != "CAMERA"), key=lambda e: -intensity(e)):
        if all(abs(e.start - p.start) > 15 for p in peaks):
            peaks.append(e)
        if len(peaks) == k:
            break
    moments = []
    for p in peaks:
        inside = [e for e in ep.events if e.start >= p.start - 6 and e.end <= p.end + 6] or [p]
        text = " ".join(e.text.lower() for e in inside)
        themes = [t for t, kws in THEME_KEYWORDS.items() if any(kw in text for kw in kws)][:3]
        near_end = p.end >= ep.duration - 15
        level = 1 + min(4, intensity(p) // 3)
        moments.append({
            "first_event": min(inside, key=lambda e: e.start).id, "last_event": max(inside, key=lambda e: e.end).id,
            "peak_event": p.id, "type": "other", "themes": themes, "characters": [p.speaker] if p.speaker else [],
            "summary": f"{p.label()}: {p.text}", "open_question": "",
            "hook_strength": level, "cliffhanger_strength": 4 if near_end else 2,
            "standalone_clarity": 3, "emotional_intensity": level, "spoiler_level": "none",
            "why": "baseline: keyword/emotion intensity",
        })
    return {"moments": moments}


# ------------------------------------------------------------------------------------------- bank I/O
def load_bank(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"episodes": {}}


def save_bank(bank: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")


def all_moments(bank: dict) -> list[dict]:
    return [m for ep in bank.get("episodes", {}).values() for m in ep.get("moments", [])]


def bank_markdown(bank: dict) -> str:
    out = ["# Moment bank", "", f"Provider: {bank.get('provider', '')}", ""]
    for key in sorted(bank.get("episodes", {}), key=int):
        ep = bank["episodes"][key]
        out += [f"## Episode {key}", "", "| ID | Time | Sec | Type | Themes | Hook | Cliff | Clear | Int | Spoiler | Score | Summary |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in ep.get("moments", []):
            out.append(f"| {m['id']} | {m['start_ts']}-{m['end_ts']} | {m['seconds']} | {m['type']} | {', '.join(m['themes'])} "
                       f"| {m['hook_strength']} | {m['cliffhanger_strength']} | {m['standalone_clarity']} "
                       f"| {m['emotional_intensity']} | {m['spoiler_level']} | {m['score']} | {m['summary'].replace('|', '/')} |")
        for n in ep.get("notes", []):
            out.append(f"\n- note: {n}")
        out.append("")
    return "\n".join(out)
