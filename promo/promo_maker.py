"""Promo maker: selected episode(s) in -> one checked promo out.

Flow (one promo):
  1. gather the events of the selected episodes (code)
  2. AI picks segments, hook and cliffhanger by EVENT ID (AI)
  3. checker converts IDs to timestamps, applies safe fixes, finds rule breaks (code)
  4. if rules are broken, the checker's report goes back to the AI to repair (AI, up to N tries)
  5. output: cut list (episode + start + end per segment) + hook + cliffhanger + why + transcript
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .baseline import baseline_montage, baseline_promo
from .checker import CheckResult, Rules, check_promo, tighten
from .parser import Event, Show, fmt_ts, parse_ts
from .prompts import (MONTAGE_SYSTEM, PROMO_SYSTEM, build_montage_prompt, build_promo_prompt,
                      build_repair_prompt)


@dataclass
class Selection:
    episode: int
    start: float | None = None  # optional time range inside the episode
    end: float | None = None

    def label(self) -> str:
        if self.start is None and self.end is None:
            return f"E{self.episode:02d}"
        return f"E{self.episode:02d}@{fmt_ts(self.start or 0)}-{fmt_ts(self.end) if self.end is not None else 'end'}"


def parse_selection(tokens: list[str]) -> list[Selection]:
    """'3' | '3-5' (episodes 3,4,5) | '12@00:40-02:10' (episode 12, that time range only)."""
    out: list[Selection] = []
    for tok in tokens:
        for part in re.split(r"[,\s]+", tok.strip()):
            if not part:
                continue
            if "@" in part:
                ep, rng = part.split("@", 1)
                a, _, b = rng.partition("-")
                out.append(Selection(int(ep), parse_ts(a) if a else None, parse_ts(b) if b else None))
            elif "-" in part:
                a, b = part.split("-", 1)
                out += [Selection(n) for n in range(int(a), int(b) + 1)]
            else:
                out.append(Selection(int(part)))
    return sorted(out, key=lambda s: s.episode)


def selected_events(show: Show, selection: list[Selection]) -> list[Event]:
    events: list[Event] = []
    for sel in selection:
        if sel.episode not in show.episodes:
            raise ValueError(f"Episode {sel.episode} not found (have {min(show.episodes)}-{max(show.episodes)})")
        for e in show.episodes[sel.episode].events:
            if sel.start is not None and e.end <= sel.start:
                continue
            if sel.end is not None and e.start >= sel.end:
                continue
            events.append(e)
    return events


def story_so_far(show: Show, first_ep: int, recent: int = 3) -> str:
    """Episode 1's logline plus the few episodes right before the selection (code, no AI)."""
    eps = sorted(n for n in show.episodes if n < first_ep)
    if not eps:
        return ""
    keep = sorted(set(eps[:1] + eps[-recent:]))
    lines = []
    for n in keep:
        if lines and n != keep[keep.index(n) - 1] + 1:
            lines.append("...")
        lines.append(f"Ep {n}: {show.episodes[n].logline}")
    return "\n".join(lines)


def montage_prompt_parts(show: Show, plan: dict, index: dict, include_camera: bool) -> tuple[str, str]:
    """Plan text + events grouped per planned moment (in playing order)."""
    plan_lines = [f"Theme: {plan.get('theme', '')} - {plan.get('title', '')}",
                  f"Through-line: {plan.get('through_line', '')}", "Moments in playing order:"]
    blocks = []
    n = len(plan["moments"])
    for j, m in enumerate(plan["moments"], 1):
        role = "HOOK (cold open allowed)" if j == 1 else ("CLIFFHANGER" if j == n else "middle")
        plan_lines.append(f"{j}. {m['id']} [{role}] Episode {m['episode']} {m['start_ts']}-{m['end_ts']}: {m['summary']}"
                          + (f" | open question: {m['open_question']}" if m.get("open_question") else ""))
        evs = sorted((e for e in index.values() if e.episode == m["episode"]
                      and e.end > m["start"] - 1.5 and e.start < m["end"] + 1.5
                      and (include_camera or e.kind != "CAMERA")), key=lambda e: (e.start, e.index))
        blocks.append(f"--- Moment {j}: {m['id']} (episode {m['episode']}) ---\n" + "\n".join(e.line() for e in evs))
    return "\n".join(plan_lines), "\n\n".join(blocks)


def make_promo(show: Show, selection: list[Selection], llm=None, rules: Rules | None = None,
               max_attempts: int = 3, include_camera: bool = True, mode: str = "scene",
               plan: dict | None = None, bible_text: str = "", promo_id: str | None = None) -> dict:
    """llm=None -> rule-based baseline (no AI). mode='montage' needs `plan` (moments in playing order)."""
    rules = rules or Rules()
    events = selected_events(show, selection)
    index = {e.id: e for e in events}
    all_by_ep = {n: ep.events for n, ep in show.episodes.items()}
    sel_label = promo_id or "+".join(s.label() for s in selection)

    attempts: list[dict] = []
    if llm is None and mode == "montage":
        promo = baseline_montage(plan["moments"], index)
        result = check_promo(promo, index, all_by_ep, rules)
        attempts.append({"answer": promo, "fixes": result.fixes, "issues": result.issues})
        provider = "baseline"
    elif llm is None:
        promo = baseline_promo(events)
        result = check_promo(promo, index, all_by_ep, rules)
        attempts.append({"answer": promo, "fixes": result.fixes, "issues": result.issues})
        provider = "baseline"
    else:
        provider = f"{llm.provider}:{llm.model}"
        prompt_events = [e for e in events if include_camera or e.kind != "CAMERA"]
        system = PROMO_SYSTEM
        notes = "\n".join(f"Episode {s.episode} ({fmt_ts(show.episodes[s.episode].duration)} long"
                          f"{', using ' + s.label() if s.start is not None or s.end is not None else ''}): "
                          f"{show.episodes[s.episode].logline}" for s in selection)
        prompt = build_promo_prompt(show.title, story_so_far(show, selection[0].episode), notes,
                                    "\n".join(e.line() for e in prompt_events))
        if mode == "montage":
            system = MONTAGE_SYSTEM
            plan_block, events_block = montage_prompt_parts(show, plan, index, include_camera)
            prompt = build_montage_prompt(show.title, bible_text, plan_block, events_block)
        user = prompt
        promo, result = {}, CheckResult()
        for attempt in range(1, max_attempts + 1):
            promo = llm.complete_json(system, user, tag=f"promo_{sel_label}_try{attempt}")
            result = check_promo(promo, index, all_by_ep, rules)
            attempts.append({"answer": promo, "fixes": result.fixes, "issues": result.issues})
            if result.ok:
                break
            if attempt == max_attempts:
                print(f"  [checker] try {attempt}: {len(result.issues)} problem(s) left; giving up (status=failed_checks)")
                break
            print(f"  [checker] try {attempt}: {len(result.issues)} problem(s) -> asking the model to repair")
            user = build_repair_prompt(prompt, promo, result.report())

    promo, result = _last_resort(promo, result, index, all_by_ep, rules, attempts)
    if llm is not None:
        provider = f"{llm.provider}:{llm.model}"  # the model may have switched on a quota fallback
    out = assemble(show, selection, sel_label, provider, promo, result, attempts)
    out["type"] = mode
    if plan:
        out["plan"] = {k: v for k, v in plan.items() if k != "moments"} | {"moments": [m["id"] for m in plan["moments"]]}
    return out


def _last_resort(promo: dict, result: CheckResult, index: dict, all_by_ep: dict, rules: Rules,
                 attempts: list[dict]) -> tuple[dict, CheckResult]:
    """If the only problem left is a small length overrun, let code cut the middle (see checker.tighten)."""
    if not result.ok and result.issues and all("over" in i and "Total length" in i for i in result.issues):
        cut = tighten(promo, index, all_by_ep, rules)
        if cut is not None:
            note = cut.pop("_code_cut")
            promo = cut
            result = check_promo(promo, index, all_by_ep, rules)
            result.fixes.append(note)
            attempts.append({"answer": promo, "fixes": result.fixes, "issues": result.issues, "by": "code"})
            print(f"  [checker] {note}")
    return promo, result


def recheck_promo(show: Show, old: dict, selection: list[Selection], rules: Rules) -> dict:
    """Re-run today's checker on a saved promo's last answer (0 AI calls)."""
    events = selected_events(show, selection)
    index = {e.id: e for e in events}
    all_by_ep = {n: ep.events for n, ep in show.episodes.items()}
    attempts = list(old.get("attempt_history", []))
    promo = attempts[-1]["answer"] if attempts else {}
    result = check_promo(promo, index, all_by_ep, rules)
    attempts.append({"answer": promo, "fixes": result.fixes, "issues": result.issues, "by": "recheck"})
    promo, result = _last_resort(promo, result, index, all_by_ep, rules, attempts)
    out = assemble(show, selection, old["selection"], old["provider"], promo, result, attempts)
    out["promo_id"] = old["promo_id"]
    for k in ("type", "plan", "window"):
        if k in old:
            out[k] = old[k]
    return out


def transcript(show: Show, result: CheckResult) -> list[dict]:
    """What the viewer sees and hears, in promo order, with promo-relative times."""
    rows, offset = [], 0.0
    for i, seg in enumerate(result.segments, 1):
        for e in sorted(show.episodes[seg.episode].events, key=lambda e: (e.start, e.index)):
            if seg.overlaps(e):
                rows.append({
                    "segment": i,
                    "promo_time": fmt_ts(offset + max(0.0, e.start - seg.start)),
                    "event": e.id,
                    "kind": e.kind,
                    "who": e.label(),
                    "text": e.text,
                })
        offset += seg.duration
    return rows


def assemble(show: Show, selection: list[Selection], sel_label: str, provider: str, promo: dict,
             result: CheckResult, attempts: list[dict]) -> dict:
    index = show.event_index()
    hook = promo.get("hook") or {}
    cliff = promo.get("cliffhanger") or {}
    hook_ev = index.get(str(hook.get("event", "")).upper())
    cliff_ev = index.get(str(cliff.get("event", "")).upper())
    return {
        "promo_id": f"{sel_label}__{provider.split(':')[0]}",
        "status": "ok" if result.ok else "failed_checks",
        "selection": sel_label,
        "provider": provider,
        "attempts": len(attempts),
        "title": promo.get("title", ""),
        "total_seconds": round(result.total, 2),
        "segments": [
            {"order": i, "episode": s.episode, "start": fmt_ts(s.start), "end": fmt_ts(s.end),
             "seconds": round(s.duration, 2), "first_event": s.first_event, "last_event": s.last_event,
             "purpose": s.purpose}
            for i, s in enumerate(result.segments, 1)
        ],
        "hook": {**hook, "line": hook_ev.line() if hook_ev else None,
                 "starts_at_promo_second": None if result.hook_offset is None else round(result.hook_offset, 2)},
        "cliffhanger": {**cliff, "line": cliff_ev.line() if cliff_ev else None,
                        "seconds_after_it": None if result.cliff_tail is None else round(result.cliff_tail, 2)},
        "context_for_new_viewer": promo.get("context_for_new_viewer", ""),
        "why_it_works": promo.get("why_it_works", ""),
        "model_confidence": promo.get("confidence"),
        "checker": {"fixes": result.fixes, "issues": result.issues},
        "attempt_history": attempts,
        "transcript": transcript(show, result),
    }


def to_markdown(p: dict, include_camera: bool = True) -> str:
    out = [f"# Promo {p['promo_id']}: {p['title']}", "",
           f"**Status:** {p['status']}  |  **Length:** {p['total_seconds']}s  |  **Made by:** {p['provider']}  "
           f"|  **Attempts:** {p['attempts']}  |  **Model confidence:** {p['model_confidence']}", "",
           "## Cut list", "", "| # | Episode | Start | End | Seconds | Purpose |", "|---|---|---|---|---|---|"]
    out += [f"| {s['order']} | {s['episode']} | {s['start']} | {s['end']} | {s['seconds']} | {s['purpose']} |"
            for s in p["segments"]]
    h, c = p["hook"], p["cliffhanger"]
    out += ["", f"**Hook** ({h.get('type', '')}, starts at promo second {h.get('starts_at_promo_second')}): "
                f"{h.get('description', '')}",
            f"> {h.get('line')}", "",
            f"**Cliffhanger** ({c.get('seconds_after_it')}s of footage after it): {c.get('open_question', '')}",
            f"> {c.get('line')}", "",
            f"**For a new viewer:** {p['context_for_new_viewer']}", "",
            f"**Why it works:** {p['why_it_works']}", ""]
    if p["checker"]["fixes"] or p["checker"]["issues"]:
        out += ["## Checker notes", ""]
        out += [f"- fixed: {x}" for x in p["checker"]["fixes"]]
        out += [f"- PROBLEM: {x}" for x in p["checker"]["issues"]]
        out.append("")
    out += ["## What the viewer sees and hears", "", "| Promo time | Seg | Event | Who / what | Text |",
            "|---|---|---|---|---|"]
    for r in p["transcript"]:
        if r["kind"] == "CAMERA" and not include_camera:
            continue
        text = r["text"].replace("|", "/")
        text = f"*{text}*" if r["kind"] == "CAMERA" else text
        out.append(f"| {r['promo_time']} | {r['segment']} | {r['event']} | {r['who']} | {text} |")
    return "\n".join(out) + "\n"
