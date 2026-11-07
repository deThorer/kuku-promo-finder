"""Judge stage: rank all finished promos for posting.

1. cold viewer (AI, no show context): reads each promo's transcript and says what it understood
2. judge (AI, one comparative call): scores every promo against the others (hook, cliffhanger, clarity, spoilers,
   duplicates) - comparing in one call fights the "everything is 5/5" score inflation seen in the moment bank
3. final score (code): weighted scores, minus penalties for spoilers, failed hard rules and duplicates;
   duplicates are also detected in code (two promos sharing most of the same footage)
"""
from __future__ import annotations

from .prompts import COLD_VIEWER_SYSTEM, JUDGE_SYSTEM, build_cold_viewer_prompt, build_judge_prompt

SPOILER_POINTS = {"low": 0.0, "medium": 1.0, "high": 3.0}


def viewer_lines(p: dict, max_camera_chars: int = 110) -> list[str]:
    """What a viewer sees and hears, in promo order."""
    out = []
    for r in p["transcript"]:
        t = r["promo_time"][:5]
        if r["kind"] == "DIALOGUE":
            who = r["who"].split(" (")[0]
            out.append(f'[{t}] {who}: "{r["text"]}"')
        elif r["kind"] == "SOUND":
            out.append(f"[{t}] SOUND: {r['text']}")
        else:
            out.append(f"[{t}] SCREEN: {r['text'][:max_camera_chars]}")
    return out


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def cold_viewer(promos: list[dict], llm, batch: int = 7) -> dict[str, dict]:
    """Returns {promo_id: reaction}. llm=None -> placeholder reactions (offline testing)."""
    out: dict[str, dict] = {}
    for i in range(0, len(promos), batch):
        group = promos[i:i + batch]
        keys = {f"C{j + 1:02d}": p for j, p in enumerate(group)}
        if llm is None:
            for p in group:
                out[p["promo_id"]] = {"who_is_who": "", "what_happens": "", "question_left": "", "confusing": "",
                                      "keep_watching": 3, "source": "baseline"}
            continue
        block = "\n\n".join(f"=== CLIP {k} ({p['total_seconds']:.0f}s) ===\n" + "\n".join(viewer_lines(p))
                            for k, p in keys.items())
        raw = llm.complete_json(COLD_VIEWER_SYSTEM, build_cold_viewer_prompt(block),
                                tag=f"cold_viewer_{i // batch + 1}")
        clips = (raw or {}).get("clips") or {}
        for k, p in keys.items():
            r = clips.get(k) or {}
            out[p["promo_id"]] = {
                "who_is_who": str(r.get("who_is_who", "")), "what_happens": str(r.get("what_happens", "")),
                "question_left": str(r.get("question_left", "")), "confusing": str(r.get("confusing", "")),
                "keep_watching": _clamp(r.get("keep_watching"), 1, 5, 3), "source": "ai" if r else "missing",
            }
    return out


def judge(promos: list[dict], cold: dict[str, dict], premise: str, llm) -> dict[str, dict]:
    keys = {f"P{j + 1:02d}": p for j, p in enumerate(promos)}
    if llm is None:
        return {p["promo_id"]: {"hook": 6, "cliffhanger": 6, "clarity": 5, "spoiler_risk": "low", "duplicates": [],
                                "verdict": "(baseline: no judge)", "source": "baseline"} for p in promos}
    blocks = []
    for k, p in keys.items():
        c = cold.get(p["promo_id"], {})
        blocks.append(
            f"=== {k} | {p.get('type', 'scene')} | {p['total_seconds']:.0f}s | {p['selection']} ===\n"
            + "\n".join(viewer_lines(p)) + "\n"
            + f"EDITOR'S HOOK: {p['hook'].get('description', '')}\n"
            + f"EDITOR'S CLIFFHANGER QUESTION: {p['cliffhanger'].get('open_question', '')}\n"
            + f"TEST VIEWER: who={c.get('who_is_who', '')} | what={c.get('what_happens', '')} | "
            + f"wants to know={c.get('question_left', '')} | confused by={c.get('confusing', '') or 'nothing'} | "
            + f"would keep watching={c.get('keep_watching', '?')}/5")
    raw = llm.complete_json(JUDGE_SYSTEM, build_judge_prompt(premise, "\n\n".join(blocks)), tag="judge")
    scores = (raw or {}).get("promos") or {}
    out = {}
    for k, p in keys.items():
        r = scores.get(k) or {}
        dups = [keys[d]["promo_id"] for d in (r.get("duplicates") or []) if isinstance(d, str) and d in keys and d != k]
        risk = str(r.get("spoiler_risk", "low")).lower()
        out[p["promo_id"]] = {
            "hook": _clamp(r.get("hook"), 1, 10, 5), "cliffhanger": _clamp(r.get("cliffhanger"), 1, 10, 5),
            "clarity": _clamp(r.get("clarity"), 1, 10, 5),
            "spoiler_risk": risk if risk in SPOILER_POINTS else "medium",
            "duplicates": dups, "verdict": str(r.get("verdict", "")), "source": "ai" if r else "missing",
        }
    return out


def footage_overlap(a: dict, b: dict) -> float:
    """Share of the shorter promo's footage that also appears in the other promo (0..1)."""
    from .parser import parse_ts
    def spans(p):
        return [(s["episode"], parse_ts(s["start"]), parse_ts(s["end"])) for s in p["segments"]]
    shared = sum(max(0.0, min(e1, e2) - max(s1, s2))
                 for ep1, s1, e1 in spans(a) for ep2, s2, e2 in spans(b) if ep1 == ep2)
    shorter = min(a["total_seconds"], b["total_seconds"]) or 1.0
    return shared / shorter


def _episodes(p: dict) -> set[int]:
    return {s["episode"] for s in p["segments"]}


def final_ranking(promos: list[dict], cold: dict, scores: dict, dup_overlap: float = 0.4,
                  final_act_from: int = 0, final_act_penalty: float = 1.0) -> list[dict]:
    rows = []
    for p in promos:
        s, c = scores[p["promo_id"]], cold.get(p["promo_id"], {})
        base = (0.35 * s["hook"] + 0.30 * s["cliffhanger"] + 0.25 * s["clarity"]
                + 0.10 * 2 * c.get("keep_watching", 3))
        flags = []
        pen = SPOILER_POINTS[s["spoiler_risk"]]
        if pen:
            flags.append(f"spoiler {s['spoiler_risk']}")
        if final_act_from and min(_episodes(p)) >= final_act_from:
            pen += final_act_penalty  # same policy as the ranking stage: don't give away the ending
            flags.append(f"final act (ep {final_act_from}+)")
        if p["status"] != "ok":
            pen += 5.0
            flags.append("FAILED hard rules")
        if any("code cut" in f for f in p["checker"]["fixes"]):
            flags.append("trimmed by code")
        rows.append({"promo": p, "base": round(base, 2), "score": round(base - pen, 2), "flags": flags})
    rows.sort(key=lambda r: -r["score"])
    failed = [r for r in rows if r["promo"]["status"] != "ok"]
    rows = [r for r in rows if r["promo"]["status"] == "ok"]
    accepted, later = [], []
    for r in rows:
        pid = r["promo"]["promo_id"]
        # duplicate = shares footage (code), or the judge says so AND they share an episode
        # (the judge alone is not trusted: in testing it flagged two unrelated scenes as duplicates)
        def dup(a: dict) -> bool:
            aid = a["promo"]["promo_id"]
            if footage_overlap(r["promo"], a["promo"]) >= dup_overlap:
                return True
            judged = pid in scores[aid]["duplicates"] or aid in scores[pid]["duplicates"]
            return judged and bool(_episodes(r["promo"]) & _episodes(a["promo"]))
        clash = next((a for a in accepted if dup(a)), None)
        if clash:
            r["flags"].append(f"duplicate of {clash['promo']['promo_id']}")
            r["score"] = round(r["score"] - 1.5, 2)
            later.append(r)
        else:
            accepted.append(r)
    # distinct promos first, then duplicates of higher-ranked ones, then promos that broke the hard rules
    ranked = accepted + sorted(later, key=lambda r: -r["score"]) + failed
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def final_ranking_markdown(ranked: list[dict], cold: dict, scores: dict, model: str, top_n: int = 5) -> str:
    out = ["# Final promo ranking", "",
           f"Judged by: {model}. Score = 0.35 hook + 0.30 cliffhanger + 0.25 clarity + 0.10 test viewer's "
           "keep-watching (x2); minus 1 (medium) or 3 (high) for spoilers, 1 for final-act promos, "
           "5 for failed hard rules, 1.5 for duplicating a higher-ranked promo.", "",
           "| Rank | Promo | Type | Sec | Hook | Cliff | Clarity | Keep watching | Spoiler | Score | Flags | Verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked:
        p, s, c = r["promo"], scores[r["promo"]["promo_id"]], cold.get(r["promo"]["promo_id"], {})
        out.append(f"| {r['rank']} | {p['promo_id'].split('__')[0]} | {p.get('type', '')} | {p['total_seconds']:.0f} "
                   f"| {s['hook']} | {s['cliffhanger']} | {s['clarity']} | {c.get('keep_watching', '')}/5 "
                   f"| {s['spoiler_risk']} | {r['score']} | {'; '.join(r['flags'])} | {s['verdict'].replace('|', '/')} |")
    out += ["", f"## Post these first (top {top_n})", ""]
    for r in ranked[:top_n]:
        p, s, c = r["promo"], scores[r["promo"]["promo_id"]], cold.get(r["promo"]["promo_id"], {})
        cut = " + ".join(f"Ep {g['episode']} {g['start']}-{g['end']}" for g in p["segments"])
        out += [f"### {r['rank']}. {p['promo_id'].split('__')[0]}: {p['title']} ({p['total_seconds']:.0f}s)",
                f"- **Cut:** {cut}",
                f"- **Hook:** {p['hook'].get('line') or ''}",
                f"- **Cliffhanger:** {p['cliffhanger'].get('open_question', '')}",
                f"- **Why post it:** {s['verdict']}",
                f"- **Test viewer (never saw the show):** {c.get('what_happens', '')} Wants to know: {c.get('question_left', '')}",
                ""]
    weak = [r for r in ranked if scores[r["promo"]["promo_id"]]["clarity"] <= 4 or cold.get(r["promo"]["promo_id"], {}).get("confusing")]
    out += ["## Quality check: where the tool picked badly", ""]
    for r in ranked:
        p, s, c = r["promo"], scores[r["promo"]["promo_id"]], cold.get(r["promo"]["promo_id"], {})
        notes = []
        if s["clarity"] <= 4:
            notes.append(f"unclear to a new viewer (clarity {s['clarity']})")
        if c.get("confusing"):
            notes.append(f"test viewer confused by: {c['confusing']}")
        if s["spoiler_risk"] != "low":
            notes.append(f"{s['spoiler_risk']} spoiler risk")
        if s["hook"] <= 4:
            notes.append(f"weak hook ({s['hook']})")
        if s["cliffhanger"] <= 4:
            notes.append(f"weak cliffhanger ({s['cliffhanger']})")
        notes += [f for f in r["flags"] if f.startswith(("duplicate", "FAILED", "trimmed", "final act"))]
        if notes:
            out.append(f"- **{p['promo_id'].split('__')[0]}**: " + "; ".join(notes))
    if not weak:
        out.append("")
    return "\n".join(out) + "\n"
