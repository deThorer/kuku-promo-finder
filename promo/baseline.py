"""Rule-based baseline (no AI).

Two uses:
  1. run the whole pipeline offline, with no API key, to test it;
  2. a comparison point for the quality section: is the AI better than simple keyword rules?

Logic: hook = the most intense moment (keywords + emotion tags) early in the selection;
cliffhanger = the last line of the final selected episode (episodes are written to end on one).
Returns the same JSON shape as the AI, so it goes through the same checker.
"""
from __future__ import annotations

from .parser import Event

KEYWORDS = {
    "slap": 6, "punch": 5, "strike": 4, "struck": 4, "shove": 4, "grab": 3, "kneel": 3, "blood": 4,
    "scream": 4, "gasp": 3, "shock": 3, "reveal": 4, "kiss": 3, "sword": 4, "dagger": 4, "pistol": 4,
    "gun": 4, "faint": 3, "collapse": 3, "throw": 3, "smash": 4, "shatter": 4, "crash": 3, "fury": 3,
    "betray": 4, "humiliat": 4, "pregnan": 5, "divorce": 4, "duke": 3, "dead": 3, "kill": 4,
    "poison": 4, "arrest": 4, "liar": 3, "how dare": 5, "never": 1, "!": 1,
}
EMOTIONS = {
    "furious": 3, "enraged": 3, "shocked": 3, "horrified": 3, "cruel": 3, "contempt": 2, "venom": 3,
    "menac": 3, "explosive": 3, "scream": 3, "desperate": 2, "mocking": 2, "cold": 1, "triumphant": 2,
    "threat": 3, "seething": 3, "disgust": 2, "outraged": 3,
}


def intensity(e: Event) -> int:
    text = e.text.lower()
    score = sum(w for k, w in KEYWORDS.items() if k in text)
    if e.emotion:
        emo = e.emotion.lower()
        score += sum(w for k, w in EMOTIONS.items() if k in emo)
    if e.kind == "CAMERA":
        score = score // 2  # shot descriptions repeat the action; count them less
    return score


def baseline_promo(selected: list[Event]) -> dict:
    events = sorted(selected, key=lambda e: (e.episode, e.start))
    last_ep = events[-1].episode
    last_ep_events = [e for e in events if e.episode == last_ep]

    # Cliffhanger: last line of dialogue in the final episode (fallback: last action).
    lines = [e for e in last_ep_events if e.kind == "DIALOGUE"] or [e for e in last_ep_events if e.kind == "ACTION"]
    cliff = lines[-1] if lines else last_ep_events[-1]

    # Hook: most intense dialogue/action/sound at least 20s before the cliffhanger (or anywhere, if too short).
    def before_cliff(e: Event) -> bool:
        return (e.episode, e.start) < (cliff.episode, cliff.start - 20)

    pool = [e for e in events if e.kind != "CAMERA" and before_cliff(e)] or \
           [e for e in events if e.kind != "CAMERA" and (e.episode, e.start) < (cliff.episode, cliff.start)]
    hook = max(pool, key=lambda e: (intensity(e), -e.episode, -e.start)) if pool else events[0]

    same_ep_span = cliff.end - hook.start if hook.episode == cliff.episode else None
    if same_ep_span is not None and 20 <= same_ep_span <= 45:
        segments = [{"first_event": hook.id, "last_event": cliff.id, "purpose": "hook through to cliffhanger"}]
    else:
        # Segment 1: ~20s from the hook. Segment 2: the ~20s leading into the cliffhanger.
        hook_ep = [e for e in events if e.episode == hook.episode]
        seg1_last = max((e for e in hook_ep if e.start >= hook.start and e.end <= hook.start + 20),
                        key=lambda e: e.end, default=hook)
        seg2_first = min((e for e in last_ep_events if e.start >= cliff.end - 20 and e.start < cliff.start),
                         key=lambda e: e.start, default=cliff)
        if (seg2_first.episode, seg2_first.start) <= (seg1_last.episode, seg1_last.end):
            segments = [{"first_event": hook.id, "last_event": cliff.id, "purpose": "hook through to cliffhanger"}]
        else:
            segments = [
                {"first_event": hook.id, "last_event": seg1_last.id, "purpose": "hook"},
                {"first_event": seg2_first.id, "last_event": cliff.id, "purpose": "build to cliffhanger"},
            ]

    return {
        "title": f"Baseline: {hook.text[:40]}",
        "segments": segments,
        "hook": {"event": hook.id, "type": "other", "description": f"{hook.label()}: {hook.text}"},
        "cliffhanger": {"event": cliff.id, "open_question": f"(baseline cannot infer) last line: {cliff.text}"},
        "context_for_new_viewer": "(baseline does not judge this)",
        "why_it_works": "Rule-based baseline: highest keyword/emotion score as hook, final line of the last episode as cliffhanger.",
        "confidence": 1,
    }


def baseline_montage(plan_moments: list[dict], index: dict[str, Event], budget_total: float = 40.0) -> dict:
    """No AI: one segment per planned moment. Hook moments start at their peak; the last moment ends at its end."""
    n = len(plan_moments)
    budget = budget_total / max(n, 1)
    segments = []
    for j, m in enumerate(plan_moments):
        evs = sorted((e for e in index.values() if e.episode == m["episode"]
                      and e.start >= m["start"] - 0.01 and e.end <= m["end"] + 0.01), key=lambda e: (e.start, e.index))
        if not evs:
            continue
        if j == n - 1:
            last = max(evs, key=lambda e: e.end)
            first = min((e for e in evs if e.start >= last.end - budget), key=lambda e: e.start, default=last)
        else:
            peak = index.get(m["peak_event"]) or evs[0]
            first = peak
            last = max((e for e in evs if e.start >= peak.start and e.end <= peak.start + budget),
                       key=lambda e: e.end, default=peak)
        purpose = "hook" if j == 0 else ("cliffhanger" if j == n - 1 else "escalation")
        segments.append({"first_event": first.id, "last_event": last.id, "purpose": purpose})
    hook = index.get(plan_moments[0]["peak_event"])
    seg_first, seg_last = index[segments[-1]["first_event"]], index[segments[-1]["last_event"]]
    cliff = max((e for e in index.values() if e.episode == seg_last.episode and e.kind in ("DIALOGUE", "ACTION")
                 and e.start >= seg_first.start and e.end <= seg_last.end + 0.01), key=lambda e: e.end, default=seg_last)
    return {
        "title": "Baseline montage",
        "segments": segments,
        "hook": {"event": hook.id if hook else segments[0]["first_event"], "type": "other", "description": "baseline"},
        "cliffhanger": {"event": cliff.id, "open_question": plan_moments[-1].get("open_question", "")},
        "context_for_new_viewer": "(baseline does not judge this)",
        "why_it_works": "Rule-based baseline: planned moments trimmed to equal shares of 40s.",
        "confidence": 1,
    }
