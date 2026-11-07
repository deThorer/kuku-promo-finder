"""Rule checker: turns the AI's event-ID answer into real timestamps and enforces the hard rules.

Code, not AI. It does two things:
  - small safe FIXES it can make itself (logged), e.g. snapping a cut to the edge of a dialogue line,
    trimming dead time before the hook, splitting a segment that runs across an episode boundary;
  - ISSUES it cannot fix without editorial judgement (sent back to the AI), e.g. promo too long.

What code cannot check: whether the hook is actually gripping, whether the ending is truly unresolved,
and whether a stranger would follow it. Those stay with the AI judge and the human editor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parser import Event, fmt_ts

EPS = 0.05


@dataclass
class Rules:
    min_total: float = 20.0
    max_total: float = 45.0      # brief allows 60; 45 is our hard limit (short promos perform better on social)
    merge_gap: float = 3.0       # segments in the same episode closer than this are joined (no jumpy micro-cuts)
    hook_window: float = 3.0     # hook must start within this many seconds of promo start
    cliff_tail: float = 3.0      # at most this much footage after the cliffhanger event ends
    min_segment: float = 2.0
    max_segments: int = 4
    allow_cold_open: bool = False  # montages: segment 1 (the hook) may come from a later episode


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_event: str
    last_event: str
    purpose: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, e: Event) -> bool:
        return e.episode == self.episode and e.start < self.end - EPS and e.end > self.start + EPS

    def describe(self) -> str:
        return f"E{self.episode:02d} {fmt_ts(self.start)}-{fmt_ts(self.end)} ({self.duration:.1f}s)"


@dataclass
class CheckResult:
    segments: list[Segment] = field(default_factory=list)
    total: float = 0.0
    hook_offset: float | None = None
    cliff_tail: float | None = None
    fixes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and bool(self.segments)

    def report(self) -> str:
        lines = [f"- Segment {i}: {s.describe()}" for i, s in enumerate(self.segments, 1)]
        lines.append(f"- Total length: {self.total:.1f}s")
        if self.hook_offset is not None:
            lines.append(f"- Hook starts at promo second {self.hook_offset:.1f}")
        if self.cliff_tail is not None:
            lines.append(f"- Footage after the cliffhanger ends: {self.cliff_tail:.1f}s")
        lines += [f"- PROBLEM: {x}" for x in self.issues]
        return "\n".join(lines)


def _norm_id(x) -> str:
    return str(x or "").strip().upper()


def snap_to_lines(seg: Segment, ep_events: list[Event], fixes: list[str], label: str) -> None:
    """Move cuts outward so no line of dialogue is cut in half."""
    changed = True
    while changed:
        changed = False
        for e in ep_events:
            if e.kind != "DIALOGUE":
                continue
            if e.start < seg.start - EPS and e.end > seg.start + EPS:
                fixes.append(f"{label}: start moved {fmt_ts(seg.start)} -> {fmt_ts(e.start)} so line {e.id} is not cut.")
                seg.start = e.start
                changed = True
            if e.start < seg.end - EPS and e.end > seg.end + EPS:
                fixes.append(f"{label}: end moved {fmt_ts(seg.end)} -> {fmt_ts(e.end)} so line {e.id} is not cut.")
                seg.end = e.end
                changed = True


def _merge_close(segments: list[Segment], gap: float, fixes: list[str]) -> list[Segment]:
    """Join consecutive segments of the same episode that are < `gap` seconds apart (or overlap)."""
    merged: list[Segment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if prev and prev.episode == seg.episode and seg.start >= prev.start and seg.start - prev.end < gap:
            g = seg.start - prev.end
            what = f"overlapped by {-g:.1f}s" if g < 0 else f"were only {g:.1f}s apart"
            fixes.append(f"Joined two segments of episode {seg.episode} that {what} "
                         f"({fmt_ts(prev.end)} / {fmt_ts(seg.start)}); a cut that small looks like a jump.")
            prev.end = max(prev.end, seg.end)
            prev.last_event = seg.last_event
            prev.purpose = " + ".join(x for x in (prev.purpose, seg.purpose) if x)
        else:
            merged.append(seg)
    return merged


def check_promo(promo: dict, index: dict[str, Event], episode_events: dict[int, list[Event]],
                rules: Rules | None = None) -> CheckResult:
    """index: every event of the show by ID; episode_events: ALL events per episode (for line snapping).
    Only IDs present in `index` are accepted - pass an index restricted to the selection."""
    rules = rules or Rules()
    res = CheckResult()
    raw_segments = promo.get("segments") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        res.issues.append("No segments returned.")
        return res

    # 1. Event IDs -> timed segments
    for i, s in enumerate(raw_segments, 1):
        s = s if isinstance(s, dict) else {}
        a_id = _norm_id(s.get("first_event") or s.get("start_event"))
        b_id = _norm_id(s.get("last_event") or s.get("end_event"))
        a, b = index.get(a_id), index.get(b_id)
        if a is None or b is None:
            bad = [x for x, ev in ((a_id, a), (b_id, b)) if ev is None]
            res.issues.append(f"Segment {i} uses event ID(s) not in the selected episodes: {', '.join(bad) or '(empty)'}.")
            continue
        if (a.episode, a.start) > (b.episode, b.start):
            res.issues.append(f"Segment {i}: first_event {a.id} comes after last_event {b.id}.")
            continue
        purpose = str(s.get("purpose", ""))
        if a.episode == b.episode:
            res.segments.append(Segment(a.episode, a.start, max(b.end, a.start), a.id, b.id, purpose))
        else:
            # A segment cannot play across an episode boundary: split it.
            for ep in range(a.episode, b.episode + 1):
                evs = [e for e in episode_events.get(ep, []) if e.id in index]
                if not evs:
                    continue
                first = a if ep == a.episode else min(evs, key=lambda e: e.start)
                last = b if ep == b.episode else max(evs, key=lambda e: e.end)
                res.segments.append(Segment(ep, first.start, last.end, first.id, last.id, purpose))
            res.fixes.append(f"Segment {i} ran from episode {a.episode} into {b.episode}; split at the episode boundary.")
    if not res.segments:
        return res
    res.segments = _merge_close(res.segments, rules.merge_gap, res.fixes)

    # 2. Hook: must sit in segment 1 and start within the hook window
    hook_id = _norm_id((promo.get("hook") or {}).get("event"))
    hook = index.get(hook_id)
    seg0 = res.segments[0]
    if hook is None:
        res.issues.append(f"Hook event '{hook_id}' is not an event in the selected episodes.")
    elif not seg0.overlaps(hook):
        res.issues.append(f"Hook {hook.id} is not inside the first segment; the promo must open on the hook.")
    elif hook.start - seg0.start > rules.hook_window:
        res.fixes.append(f"Hook {hook.id} started {hook.start - seg0.start:.1f}s in; trimmed segment 1 to open on it.")
        seg0.start = hook.start

    # 3. Cliffhanger: must sit in the last segment and be (nearly) the last thing shown
    cliff_id = _norm_id((promo.get("cliffhanger") or {}).get("event"))
    cliff = index.get(cliff_id)
    last = res.segments[-1]
    if cliff is None:
        res.issues.append(f"Cliffhanger event '{cliff_id}' is not an event in the selected episodes.")
    elif not last.overlaps(cliff) and not (cliff.episode == last.episode and abs(cliff.start - last.end) < EPS):
        res.issues.append(f"Cliffhanger {cliff.id} is not inside the last segment; the promo must end on it.")
    else:
        if cliff.end > last.end + EPS:
            res.fixes.append(f"Extended the last segment to {fmt_ts(cliff.end)} to include the full cliffhanger {cliff.id}.")
            last.end = cliff.end
        elif last.end - cliff.end > rules.cliff_tail:
            res.fixes.append(f"Trimmed {last.end - cliff.end:.1f}s after cliffhanger {cliff.id} so the promo ends on it.")
            last.end = cliff.end

    # 4. No cut lines
    for i, seg in enumerate(res.segments, 1):
        snap_to_lines(seg, episode_events.get(seg.episode, []), res.fixes, f"Segment {i}")
    # snapping can push two segments close together again
    res.segments = _merge_close(res.segments, rules.merge_gap, res.fixes)

    # too-short segments: grow them by whole events (the last one grows backwards, so it still ends on the cliffhanger)
    for i, seg in enumerate(res.segments):
        if seg.duration >= rules.min_segment:
            continue
        evs = sorted((e for e in episode_events.get(seg.episode, []) if e.id in index), key=lambda e: (e.start, e.index))
        before = seg.duration
        if i < len(res.segments) - 1:
            nxt = res.segments[i + 1]
            for e in evs:
                if seg.duration >= rules.min_segment + 1:
                    break
                if e.start >= seg.end - EPS and not (nxt.episode == seg.episode and e.end > nxt.start - rules.merge_gap):
                    seg.end = max(seg.end, e.end)
        else:
            prv = res.segments[i - 1] if i else None
            for e in reversed(evs):
                if seg.duration >= rules.min_segment + 1:
                    break
                if e.end <= seg.start + EPS and not (prv and prv.episode == seg.episode and e.start < prv.end + rules.merge_gap):
                    seg.start = min(seg.start, e.start)
        if seg.duration > before:
            snap_to_lines(seg, episode_events.get(seg.episode, []), res.fixes, f"Segment {i + 1}")
            res.fixes.append(f"Segment {i + 1} was only {before:.1f}s; extended to {seg.duration:.1f}s with the next whole lines.")
    seg0, last = res.segments[0], res.segments[-1]

    # 5. Measure and check hard rules
    res.total = sum(s.duration for s in res.segments)
    if hook is not None and seg0.overlaps(hook):
        res.hook_offset = max(0.0, hook.start - seg0.start)
        if res.hook_offset > rules.hook_window + EPS:
            res.issues.append(f"Hook starts at second {res.hook_offset:.1f}; it must start within {rules.hook_window:.0f}s.")
    if cliff is not None and last.episode == cliff.episode:
        res.cliff_tail = max(0.0, last.end - cliff.end)
        if res.cliff_tail > rules.cliff_tail + 1.5:  # allow a little extra when line-snapping pushed the end
            res.issues.append(f"{res.cliff_tail:.1f}s of footage plays after the cliffhanger; end on it.")
    if len(res.segments) > rules.max_segments:
        res.issues.append(f"{len(res.segments)} segments; use at most {rules.max_segments}.")
    for i, seg in enumerate(res.segments, 1):
        if seg.duration < rules.min_segment:
            res.issues.append(f"Segment {i} is only {seg.duration:.1f}s; make it at least {rules.min_segment:.0f}s or drop it.")
    for i in range(1, len(res.segments)):
        if rules.allow_cold_open and i == 1:
            continue  # the cold-open hook may come from anywhere; the rest must be in story order
        p, c = res.segments[i - 1], res.segments[i]
        if (c.episode, c.start) < (p.episode, p.end - EPS):
            res.issues.append(f"Segment {i + 1} overlaps or comes before segment {i}; keep segments in story order.")
    if res.total < rules.min_total:
        res.issues.append(f"Total length {res.total:.1f}s is under {rules.min_total:.0f}s; add setup or escalation.")
    if res.total > rules.max_total:
        res.issues.append(f"Total length {res.total:.1f}s is over {rules.max_total:.0f}s; cut {res.total - rules.max_total:.0f}s+ "
                          f"(skip slow middle parts using an extra segment, or start/end tighter).")
    return res


def tighten(promo: dict, index: dict[str, Event], episode_events: dict[int, list[Event]],
            rules: Rules | None = None, max_cut: float = 15.0) -> dict | None:
    """Last resort when the model keeps returning a promo that is too long.
    Cuts the smallest chunk that fixes the length out of the MIDDLE of one segment, at clean line boundaries,
    never touching the hook or the cliffhanger. Only small overruns (<= max_cut seconds) are fixed this way;
    anything bigger is an editing decision and stays a failure. Returns a new promo dict or None."""
    rules = rules or Rules()
    res = check_promo(promo, index, episode_events, rules)
    excess = res.total - rules.max_total
    if not res.segments or excess <= 0 or excess > max_cut or len(res.segments) >= rules.max_segments:
        return None
    hook = index.get(_norm_id((promo.get("hook") or {}).get("event")))
    cliff = index.get(_norm_id((promo.get("cliffhanger") or {}).get("event")))
    best = None
    for si, seg in enumerate(res.segments):
        evs = [e for e in episode_events.get(seg.episode, []) if e.id in index
               and e.start >= seg.start - EPS and e.end <= seg.end + EPS]
        dlg = [e for e in episode_events.get(seg.episode, []) if e.kind == "DIALOGUE"]

        def clean(t: float) -> bool:
            return not any(d.start < t - EPS and d.end > t + EPS for d in dlg)

        keep_head = rules.min_segment + 1
        if si == 0 and hook is not None and seg.overlaps(hook):
            keep_head = max(keep_head, hook.end - seg.start)
        keep_tail = rules.min_segment + 1
        if si == len(res.segments) - 1 and cliff is not None and seg.overlaps(cliff):
            keep_tail = max(keep_tail, seg.end - cliff.start)
        lo, hi = seg.start + keep_head, seg.end - keep_tail
        ends = sorted({e.end for e in evs if clean(e.end) and lo <= e.end})
        starts = sorted({e.start for e in evs if clean(e.start) and e.start <= hi})
        for a in ends:
            b = next((t for t in starts if t - a >= max(excess, rules.merge_gap)), None)
            if b is None or b - a > max_cut + rules.merge_gap:
                continue
            cand = (b - a - excess, abs((a + b) / 2 - (seg.start + seg.end) / 2), si, a, b)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    _, _, si, a, b = best
    seg = res.segments[si]

    def ep_events(ep: int) -> list[Event]:
        return sorted((e for e in episode_events.get(ep, []) if e.id in index), key=lambda e: (e.start, e.index))

    def first_at(ep: int, t: float) -> Event:
        return min((e for e in ep_events(ep) if e.start >= t - EPS), key=lambda e: (e.start, e.index))

    def last_at(ep: int, t: float, after: float) -> Event:
        """the event that ends latest (<= t) among events starting inside the piece (>= after)"""
        return max((e for e in ep_events(ep) if e.end <= t + EPS and e.start >= after - EPS), key=lambda e: (e.end, -e.index))

    new_segments = []
    for i, sg in enumerate(res.segments):
        ep = sg.episode
        if i != si:
            new_segments.append({"first_event": first_at(ep, sg.start).id, "last_event": last_at(ep, sg.end, sg.start).id,
                                 "purpose": sg.purpose})
        else:
            new_segments.append({"first_event": first_at(ep, sg.start).id, "last_event": last_at(ep, a, sg.start).id,
                                 "purpose": sg.purpose})
            new_segments.append({"first_event": first_at(ep, b).id, "last_event": last_at(ep, sg.end, b).id,
                                 "purpose": sg.purpose + " (after code cut)"})
    out = dict(promo, segments=new_segments)
    check = check_promo(out, index, episode_events, rules)
    if not check.ok:
        return None
    out["_code_cut"] = f"code cut {b - a:.1f}s from the middle of segment {si + 1} (episode {seg.episode}, " \
                       f"{fmt_ts(a)}-{fmt_ts(b)}) at line boundaries because the model kept exceeding {rules.max_total:.0f}s"
    return out
