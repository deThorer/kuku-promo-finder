"""Offline tests: no API key needed. Run:  python tests/test_offline.py [path/to/show]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo.checker import check_promo  # noqa: E402
from promo.llm import extract_json  # noqa: E402
from promo.parser import load_show  # noqa: E402
from promo.promo_maker import Selection, make_promo, parse_selection, selected_events  # noqa: E402

SHOW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input")
try:
    show = load_show(SHOW_DIR)
except FileNotFoundError:
    sys.exit(f"The tests run on real scripts: put a show's episode .txt files in {SHOW_DIR}/ "
             f"(they are not in the repo - the input pack is confidential), or pass a folder: "
             f"python tests/test_offline.py path/to/show")
ALL = {n: ep.events for n, ep in show.episodes.items()}
passed = 0


def ok(cond: bool, msg: str) -> None:
    global passed
    assert cond, msg
    passed += 1
    print(f"  ok - {msg}")


def idx(*eps: int) -> dict:
    return {e.id: e for e in selected_events(show, [Selection(n) for n in eps])}


class FakeLLM:
    """Stands in for a real model: returns queued answers, records the prompts it got."""
    provider, model = "fake", "fake-model"

    def __init__(self, answers: list[dict]):
        self.answers, self.prompts = list(answers), []

    def complete_json(self, system: str, user: str, tag: str = "") -> dict:
        self.prompts.append(user)
        return self.answers.pop(0)


print("parser")
ok(len(show.episodes) >= 1, f"loaded {len(show.episodes)} episodes")
ok(all(e.end >= e.start for ep in show.episodes.values() for e in ep.events), "every event has end >= start")
ok(all(e.text for ep in show.episodes.values() for e in ep.events), "no empty events")

print("selection parsing")
s = parse_selection(["3-5", "12@00:40-02:10"])
ok([x.episode for x in s] == [3, 4, 5, 12] and s[-1].start == 40.0 and s[-1].end == 130.0, "ranges and time windows")

print("checker")
e1 = show.episodes[1].events
dlg = [e for e in e1 if e.kind == "DIALOGUE"]
hook = dlg[5]
# a) promo far too long + hook late -> hook trimmed (fix) but length still an issue
r = check_promo({"segments": [{"first_event": e1[0].id, "last_event": e1[-1].id}],
                 "hook": {"event": hook.id}, "cliffhanger": {"event": dlg[-1].id}}, idx(1), ALL)
ok(any("trimmed segment 1" in f for f in r.fixes), "late hook is auto-trimmed to open on the hook")
ok(r.hook_offset is not None and r.hook_offset <= 3.0, "hook now within first 3s")
ok(any("over 45" in i for i in r.issues), "too-long promo (>45s) is flagged for the AI to fix")

# b) a cut that lands mid-line is snapped outward
cam = next(c for c in e1 if c.kind == "CAMERA" and any(d.start < c.end - 0.05 < d.end - 0.1 and d.start < c.end - 0.1 for d in dlg))
straddled = next(d for d in dlg if d.start < cam.end - 0.05 and d.end > cam.end + 0.05)
first = next(e for e in e1 if e.start >= cam.end - 30)
r = check_promo({"segments": [{"first_event": first.id, "last_event": cam.id}],
                 "hook": {"event": first.id}, "cliffhanger": {"event": cam.id}}, idx(1), ALL)
ok(abs(r.segments[0].end - straddled.end) < 0.01 or r.segments[0].end >= straddled.end,
   f"cut at {cam.id} extended to finish line {straddled.id}")

# c) segment across an episode boundary is split
a, b = show.episodes[1].events[-10], show.episodes[2].events[5]
r = check_promo({"segments": [{"first_event": a.id, "last_event": b.id}],
                 "hook": {"event": a.id}, "cliffhanger": {"event": b.id}}, idx(1, 2), ALL)
ok(len(r.segments) == 2 and r.segments[0].episode == 1 and r.segments[1].episode == 2, "cross-episode segment split in two")

# d) invented IDs are rejected
r = check_promo({"segments": [{"first_event": "E01.999", "last_event": "E01.010"}],
                 "hook": {"event": "E01.999"}, "cliffhanger": {"event": "E01.010"}}, idx(1), ALL)
ok(not r.ok and any("not in the selected episodes" in i for i in r.issues), "invented event IDs rejected")

# e) IDs outside the selected episodes are rejected
r = check_promo({"segments": [{"first_event": "E05.001", "last_event": "E05.020"}],
                 "hook": {"event": "E05.001"}, "cliffhanger": {"event": "E05.020"}}, idx(1), ALL)
ok(not r.ok, "events from unselected episodes rejected")

# f) hook outside first segment -> issue
r = check_promo({"segments": [{"first_event": e1[0].id, "last_event": e1[10].id},
                              {"first_event": e1[40].id, "last_event": e1[60].id}],
                 "hook": {"event": e1[45].id}, "cliffhanger": {"event": e1[60].id}}, idx(1), ALL)
ok(any("not inside the first segment" in i for i in r.issues), "hook must be in segment 1")

# g) segments < 3s apart in the same episode are joined (the E03 v2 case: 01:03.50 / 01:06.00)
e3 = show.episodes[3].events
def at(ep_events, t):  # event starting exactly at time t
    return next(e for e in ep_events if abs(e.start - t) < 0.01)
s1a, s2a = at(e3, 58.3), at(e3, 66.0)
s1b = next(e for e in e3 if abs(e.end - 63.5) < 0.01)
s2b = at(e3, 85.6)
r = check_promo({"segments": [{"first_event": s1a.id, "last_event": s1b.id},
                              {"first_event": s2a.id, "last_event": s2b.id}],
                 "hook": {"event": s1a.id}, "cliffhanger": {"event": s2b.id}}, idx(3), ALL)
ok(len(r.segments) == 1 and any("Joined two segments" in f for f in r.fixes), "2.5s gap between segments is merged into one")
# h) a real skip (> 3s) is kept as two segments
s2c = at(e3, 119.8)
r = check_promo({"segments": [{"first_event": s1a.id, "last_event": s1b.id},
                              {"first_event": s2c.id, "last_event": s2c.id}],
                 "hook": {"event": s1a.id}, "cliffhanger": {"event": s2c.id}}, idx(3), ALL)
ok(len(r.segments) == 2, "a skip of more than 3s stays two segments")

print("repair loop (fake model)")
bad = {"title": "too long", "segments": [{"first_event": e1[0].id, "last_event": e1[-1].id}],
       "hook": {"event": hook.id}, "cliffhanger": {"event": dlg[-1].id}, "confidence": 3}
# a good answer: from the hook, ~35s, ending on a line
end_line = max((d for d in dlg if hook.start + 25 <= d.end <= hook.start + 40), key=lambda d: d.end)
good = {"title": "fixed", "segments": [{"first_event": hook.id, "last_event": end_line.id}],
        "hook": {"event": hook.id, "type": "shocking line", "description": "x"},
        "cliffhanger": {"event": end_line.id, "open_question": "y"}, "confidence": 4}
fake = FakeLLM([bad, good])
p = make_promo(show, [Selection(1)], llm=fake)
ok(p["status"] == "ok" and p["attempts"] == 2, "bad first answer -> checker report -> repaired on try 2")
ok("THE AUTOMATIC CHECKER" in fake.prompts[1] and "over 45" in fake.prompts[1], "repair prompt carries the checker's measurements")
ok(20 <= p["total_seconds"] <= 45 and p["hook"]["starts_at_promo_second"] <= 3, "final promo obeys length (20-45s) and hook rules")
ok(len(p["transcript"]) > 0 and p["transcript"][0]["promo_time"] == "00:00.00", "transcript built with promo-relative times")

fake = FakeLLM([bad, bad, bad])
p = make_promo(show, [Selection(1)], llm=fake, max_attempts=3)
ok(p["status"] == "failed_checks" and p["attempts"] == 3, "gives up honestly after max attempts")

print("last-resort fixes")
from promo.checker import tighten  # noqa: E402
# a single 52s stretch of episode 9 (7s over the limit) -> code cuts a middle chunk at line boundaries
e9 = show.episodes[9].events
d9 = [e for e in e9 if e.kind == "DIALOGUE"]
h9 = d9[3]
c9 = max((d for d in d9 if d.end <= h9.start + 53), key=lambda d: d.end)
long_p = {"segments": [{"first_event": h9.id, "last_event": c9.id}], "hook": {"event": h9.id}, "cliffhanger": {"event": c9.id}}
r = check_promo(long_p, idx(9), ALL)
ok(not r.ok and r.total > 45, f"setup: {r.total:.1f}s promo fails the 45s limit")
cut = tighten(long_p, idx(9), ALL)
r2 = check_promo(cut, idx(9), ALL) if cut else None
ok(cut is not None and r2.ok and len(r2.segments) == 2 and r2.hook_offset <= 3 and (r2.cliff_tail or 0) <= 3,
   "small overrun fixed by cutting the middle; hook first and cliffhanger last kept")
ok(tighten(bad, idx(1), ALL) is None, "a promo far too long is NOT cut by code (that is an editing decision)")
# two-episode promo, 5s over: the cut must keep the OTHER episode's segment intact
two = {"segments": [{"first_event": "E01.011", "last_event": "E01.027"}, {"first_event": "E02.062", "last_event": "E02.073"}],
       "hook": {"event": "E01.011"}, "cliffhanger": {"event": "E02.073"}}
cut2 = tighten(two, idx(1, 2), ALL)
r4 = check_promo(cut2, idx(1, 2), ALL) if cut2 else None
ok(cut2 is not None and r4.ok and r4.segments[-1].episode == 2 and abs(r4.segments[-1].end - 107.4) < 0.01,
   "multi-episode promo trimmed in one episode, the other episode's segment unchanged")
# a 1.4s opening segment is extended with the next whole lines instead of failing
short = {"segments": [{"first_event": d9[2].id, "last_event": d9[2].id}, {"first_event": d9[12].id, "last_event": d9[16].id}],
         "hook": {"event": d9[2].id}, "cliffhanger": {"event": d9[16].id}}
r3 = check_promo(short, idx(9), ALL)
ok(r3.segments[0].duration >= 2 and any("extended" in f for f in r3.fixes), "too-short segment extended by code")

print("json extraction")
ok(extract_json('```json\n{"a": 1}\n```') == {"a": 1}, "strips code fences")
ok(extract_json('Sure! {"a": 2} hope this helps') == {"a": 2}, "ignores prose around JSON")

print("baseline on every episode")
fails = []
for n in show.episodes:
    p = make_promo(show, [Selection(n)])
    if p["status"] != "ok":
        fails.append((n, p["checker"]["issues"]))
print(f"  baseline single-episode promos passing all hard rules: {len(show.episodes) - len(fails)}/{len(show.episodes)}")
for n, iss in fails:
    print(f"    E{n:02d}: {iss}")

print("moment bank validation (fake model answers)")
from promo.bible import baseline_bible  # noqa: E402
from promo.moments import validate_moments  # noqa: E402
from promo.montage import validate_plans  # noqa: E402
from promo.ranking import episode_ranking, pick_windows, scene_windows  # noqa: E402

bible = baseline_bible(show)
e3 = show.episodes[3].events
raw = {"moments": [
    {"first_event": "E03.046", "last_event": "E03.054", "peak_event": "E03.047", "type": "humiliation",
     "themes": ["family_cruelty", "made_up_theme"], "summary": "guards drag her out", "open_question": "who will help her?",
     "hook_strength": 5, "cliffhanger_strength": 4, "standalone_clarity": 5, "emotional_intensity": 5, "spoiler_level": "none"},
    {"first_event": "E03.999", "last_event": "E03.010", "peak_event": "E03.005"},                       # invented ID
    {"first_event": "E03.047", "last_event": "E03.050", "peak_event": "E03.047", "hook_strength": 9},   # overlaps #1
    {"first_event": "E03.116", "last_event": "E03.112", "peak_event": "E03.116", "themes": ["romance"],  # reversed
     "hook_strength": 3, "cliffhanger_strength": 5, "standalone_clarity": 4, "emotional_intensity": 4, "spoiler_level": "high"},
]}
moms, notes = validate_moments(raw, show, 3, bible)
ok(len(moms) == 2, "invented-ID and overlapping moments dropped, 2 kept")
ok(moms[0]["id"] == "M03.1" and moms[0]["themes"] == ["family_cruelty"], "best moment ranked first; unknown theme removed")
ok(any("reversed" in n for n in notes) and moms[1]["spoiler_level"] == "high", "reversed boundaries fixed; spoiler kept")
ok(moms[1]["score"] < moms[0]["score"], "spoiler-high moment is penalised in the score")

print("rankings + montage plan validation")
fake_bank = {"episodes": {}}
for n in (3, 4, 5, 6):
    fake_bank["episodes"][str(n)] = {"moments": [dict(moms[0], id=f"M{n:02d}.1", episode=n, spoiler_level="none")]}
eps_rank = episode_ranking(fake_bank)
ok(len(eps_rank) == 4 and eps_rank[0]["rank"] == 1, "episode ranking built from the bank")
wins = scene_windows(fake_bank, max_run=3)
def mm_ep(mid): return int(mid[1:3])
ok(len(wins) == 4 + 3 + 2, "runs of 1, 2, 3 over 4 episodes = 9 windows")
picked = pick_windows(wins, {1: 4, 2: 3, 3: 2}, episode_reuse=0)
ok(len({w["hook_moment"] for w in picked}) == len(picked), "picked windows never share a hook moment")
p1 = pick_windows(wins, {1: 4, 2: 3, 3: 2}, episode_reuse=1)
ok(len([e for w in p1 for e in w["episodes"]]) == len({e for w in p1 for e in w["episodes"]}), "with episode_reuse=1 no episode is in two picks")
ok(all(w["run"] == 1 or mm_ep(w["hook_moment"]) == w["episodes"][0] for w in wins), "multi-episode windows hook in their first episode")
ok(all(w["run"] == 1 or mm_ep(w["cliff_moment"]) == w["episodes"][-1] for w in wins), "multi-episode windows end in their last episode")
mm = {f"M{n:02d}.1": dict(moms[0], id=f"M{n:02d}.1", episode=n, spoiler_level="none") for n in (3, 4, 5, 6)}
pool = set(mm)
plans, pnotes = validate_plans({"montages": [
    {"theme": "x", "moments": ["M06.1", "M03.1", "M04.1"]},            # valid: cold open from ep 6, then 3 -> 4
    {"theme": "x", "moments": ["M03.1", "M05.1", "M04.1"]},            # after the hook, 5 then 4 = wrong order
    {"theme": "x", "moments": ["M03.1", "M99.1", "M04.1"]},            # unknown moment
    {"theme": "x", "moments": ["M03.1", "M04.1"]},                     # too few
    {"theme": "x", "moments": ["M06.1", "M03.1", "M04.1"]},            # duplicate of plan 1
]}, mm, pool)
ok(len(plans) == 1 and len(pnotes) == 4, "only the valid montage plan survives; 4 rejected with reasons")

print("judge: final ranking rules")
from promo.judge import final_ranking, footage_overlap  # noqa: E402
def fake_promo(pid, segs, status="ok", fixes=()):
    return {"promo_id": pid, "status": status, "total_seconds": sum(b - a for _, a, b in segs),
            "segments": [{"episode": e, "start": f"00:{a:05.2f}", "end": f"00:{b:05.2f}"} for e, a, b in segs],
            "checker": {"fixes": list(fixes)}}
A = fake_promo("A", [(3, 10, 40)]); B = fake_promo("B", [(3, 15, 35)]); C = fake_promo("C", [(7, 0, 30)], status="failed_checks")
D = fake_promo("D", [(9, 0, 30)])
sc = {k: {"hook": 8, "cliffhanger": 8, "clarity": 8, "spoiler_risk": "low", "duplicates": [], "verdict": ""} for k in "ABCD"}
sc["B"]["hook"] = 9; sc["D"]["spoiler_risk"] = "high"
cv = {k: {"keep_watching": 4} for k in "ABCD"}
ok(abs(footage_overlap(A, B) - 1.0) < 1e-6, "footage overlap: B lies fully inside A")
order = [r["promo"]["promo_id"] for r in final_ranking([A, B, C, D], cv, sc)]
ok(order[0] == "B" and order.index("A") > order.index("D") and order[-1] == "C",
   "best first; a duplicate drops below non-duplicates; a failed promo ranks last")

E = fake_promo("E", [(12, 0, 30)]); F = fake_promo("F", [(2, 0, 30)]); G = fake_promo("G", [(40, 0, 30)])
sc2 = {k: {"hook": 8, "cliffhanger": 8, "clarity": 8, "spoiler_risk": "low", "duplicates": [], "verdict": ""} for k in "EFG"}
sc2["E"]["hook"] = 9; sc2["F"]["duplicates"] = ["E"]; sc2["G"]["hook"] = 10
cv2 = {k: {"keep_watching": 4} for k in "EFG"}
r2 = {r["promo"]["promo_id"]: r for r in final_ranking([E, F, G], cv2, sc2, final_act_from=36)}
ok(not any("duplicate" in f for f in r2["F"]["flags"]), "judge-only duplicate claim ignored when promos share no episode")
ok(r2["G"]["rank"] > r2["E"]["rank"] and any("final act" in f for f in r2["G"]["flags"]), "final-act promo penalised in the final score")

print(f"\n{passed} checks passed")
