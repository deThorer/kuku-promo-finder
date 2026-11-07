"""Rankings computed from the moment bank. Pure code, no AI calls.

- episode ranking: an episode is as good as its 3 best moments
- scene windows: every run of 1..max_run consecutive episodes, scored by its best hook -> cliffhanger pair
- montage pools: moments from the top episodes, grouped by theme
"""
from __future__ import annotations

import copy

from .moments import SPOILER_PENALTY, all_moments, moment_score


def apply_story_policy(bank: dict, bible: dict, final_act_from: int, penalty: float) -> dict:
    """Return a COPY of the bank with spoiler-aware scores (the saved bank is never changed).

    Promos exist to send new viewers to episode 1, so showing the final act (how the big conflict ends)
    spoils the show. The AI under-flags spoilers, so two code rules back it up:
      - moments in episodes >= final_act_from lose `penalty` points
      - moments in a big-reveal episode from the bible count as at least a 'low' spoiler"""
    b = copy.deepcopy(bank)
    reveal_eps = {r["episode"] for r in bible.get("big_reveals", [])}
    for m in all_moments(b):
        notes = []
        if m["episode"] in reveal_eps and m["spoiler_level"] == "none":
            m["spoiler_level"] = "low"
            notes.append("big-reveal episode")
        late = penalty if final_act_from and m["episode"] >= final_act_from else 0.0
        if late:
            notes.append(f"final act (ep {final_act_from}+)")
        m["late_penalty"] = late
        m["score"] = round(moment_score(m) - late, 3)
        m["policy_notes"] = notes
    return b


def episode_ranking(bank: dict, top_k: int = 3) -> list[dict]:
    rows = []
    for key, ep in bank.get("episodes", {}).items():
        moms = sorted(ep.get("moments", []), key=lambda m: -m["score"])
        best = [m["score"] for m in moms[:top_k]]
        best += [0.0] * (top_k - len(best))  # fewer strong moments -> lower score
        rows.append({
            "episode": int(key),
            "score": round(sum(best) / top_k, 3),
            "best_moment": moms[0]["id"] if moms else None,
            "best_summary": moms[0]["summary"] if moms else "",
            "moments": [m["id"] for m in moms],
        })
    rows.sort(key=lambda r: (-r["score"], r["episode"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _pair_score(h: dict, c: dict) -> float:
    """Hook moment h opens, cliffhanger moment c closes (c can be h in a single-episode window)."""
    pen = SPOILER_PENALTY.get(h["spoiler_level"], 0) + (SPOILER_PENALTY.get(c["spoiler_level"], 0) if c is not h else 0)
    pen += h.get("late_penalty", 0) + (c.get("late_penalty", 0) if c is not h else 0)
    return (0.45 * h["hook_strength"] + 0.30 * c["cliffhanger_strength"]
            + 0.15 * (h["standalone_clarity"] + c["standalone_clarity"]) / 2
            + 0.10 * max(h["emotional_intensity"], c["emotional_intensity"]) - pen)


def scene_windows(bank: dict, max_run: int = 3) -> list[dict]:
    """Score every run of 1..max_run consecutive episodes.

    - run of 1: best hook moment followed by a cliffhanger moment in the same episode
    - run of 2-3: the hook must be in the FIRST episode and the cliffhanger in the LAST one, so a multi-episode
      window really joins a setup to a later payoff (otherwise it is just a single-episode promo in disguise)
    Many AI scores are 5/5, so pair scores tie a lot. Ties are broken by `depth` (the overall scores of the two
    moments, then the average moment score of the whole window) instead of episode order."""
    by_ep: dict[int, list[dict]] = {}
    for m in all_moments(bank):
        by_ep.setdefault(m["episode"], []).append(m)
    eps = sorted(by_ep)
    windows = []
    for run in range(1, max_run + 1):
        for first in eps:
            run_eps = list(range(first, first + run))
            if not all(e in by_ep for e in run_eps):
                continue
            last = run_eps[-1]
            moms = sorted((m for e in run_eps for m in by_ep[e]), key=lambda m: (m["episode"], m["start"]))
            hooks = [m for m in moms if m["episode"] == first]
            best = None
            for h in hooks:
                for c in moms:
                    if (c["episode"], c["start"]) < (h["episode"], h["start"]):
                        continue
                    if run > 1 and c["episode"] != last:
                        continue
                    s = _pair_score(h, c)
                    depth = (h["score"] + c["score"]) / 2
                    if best is None or (s, depth) > (best[0], best[1]):
                        best = (s, depth, h, c)
            if best:
                s, depth, h, c = best
                window_avg = sum(m["score"] for m in moms) / len(moms)
                windows.append({
                    "episodes": run_eps, "run": run, "score": round(s, 3), "depth": round(depth, 3),
                    "window_avg": round(window_avg, 3),
                    "hook_moment": h["id"], "cliff_moment": c["id"],
                    "theme": (h["themes"] or c["themes"] or ["-"])[0],
                    "hook_summary": h["summary"], "cliff_question": c["open_question"],
                })
    windows.sort(key=lambda w: (-w["score"], -w["depth"], -w["window_avg"], w["episodes"][0]))
    return windows


def pick_windows(windows: list[dict], per_run: dict[int, int], max_per_theme: int = 3,
                 episode_reuse: int = 1, min_score: float = 0.0) -> list[dict]:
    """Best windows per category (runs of 1, 2, 3 episodes).
    - no hook or cliffhanger moment is used twice
    - each episode appears in at most `episode_reuse` picks (otherwise a pair/triple usually re-cuts the
      same best scene as a single episode it contains)
    - at most `max_per_theme` picks open on the same theme (variety)
    - windows scoring below `min_score` are never picked (a category may stay short rather than get a weak promo)
    Longest runs are picked FIRST: a strong 3-episode arc is the rarest window, and singles can fill the gaps."""
    picked, used, theme_count, ep_count = [], set(), {}, {}
    for run in sorted(per_run, reverse=True):
        want, got = per_run[run], 0
        for w in windows:
            if got == want:
                break
            if w["run"] != run or w["hook_moment"] in used or w["cliff_moment"] in used:
                continue
            if w["score"] < min_score:
                break  # windows are sorted by score: nothing better left for this run length
            if theme_count.get(w["theme"], 0) >= max_per_theme and w["theme"] != "-":
                continue
            if episode_reuse and any(ep_count.get(e, 0) >= episode_reuse for e in w["episodes"]):
                continue
            picked.append(w)
            used.update({w["hook_moment"], w["cliff_moment"]})
            theme_count[w["theme"]] = theme_count.get(w["theme"], 0) + 1
            for e in w["episodes"]:
                ep_count[e] = ep_count.get(e, 0) + 1
            got += 1
    return sorted(picked, key=lambda w: (w["run"], -w["score"], -w["depth"]))


def montage_pools(bank: dict, top_episodes: list[int], per_theme: int = 8, min_moments: int = 3) -> dict[str, list[dict]]:
    """Theme -> best moments from the top episodes (spoiler-high moments excluded).
    A theme qualifies only if it has >= min_moments moments across >= 2 episodes."""
    allowed = set(top_episodes)
    pools: dict[str, list[dict]] = {}
    for m in all_moments(bank):
        if m["episode"] not in allowed or m["spoiler_level"] == "high":
            continue
        for t in m["themes"]:
            pools.setdefault(t, []).append(m)
    out = {}
    for t, moms in pools.items():
        moms = sorted(moms, key=lambda m: -m["score"])[:per_theme]
        if len(moms) >= min_moments and len({m["episode"] for m in moms}) >= 2:
            out[t] = moms
    return dict(sorted(out.items(), key=lambda kv: -sum(m["score"] for m in kv[1][:4])))


def ranking_markdown(eps: list[dict], windows: list[dict], picked: list[dict], top_n: int) -> str:
    out = ["# Rankings", "", f"## Top {top_n} episodes (score = mean of the best moment scores; see --rank-k)", "",
           "| Rank | Episode | Score | Best moment |", "|---|---|---|---|"]
    out += [f"| {r['rank']} | {r['episode']} | {r['score']} | {r['best_moment']}: {r['best_summary'].replace('|', '/')} |"
            for r in eps[:top_n]]
    out += ["", f"## Scene windows: {len(windows)} runs scored, {len(picked)} picked for promos", "",
            "| Pick | Episodes | Score | Tie-break | Theme | Hook moment | Cliffhanger moment | Open question |",
            "|---|---|---|---|---|---|---|---|"]
    for i, w in enumerate(picked, 1):
        out.append(f"| {i} | {'-'.join(map(str, w['episodes']))} | {w['score']} | {w['depth']} | {w['theme']} | {w['hook_moment']}: "
                   f"{w['hook_summary'][:80].replace('|', '/')} | {w['cliff_moment']} | {w['cliff_question'].replace('|', '/')} |")
    return "\n".join(out) + "\n"
