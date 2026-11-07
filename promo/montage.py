"""Theme montages: connect the best moments of the top episodes into 5-8 promos.

1. pools (code)    : moments from the top-N episodes, grouped by theme (ranking.montage_pools)
2. plan (AI)       : the AI designs montages from the pools: 3-4 moments, hook first, cliffhanger last
3. validate (code) : drop plans that break the rules; top up with rule-based plans if too few survive
4. cut (AI + code) : each plan goes to the promo maker in montage mode -> same checker as scene promos
"""
from __future__ import annotations

from .checker import Rules
from .prompts import PLANNER_SYSTEM, build_planner_prompt
from .promo_maker import Selection, make_promo


def pools_block(pools: dict[str, list[dict]], bible: dict) -> str:
    labels = {t["id"]: t["label"] for t in bible.get("themes", [])}
    out = []
    for theme, moms in pools.items():
        out.append(f"\nTHEME {theme}: {labels.get(theme, '')}")
        for m in moms:
            out.append(f"{m['id']} | ep {m['episode']} | {m['seconds']}s | {m['type']} | "
                       f"{m['hook_strength']}/{m['cliffhanger_strength']}/{m['standalone_clarity']}/{m['emotional_intensity']} | "
                       f"{m['spoiler_level']} | {m['summary']} | {m['open_question'] or '-'}")
    return "\n".join(out)


def validate_plans(raw: dict, moments: dict[str, dict], pool_ids: set[str], max_reuse: int = 2,
                   max_count: int = 8) -> tuple[list[dict], list[str]]:
    plans, notes, use, seen_sets = [], [], {}, []
    for i, p in enumerate((raw or {}).get("montages") or [], 1):
        if not isinstance(p, dict):
            continue
        ids = [str(x).strip().upper() for x in (p.get("moments") or [])]
        ids = list(dict.fromkeys(ids))  # drop repeats, keep order
        why_bad = None
        if not 3 <= len(ids) <= 4:
            why_bad = f"needs 3-4 moments, got {len(ids)}"
        elif any(x not in moments for x in ids):
            why_bad = f"unknown moment IDs {[x for x in ids if x not in moments]}"
        elif any(x not in pool_ids for x in ids):
            why_bad = f"uses moments outside the top episodes {[x for x in ids if x not in pool_ids]}"
        else:
            moms = [moments[x] for x in ids]
            rest = moms[1:]
            if len({m["episode"] for m in moms}) < 2:
                why_bad = "all moments from one episode"
            elif any((a["episode"], a["start"]) > (b["episode"], b["start"]) for a, b in zip(rest, rest[1:])):
                why_bad = "moments after the hook are not in story order"
            elif any(m["spoiler_level"] == "high" for m in moms):
                why_bad = "uses a spoiler-high moment"
            elif any(use.get(x, 0) >= max_reuse for x in ids):
                why_bad = f"a moment is already used in {max_reuse} montages"
            elif set(ids) in seen_sets:
                why_bad = "duplicate of another montage"
        if why_bad:
            notes.append(f"plan {i} ({p.get('title', '')}): dropped - {why_bad}")
            continue
        for x in ids:
            use[x] = use.get(x, 0) + 1
        seen_sets.append(set(ids))
        plans.append({"theme": str(p.get("theme", "")), "title": str(p.get("title", "")),
                      "through_line": str(p.get("through_line", "")), "why": str(p.get("why", "")),
                      "source": "ai", "moments": [moments[x] for x in ids]})
        if len(plans) == max_count:
            break
    return plans, notes


def baseline_plans(pools: dict[str, list[dict]], count: int, already: list[dict], max_reuse: int = 2) -> list[dict]:
    """Rule-based plans: per theme, best hook -> up to 2 best others in story order -> best cliffhanger."""
    use: dict[str, int] = {}
    for p in already:
        for m in p["moments"]:
            use[m["id"]] = use.get(m["id"], 0) + 1
    used_themes = {p["theme"] for p in already}
    plans = []
    for theme, moms in pools.items():
        if len(already) + len(plans) >= count:
            break
        if theme in used_themes:
            continue
        free = [m for m in moms if use.get(m["id"], 0) < max_reuse]
        if len(free) < 3:
            continue
        hook = max(free, key=lambda m: (m["hook_strength"], m["score"]))
        later = [m for m in free if m is not hook]
        cliff = max(later, key=lambda m: (m["cliffhanger_strength"], (m["episode"], m["start"])))
        middle = sorted((m for m in later if m is not cliff), key=lambda m: -m["score"])[:2 if len(free) >= 4 else 1]
        middle = [m for m in middle if (m["episode"], m["start"]) < (cliff["episode"], cliff["start"])]
        moms_ordered = [hook] + sorted(middle, key=lambda m: (m["episode"], m["start"])) + [cliff]
        if len(moms_ordered) < 3 or len({m["episode"] for m in moms_ordered}) < 2:
            continue
        for m in moms_ordered:
            use[m["id"]] = use.get(m["id"], 0) + 1
        plans.append({"theme": theme, "title": f"{theme} montage", "through_line": "", "why": "rule-based plan",
                      "source": "baseline", "moments": moms_ordered})
    return plans


def plan_montages(bank_moments: list[dict], pools: dict[str, list[dict]], bible: dict, bible_text: str,
                  llm=None, count: int = 6) -> tuple[list[dict], list[str]]:
    moments = {m["id"]: m for m in bank_moments}
    pool_ids = {m["id"] for moms in pools.values() for m in moms}
    plans, notes = [], []
    if llm is not None and pools:
        raw = llm.complete_json(PLANNER_SYSTEM.format(n=count), build_planner_prompt(bible_text, pools_block(pools, bible)),
                                tag="montage_plan")
        plans, notes = validate_plans(raw, moments, pool_ids, max_count=count)
    if len(plans) < count:
        extra = baseline_plans(pools, count, plans)
        if extra and llm is not None:
            notes.append(f"{len(extra)} rule-based plan(s) added to reach {count}")
        plans += extra
    return plans, notes


def build_montage(show, plan: dict, llm=None, bible_text: str = "", promo_id: str = "montage",
                  rules: Rules | None = None, max_attempts: int = 3) -> dict:
    # one window per moment (+1.5s either side), merged if two windows in the same episode touch
    windows: list[Selection] = []
    for m in sorted(plan["moments"], key=lambda m: (m["episode"], m["start"])):
        s, e = max(0.0, m["start"] - 1.5), m["end"] + 1.5
        if windows and windows[-1].episode == m["episode"] and s <= (windows[-1].end or 0):
            windows[-1].end = max(windows[-1].end or 0, e)
        else:
            windows.append(Selection(m["episode"], s, e))
    rules = rules or Rules()
    rules.allow_cold_open = True
    return make_promo(show, windows, llm=llm, rules=rules, max_attempts=max_attempts, mode="montage",
                      plan=plan, bible_text=bible_text, promo_id=promo_id)
