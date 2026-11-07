"""Show bible: characters, themes and spoilers for one show (1 AI call, or a rule-based fallback)."""
from __future__ import annotations

import re
from collections import Counter

from .parser import Show
from .prompts import BIBLE_SYSTEM, build_bible_prompt

BASELINE_THEMES = [
    {"id": "betrayal", "label": "Betrayed by the people she trusted", "description": "jilted, lied to, sold out"},
    {"id": "humiliation", "label": "Publicly humiliated", "description": "mocked, insulted, degraded in front of others"},
    {"id": "protector", "label": "The man who stands up for her", "description": "rescues, defends, protects"},
    {"id": "romance", "label": "An unlikely love", "description": "vows, tenderness, longing"},
    {"id": "revenge", "label": "She fights back", "description": "defiance, comebacks, turning the tables"},
    {"id": "hidden_identity", "label": "He is not who they think", "description": "secret identity, disguises, reveals"},
    {"id": "family_cruelty", "label": "A cruel family", "description": "disowned, abandoned by family"},
    {"id": "danger", "label": "A deadly threat", "description": "poison, violence, plots, arrests"},
]

THEME_KEYWORDS = {
    "betrayal": ["betray", "marry", "marrying", "wedding", "lie", "lied", "changed my mind", "knew"],
    "humiliation": ["humiliat", "worthless", "plain", "beg", "scrub", "maid", "discard", "low value", "crawl", "mock"],
    "protector": ["hands off", "let her go", "protect", "save", "hurt", "touch her", "stand aside"],
    "romance": ["kiss", "love", "husband", "wife", "ring", "my hand", "vow"],
    "revenge": ["regret", "revenge", "vengeance", "pay for", "how dare", "never again", "choose my own"],
    "hidden_identity": ["duke", "who are you", "secret", "disguise", "truly", "identity", "nobody"],
    "family_cruelty": ["disown", "father", "mother", "family", "sister", "daughter"],
    "danger": ["poison", "kill", "gallows", "blood", "sword", "arrest", "dagger", "pistol", "dead", "arsenic"],
}


def speaker_counts(show: Show) -> Counter:
    return Counter(e.speaker for ep in show.episodes.values() for e in ep.events if e.kind == "DIALOGUE" and e.speaker)


def build_bible(show: Show, llm=None) -> dict:
    if llm is None:
        return baseline_bible(show)
    speakers = "\n".join(f"{name}: {n}" for name, n in speaker_counts(show).most_common(60))
    episodes = "\n".join(f"Ep {n}: {ep.logline}\n   Continuity: {ep.continuity}" for n, ep in show.episodes.items())
    raw = llm.complete_json(BIBLE_SYSTEM, build_bible_prompt(show.title, speakers, episodes), tag="bible")
    return normalize_bible(raw, show)


def normalize_bible(raw: dict, show: Show) -> dict:
    themes, seen = [], set()
    for t in raw.get("themes") or []:
        tid = re.sub(r"[^a-z0-9_]+", "_", str(t.get("id") or t.get("label") or "").lower()).strip("_")
        if tid and tid not in seen:
            seen.add(tid)
            themes.append({"id": tid, "label": str(t.get("label", tid)), "description": str(t.get("description", ""))})
    if not themes:
        themes = BASELINE_THEMES
    reveals = []
    for r in raw.get("big_reveals") or []:
        try:
            reveals.append({"episode": int(r.get("episode")), "reveal": str(r.get("reveal", ""))})
        except (TypeError, ValueError):
            continue
    return {
        "show": show.title,
        "premise": str(raw.get("premise", "")),
        "characters": [c for c in (raw.get("characters") or []) if isinstance(c, dict) and c.get("name")],
        "themes": themes,
        "big_reveals": reveals,
        "source": raw.get("source", "ai"),
    }


def baseline_bible(show: Show) -> dict:
    top = [name for name, _ in speaker_counts(show).most_common(8)]
    return {
        "show": show.title,
        "premise": show.episodes[min(show.episodes)].logline,
        "characters": [{"name": n.title(), "aliases": [n], "role": ""} for n in top],
        "themes": BASELINE_THEMES,
        "big_reveals": [],
        "source": "baseline",
    }


def bible_block(bible: dict, with_themes: bool = True, with_reveals: bool = True) -> str:
    """Compact text version for prompts."""
    lines = [f"Premise: {bible.get('premise', '')}", "Characters:"]
    for c in bible.get("characters", []):
        aliases = ", ".join(c.get("aliases", []))
        lines.append(f"- {c['name']} ({c.get('role', '')}){' - labels: ' + aliases if aliases else ''}")
    if with_themes:
        lines.append("Themes (use these ids):")
        lines += [f"- {t['id']}: {t['label']} ({t.get('description', '')})" for t in bible.get("themes", [])]
    if with_reveals and bible.get("big_reveals"):
        lines.append("Big reveals (spoilers - flag moments that show them):")
        lines += [f"- Ep {r['episode']}: {r['reveal']}" for r in bible["big_reveals"]]
    return "\n".join(lines)
