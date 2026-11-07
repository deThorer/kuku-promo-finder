"""Parse microdrama screenplays into numbered, timed events.

Pure code, no AI. Every timestamp the tool ever outputs comes from here,
so the model can never invent one.

Event IDs look like E03.017 = episode 3, event 17 (in file order).
Timestamps restart at 00:00 in every episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TS = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
_TIME_RE = re.compile(r"^Start:\s*\(\s*" + _TS + r"\s*\)\s*\|\s*End:\s*\(\s*" + _TS + r"\s*\)")
_LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")

_SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_HEADING_RE = re.compile(r"^Scene heading:\s*(.*)$", re.I)
_STRUCTURAL_RE = re.compile(r"^(Scene (start|end):|Actions and dialogue:|---)", re.I)
_HEADER_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$", re.I)
_FILE_EP_RE = re.compile(r"episode-(\d+)", re.I)

NON_DIALOGUE = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SOUND",
    "SOUND": "SOUND",
    "SFX": "SOUND",
    "MUSIC": "SOUND",
}


def parse_ts(text: str) -> float:
    """'02:18.00' -> 138.0 ; also accepts '1:02:18.00'."""
    parts = text.strip().split(":")
    sec = float(parts[-1])
    mins = int(parts[-2])
    hrs = int(parts[-3]) if len(parts) == 3 else 0
    return hrs * 3600 + mins * 60 + sec


def fmt_ts(seconds: float) -> str:
    """138.0 -> '02:18.00'."""
    seconds = max(0.0, round(seconds, 2))
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    id: str
    episode: int
    index: int
    kind: str  # CAMERA | ACTION | SOUND | DIALOGUE
    speaker: str | None
    emotion: str | None
    start: float
    end: float
    text: str
    scene: int
    heading: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        if self.kind == "DIALOGUE":
            return f"{self.speaker} ({self.emotion})" if self.emotion else str(self.speaker)
        return self.kind

    def line(self) -> str:
        """Compact one-line form used in prompts."""
        return f"{self.id} [{fmt_ts(self.start)}-{fmt_ts(self.end)}] {self.label()}: {self.text}"


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    events: list[Event]
    path: str

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)


@dataclass
class Show:
    title: str
    episodes: dict[int, Episode]

    def event_index(self) -> dict[str, Event]:
        return {e.id: e for ep in self.episodes.values() for e in ep.events}


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    number: int | None = None
    headers = {"LOGLINE": "", "CONTINUITY": ""}
    header_key: str | None = None
    in_script = False

    scene, heading = 0, ""
    pending_label: str | None = None
    current: dict | None = None
    raw_events: list[dict] = []

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["text"] = " ".join(current["text"]).strip()
            raw_events.append(current)
            current = None

    for raw in lines:
        line = raw.strip()

        # ---- header block (before --- SCRIPT START ---)
        if not in_script:
            if line.upper().startswith("--- SCRIPT START"):
                in_script = True
                continue
            m = _HEADER_RE.match(line)
            if m:
                key, val = m.group(1).upper(), m.group(2).strip()
                if key == "EPISODE NUMBER":
                    n = re.search(r"\d+", val)
                    number = int(n.group()) if n else None
                    header_key = None
                else:
                    header_key = "LOGLINE" if "LOGLINE" in key else "CONTINUITY"
                    headers[header_key] = val
            elif line and header_key:
                headers[header_key] += " " + line
            continue

        # ---- script body
        m = _SCENE_RE.match(line)
        if m:
            flush()
            scene = int(m.group(1))
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush()
            heading = m.group(1).strip()
            continue
        if _STRUCTURAL_RE.match(line):
            flush()
            continue
        m = _LABEL_RE.match(line)
        if m:
            flush()
            pending_label = m.group(1).strip()
            continue
        m = _TIME_RE.match(line)
        if m and pending_label is not None:
            label = pending_label
            pending_label = None
            kind = NON_DIALOGUE.get(label.upper())
            speaker = emotion = None
            if kind is None:
                kind = "DIALOGUE"
                # "THE BEARDED STRANGER (O.S.) (Deep, Resonant)" -> speaker + "O.S.; Deep, Resonant"
                groups = [g.strip() for g in re.findall(r"\(([^)]*)\)", label)]
                speaker = label.split("(")[0].strip() or label
                emotion = "; ".join(groups) if groups else None
            current = {
                "kind": kind, "speaker": speaker, "emotion": emotion,
                "start": parse_ts(m.group(1)), "end": parse_ts(m.group(2)),
                "text": [], "scene": scene, "heading": heading,
            }
            continue
        if line and current is not None:
            current["text"].append(line)
    flush()

    if number is None:
        fm = _FILE_EP_RE.search(path.name)
        number = int(fm.group(1)) if fm else 0

    events = []
    for i, r in enumerate(raw_events, 1):
        text = r["text"]
        if r["kind"] == "DIALOGUE":
            text = text.strip().strip('"“”').strip()
        events.append(Event(
            id=f"E{number:02d}.{i:03d}", episode=number, index=i, kind=r["kind"],
            speaker=r["speaker"], emotion=r["emotion"], start=r["start"], end=max(r["end"], r["start"]),
            text=text, scene=r["scene"], heading=r["heading"],
        ))
    return Episode(number=number, logline=headers["LOGLINE"].strip(),
                   continuity=headers["CONTINUITY"].strip(), events=events, path=str(path))


def load_show(folder: str | Path) -> Show:
    """Load every episode .txt under `folder` (searches subfolders too)."""
    folder = Path(folder)
    files = sorted(folder.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No episode .txt files found under {folder}")
    episodes: dict[int, Episode] = {}
    for f in files:
        ep = parse_episode(f)
        if ep.events:
            episodes[ep.number] = ep
    parents = {f.parent for f in files}
    title = parents.pop().name if len(parents) == 1 else folder.name
    return Show(title=title, episodes=dict(sorted(episodes.items())))
