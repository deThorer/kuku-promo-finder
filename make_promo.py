"""Make one promo from selected episode(s).

Examples
  python make_promo.py --list
  python make_promo.py --episodes 1
  python make_promo.py --episodes 3 4
  python make_promo.py --episodes 10-12
  python make_promo.py --episodes "12@00:40-02:10"        # only that part of episode 12
  python make_promo.py --episodes 1 2 --provider baseline  # no AI, no key needed
  python make_promo.py --episodes 5 --provider groq --model <groq model>

Outputs go to outputs/promos/<id>.json (full detail) and <id>.md (readable),
every model call is logged in outputs/llm_logs/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from promo.checker import Rules
from promo.llm import LLM, LLMError, load_env
from promo.parser import fmt_ts, load_show
from promo.promo_maker import make_promo, parse_selection, to_markdown


def default_show_dir() -> Path:
    base = Path("input")
    subdirs = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []
    return subdirs[0] if len(subdirs) == 1 else base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Make a 20-60s promo from selected microdrama episodes.")
    ap.add_argument("--show", type=Path, default=None, help="folder of episode .txt scripts (default: input/<only folder>)")
    ap.add_argument("--episodes", nargs="+", help="e.g. 3 4 | 10-12 | 12@00:40-02:10")
    ap.add_argument("--list", action="store_true", help="list episodes with length and logline, then exit")
    ap.add_argument("--provider", default=None, help="override LLM_PROVIDER; 'baseline' = no AI")
    ap.add_argument("--model", default=None, help="override LLM_MODEL")
    ap.add_argument("--attempts", type=int, default=3, help="max AI tries (first try + repairs)")
    ap.add_argument("--no-camera", action="store_true", help="leave CAMERA shot descriptions out of the prompt (fewer tokens)")
    ap.add_argument("--min", type=float, default=20.0, help="min promo length, seconds")
    ap.add_argument("--max", type=float, default=45.0, help="max promo length, seconds (brief allows 60)")
    ap.add_argument("--out", type=Path, default=Path("outputs/promos"))
    args = ap.parse_args(argv)

    load_env()
    show = load_show(args.show or default_show_dir())

    if args.list:
        print(f"{show.title}: {len(show.episodes)} episodes")
        for n, ep in show.episodes.items():
            print(f"  {n:>3}  {fmt_ts(ep.duration)}  {ep.logline[:110]}")
        return 0
    if not args.episodes:
        ap.error("give --episodes (or --list)")

    selection = parse_selection(args.episodes)
    llm = None
    if (args.provider or "").lower() != "baseline":
        try:
            llm = LLM(provider=args.provider, model=args.model)
        except LLMError as e:
            print(f"Cannot start the model: {e}\nTip: use --provider baseline to run without AI.")
            return 2

    print(f"Making promo from {', '.join(s.label() for s in selection)} "
          f"using {'baseline (no AI)' if llm is None else llm.provider + ':' + llm.model} ...")
    try:
        promo = make_promo(show, selection, llm=llm, rules=Rules(min_total=args.min, max_total=args.max),
                           max_attempts=args.attempts, include_camera=not args.no_camera)
    except LLMError as e:
        print(f"Model call failed: {e}")
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    stem = promo["promo_id"].replace(":", "-").replace("@", "_at_")
    (args.out / f"{stem}.json").write_text(json.dumps(promo, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(to_markdown(promo), encoding="utf-8")

    print(f"\n{promo['status'].upper()}  {promo['total_seconds']}s  in {promo['attempts']} attempt(s)  -> {args.out / stem}.md")
    for s in promo["segments"]:
        print(f"  seg {s['order']}: Ep {s['episode']}  {s['start']} -> {s['end']}  ({s['seconds']}s)")
    print(f"  hook  : {promo['hook'].get('line')}")
    print(f"  cliff : {promo['cliffhanger'].get('line')}")
    for x in promo["checker"]["issues"]:
        print(f"  PROBLEM: {x}")
    return 0 if promo["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
