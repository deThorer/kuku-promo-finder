"""Full pipeline: show bible -> moment bank -> rankings -> two kinds of promos.

  1. SCENE PROMOS   : runs of 1, 2 and 3 consecutive episodes, searched over ALL episodes
                      (default best 7 singles + 5 pairs + 3 triples = 15 promos)
  2. THEME MONTAGES : the best moments of the TOP 15 episodes, joined across episodes by theme
                      (default 6 promos)

Each stage saves its output, so you can run stages one at a time and resume after a quota error.

  python run_pipeline.py bible                         # 1 AI call: characters, themes, spoilers
  python run_pipeline.py moments --batch 3             # ~17 AI calls: 5 best moments per episode (resumable)
  python run_pipeline.py rank                          # 0 AI calls: episode ranking + all 147 runs scored
  python run_pipeline.py scenes --per-run 7,5,3        # scene promos (1 call each, +repairs)
  python run_pipeline.py montages --count 6 --top 15   # theme montages (1 planning call + 1 per montage, +repairs)
  python run_pipeline.py all                           # everything, in order

  --provider baseline   runs every stage with rules only (no AI, no key) into outputs_baseline/
  --model NAME          use a different model for one stage (each model has its own free quota)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from make_promo import default_show_dir
from promo.bible import bible_block, build_bible
from promo.judge import cold_viewer, final_ranking, final_ranking_markdown, judge
from promo.llm import LLM, LLMError, load_env
from promo.moments import all_moments, bank_markdown, extract_moments, extract_moments_batch, load_bank, save_bank
from promo.montage import build_montage, plan_montages
from promo.parser import load_show
from promo.checker import Rules
from promo.promo_maker import Selection, make_promo, parse_selection, recheck_promo, to_markdown
from promo.ranking import (apply_story_policy, episode_ranking, montage_pools, pick_windows, ranking_markdown,
                           scene_windows)


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40]


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.show = load_show(args.show or default_show_dir())
        self.llm = None if args.provider == "baseline" else LLM(provider=args.provider, model=args.model)
        default_out = "outputs_baseline" if args.provider == "baseline" else "outputs"
        self.out = Path(args.out or default_out)
        self.promo_dir = self.out / "promos"

    # ------------------------------------------------------------------ stages
    def bible(self, force: bool = False) -> dict:
        path = self.out / "show_bible.json"
        if path.exists() and not force:
            return jload(path)
        print("[bible] writing the show bible ...")
        bible = build_bible(self.show, self.llm)
        jdump(bible, path)
        print(f"[bible] {len(bible['characters'])} characters, {len(bible['themes'])} themes, "
              f"{len(bible['big_reveals'])} spoilers -> {path}")
        return bible

    def moments(self) -> dict:
        bible = self.bible()
        path = self.out / "moment_bank.json"
        bank = load_bank(path)
        bank["provider"] = "baseline" if self.llm is None else f"{self.llm.provider}:{self.llm.model}"
        eps = [s.episode for s in parse_selection([self.args.episodes])] if self.args.episodes else list(self.show.episodes)
        def thin(n: int) -> bool:
            return len(bank["episodes"].get(str(n), {}).get("moments", [])) < self.args.min_moments
        todo = [n for n in eps if self.args.redo or str(n) not in bank["episodes"] or (self.args.redo_thin and thin(n))]
        print(f"[moments] {len(eps) - len(todo)} episodes already in the bank, {len(todo)} to do"
              + (f" (incl. episodes with < {self.args.min_moments} moments)" if self.args.redo_thin else ""))
        size = max(1, self.args.batch) if self.llm is not None else 1
        if size > 1:
            print(f"[moments] batching {size} episodes per call -> about {-(-len(todo) // size)} requests")
        for i in range(0, len(todo), size):
            group = todo[i:i + size]
            try:
                if size == 1:
                    recs = [extract_moments(self.show, group[0], bible, self.llm, k=5, include_camera=not self.args.no_camera)]
                else:
                    recs = extract_moments_batch(self.show, group, bible, self.llm, k=5, include_camera=not self.args.no_camera,
                                                 min_moments=self.args.min_moments)
            except LLMError as e:
                print(f"[moments] stopped at episode {group[0]}: {e}\n          progress is saved - run the same command again to resume.")
                save_bank(bank, path)
                raise SystemExit(3)
            for rec in recs:
                n = rec["episode"]
                if rec["status"] == "missing":
                    print(f"[moments] E{n:02d}: missing from the answer - will be redone on the next run")
                    continue
                bank["episodes"][str(n)] = rec
                best = rec["moments"][0] if rec["moments"] else None
                print(f"[moments] E{n:02d}: {len(rec['moments'])} moments"
                      + (f", best {best['id']} ({best['score']}): {best['summary'][:70]}" if best else "")
                      + (f"  [{len(rec['notes'])} note(s)]" if rec["notes"] else ""))
            save_bank(bank, path)  # after every call, so a crash or quota error loses nothing
        (self.out / "moment_bank.md").write_text(bank_markdown(bank), encoding="utf-8")
        return bank

    def ranked_bank(self) -> dict:
        """The moment bank with the spoiler / final-act policy applied (in memory only)."""
        bank = load_bank(self.out / "moment_bank.json")
        if not bank["episodes"]:
            bank = self.moments()
        last = max(self.show.episodes)
        fa = self.args.final_act_from if self.args.final_act_from is not None else int(0.7 * last) + 1
        self.final_act_from = fa
        return apply_story_policy(bank, self.bible(), fa, self.args.final_act_penalty)

    def rank(self) -> tuple[list[dict], list[dict]]:
        bank = self.ranked_bank()
        if self.final_act_from:
            print(f"[rank] final act = episode {self.final_act_from}+ (penalty {self.args.final_act_penalty}); "
                  f"big-reveal episodes count as spoilers")
        eps = episode_ranking(bank, top_k=self.args.rank_k)
        windows = scene_windows(bank, max_run=self.args.max_run)
        picked = pick_windows(windows, self.args.per_run_counts, max_per_theme=self.args.max_per_theme,
                              episode_reuse=self.args.episode_reuse, min_score=self.args.min_window_score)
        short = {r: n - sum(1 for w in picked if w["run"] == r) for r, n in self.args.per_run_counts.items()}
        short = {r: n for r, n in short.items() if n > 0}
        if short:
            print(f"[rank] not enough strong windows (score >= {self.args.min_window_score}) for: "
                  + ", ".join(f"{n} of the run-{r} promos" for r, n in short.items()))
        jdump({"episodes": eps, "top_episodes": [r["episode"] for r in eps[:self.args.top]]}, self.out / "episode_ranking.json")
        jdump({"all_windows": windows, "picked": picked}, self.out / "scene_windows.json")
        (self.out / "rankings.md").write_text(ranking_markdown(eps, windows, picked, self.args.top), encoding="utf-8")
        print(f"[rank] top {self.args.top} episodes: {[r['episode'] for r in eps[:self.args.top]]}")
        print(f"[rank] {len(windows)} scene windows scored (runs of 1-{self.args.max_run}); picked {len(picked)} "
              f"(per run length {self.args.per_run_counts}, max {self.args.max_per_theme} per theme): "
              f"{['-'.join(map(str, w['episodes'])) for w in picked]}")
        return eps, picked

    def scenes(self) -> list[dict]:
        _, picked = self.rank()
        wanted = {f"scene_{'-'.join(f'E{e:02d}' for e in w['episodes'])}": w for w in picked}
        self._retire_stale("scene_", set(wanted))
        results = []
        for i, (pid, w) in enumerate(wanted.items(), 1):
            done = self.promo_dir / f"{pid}.json"
            if done.exists() and not self.args.redo and not (self.args.redo_failed and jload(done)["status"] != "ok"):
                print(f"[scenes] {i}/{len(wanted)} {pid}: already made, skipping (use --redo to remake)")
                results.append(jload(done))
                continue
            print(f"[scenes] {i}/{len(wanted)} episodes {w['episodes']} ...")
            try:
                p = make_promo(self.show, [Selection(e) for e in w["episodes"]], llm=self.llm,
                               include_camera=not self.args.no_camera, promo_id=pid)
            except LLMError as e:
                print(f"[scenes] stopped: {e}\n         finished promos are saved - run the same command again to continue.")
                break
            p["window"] = w
            self._save_promo(p, pid)
            results.append(p)
        self.index()
        return results

    def montages(self) -> list[dict]:
        eps, _ = self.rank()
        bank = self.ranked_bank()
        bible = self.bible()
        top = [r["episode"] for r in eps[:self.args.top]]
        pools = montage_pools(bank, top)
        print(f"[montages] {len(pools)} themes have enough moments in the top {len(top)} episodes: {list(pools)}")
        text = bible_block(bible, with_themes=False)
        moments = {m["id"]: m for m in all_moments(bank)}
        plans_path = self.out / "montage_plans.json"

        plans = None
        if plans_path.exists() and not self.args.redo:   # reuse saved plans: no planner call on reruns
            saved = jload(plans_path)
            if (saved.get("top_episodes") == top and saved.get("count") == self.args.montage_count
                    and saved.get("plans") and all(x in moments for p in saved["plans"] for x in p["moments"])):
                plans = [{**p, "moments": [moments[x] for x in p["moments"]]} for p in saved["plans"]]
                print(f"[montages] reusing {len(plans)} saved plans (use --redo to plan again)")
        if plans is None:
            try:
                plans, notes = plan_montages(list(moments.values()), pools, bible, bible_block(bible), self.llm,
                                             self.args.montage_count)
            except LLMError as e:
                print(f"[montages] planner stopped: {e}\n           run the same command again later.")
                raise SystemExit(3)
            for n in notes:
                print(f"[montages] {n}")
            jdump({"top_episodes": top, "count": self.args.montage_count,
                   "pools": {t: [m["id"] for m in ms] for t, ms in pools.items()},
                   "plans": [{**p, "moments": [m["id"] for m in p["moments"]]} for p in plans], "notes": notes},
                  plans_path)
            if self.args.redo:
                self._retire_stale("montage_", set())

        results = []
        for i, plan in enumerate(plans, 1):
            pid = f"montage_{i:02d}_{slug(plan['theme'])}"
            done = self.promo_dir / f"{pid}.json"
            if done.exists() and not self.args.redo and not (self.args.redo_failed and jload(done)["status"] != "ok"):
                print(f"[montages] {i}/{len(plans)} {pid}: already made, skipping (use --redo to remake)")
                results.append(jload(done))
                continue
            print(f"[montages] {i}/{len(plans)} {plan['theme']}: {[m['id'] for m in plan['moments']]} ...")
            try:
                p = build_montage(self.show, plan, llm=self.llm, bible_text=text, promo_id=pid)
            except LLMError as e:
                print(f"[montages] stopped: {e}\n           finished promos and plans are saved - run the same command again to continue.")
                break
            self._save_promo(p, pid)
            results.append(p)
        self.index()
        return results

    def recheck(self) -> None:
        """Re-run the current checker (incl. code fixes) on saved promos that failed. 0 AI calls."""
        moments = {m["id"]: m for m in all_moments(load_bank(self.out / "moment_bank.json"))}
        fixed = 0
        for f in sorted(self.promo_dir.glob("*.json")):
            old = jload(f)
            if "type" not in old or (old["status"] == "ok" and not self.args.redo):
                continue
            if old["type"] == "montage":
                sel = []
                for mid in sorted(old["plan"]["moments"], key=lambda x: (moments[x]["episode"], moments[x]["start"])):
                    m = moments[mid]
                    s0, e0 = max(0.0, m["start"] - 1.5), m["end"] + 1.5
                    if sel and sel[-1].episode == m["episode"] and s0 <= (sel[-1].end or 0):
                        sel[-1].end = max(sel[-1].end or 0, e0)
                    else:
                        sel.append(Selection(m["episode"], s0, e0))
                rules = Rules(allow_cold_open=True)
            else:
                sel = [Selection(e) for e in old["window"]["episodes"]]
                rules = Rules()
            print(f"[recheck] {old['promo_id']} ({old['status']}, {old['total_seconds']}s) ...")
            new = recheck_promo(self.show, old, sel, rules)
            self._save_promo(new, f.stem)
            fixed += new["status"] == "ok" and old["status"] != "ok"
        print(f"[recheck] {fixed} promo(s) now pass the checks")
        self.index()

    def judge(self) -> None:
        """Cold-viewer test + one comparative judge call + final ranking (about 4-5 AI calls for 21 promos)."""
        promos = [jload(f) for f in sorted(self.promo_dir.glob("*.json"))]
        promos = [p for p in promos if "type" in p]
        if not promos:
            print("[judge] no promos yet - run scenes / montages first")
            return
        bible = self.bible()
        cold_path, judge_path = self.out / "judge_cold_viewer.json", self.out / "judge_scores.json"
        ids = sorted(p["promo_id"] for p in promos)
        cold = jload(cold_path) if cold_path.exists() and not self.args.redo else {}
        cold = {k: v for k, v in cold.items() if k in ids}
        todo = [p for p in promos if p["promo_id"] not in cold]
        try:
            if todo:
                print(f"[judge] cold-viewer test on {len(todo)} promo(s) (no show context) ...")
                cold.update(cold_viewer(todo, self.llm))
                jdump(cold, cold_path)
            saved = jload(judge_path) if judge_path.exists() and not self.args.redo else {}
            if sorted(saved.get("ids", [])) == ids and not todo:
                scores = saved["scores"]
                print("[judge] reusing saved judge scores (use --redo to judge again)")
            else:
                print(f"[judge] comparing all {len(promos)} promos in one call ...")
                scores = judge(promos, cold, bible.get("premise", ""), self.llm)
                jdump({"ids": ids, "model": self.llm.model if self.llm else "baseline", "scores": scores}, judge_path)
        except LLMError as e:
            print(f"[judge] stopped: {e}\n        progress is saved - run the same command again.")
            raise SystemExit(3)
        last = max(self.show.episodes)
        fa = self.args.final_act_from if self.args.final_act_from is not None else int(0.7 * last) + 1
        ranked = final_ranking(promos, cold, scores, final_act_from=fa)
        model = (jload(judge_path).get("model") if judge_path.exists() else "baseline")
        (self.out / "final_ranking.md").write_text(final_ranking_markdown(ranked, cold, scores, model), encoding="utf-8")
        jdump([{"rank": r["rank"], "promo_id": r["promo"]["promo_id"], "score": r["score"], "flags": r["flags"],
                **scores[r["promo"]["promo_id"]], "cold_viewer": cold.get(r["promo"]["promo_id"], {})} for r in ranked],
              self.out / "final_ranking.json")
        print(f"[judge] final ranking -> {self.out / 'final_ranking.md'}")
        for r in ranked[:5]:
            print(f"   {r['rank']}. {r['promo']['promo_id'].split('__')[0]:28} score {r['score']}  {'; '.join(r['flags'])}")

    # ------------------------------------------------------------------ helpers
    def _save_promo(self, p: dict, pid: str) -> None:
        jdump(p, self.promo_dir / f"{pid}.json")
        (self.promo_dir / f"{pid}.md").write_text(to_markdown(p), encoding="utf-8")
        print(f"          {p['status'].upper()} {p['total_seconds']}s, {p['attempts']} attempt(s)"
              + (f"  PROBLEMS: {p['checker']['issues']}" if p["checker"]["issues"] else ""))

    def _retire_stale(self, prefix: str, keep: set[str]) -> None:
        """Move promos from an older ranking/plan to promos/_old/ so the index lists only current ones."""
        if not self.promo_dir.exists():
            return
        old = self.promo_dir / "_old"
        for f in self.promo_dir.glob(f"{prefix}*"):
            if f.stem not in keep:
                old.mkdir(parents=True, exist_ok=True)
                f.replace(old / f.name)
                print(f"[promos] moved outdated {f.name} to promos/_old/")

    def index(self) -> None:
        rows = []
        for f in sorted(self.promo_dir.glob("*.json")):
            p = jload(f)
            if "type" not in p:
                continue
            rows.append(f"| {p['promo_id']} | {p['type']} | {p['status']} | {p['total_seconds']} | {len(p['segments'])} "
                        f"| {p.get('model_confidence')} | {(p['hook'].get('line') or '')[:70].replace('|', '/')} "
                        f"| {(p['cliffhanger'].get('open_question') or '')[:70].replace('|', '/')} |")
        md = ["# All promos", "", "| Promo | Type | Status | Sec | Segs | Model conf. | Hook | Open question |",
              "|---|---|---|---|---|---|---|---|"] + rows
        (self.out / "promo_index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[index] {len(rows)} promos listed in {self.out / 'promo_index.md'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Promo pipeline: bible -> moments -> rank -> scenes + montages")
    ap.add_argument("stage", choices=["bible", "moments", "rank", "scenes", "montages", "recheck", "judge", "all"])
    ap.add_argument("--show", type=Path)
    ap.add_argument("--provider", default=None, help="override LLM_PROVIDER; 'baseline' = no AI")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="output folder (default outputs/, or outputs_baseline/)")
    ap.add_argument("--episodes", default=None, help="moments stage only: e.g. 1-10")
    ap.add_argument("--redo-failed", action="store_true", help="scenes/montages: remake only promos that failed the checks")
    ap.add_argument("--redo", action="store_true", help="redo work that is already saved (bible, moments, promos, montage plans)")
    ap.add_argument("--top", type=int, default=15, help="top N episodes feed the montages")
    ap.add_argument("--max-run", type=int, default=3, help="scene promos: longest run of consecutive episodes")
    ap.add_argument("--count", type=int, default=None, help="how many scene promos / montages to make")
    ap.add_argument("--no-camera", action="store_true", help="leave CAMERA lines out of prompts (fewer tokens)")
    ap.add_argument("--per-run", default="7,5,3", help="scene promos per run length: singles,pairs,triples")
    ap.add_argument("--max-per-theme", type=int, default=3, help="scene picks: at most this many with the same hook theme")
    ap.add_argument("--min-window-score", type=float, default=None,
                    help="scene picks: skip windows scoring below this (default 4.0; 0 for --provider baseline)")
    ap.add_argument("--episode-reuse", type=int, default=1,
                    help="scene picks: max promos one episode may appear in (0 = no limit)")
    ap.add_argument("--final-act-from", type=int, default=None,
                    help="episodes from here on are the final act and get penalised (default: 70%% of the show; 0 = off)")
    ap.add_argument("--final-act-penalty", type=float, default=0.75, help="score penalty for final-act moments")
    ap.add_argument("--rank-k", type=int, default=3, help="rank: episode score = mean of its best K moments")
    ap.add_argument("--min-moments", type=int, default=3, help="moments: fewer than this per episode counts as too few")
    ap.add_argument("--redo-thin", action="store_true", help="moments: redo episodes in the bank that have too few moments")
    ap.add_argument("--batch", type=int, default=1, help="moments stage: episodes per AI call (e.g. 5 on low free-tier request limits)")
    args = ap.parse_args(argv)
    counts = [int(x) for x in args.per_run.split(",") if x.strip()]
    args.per_run_counts = {i + 1: n for i, n in enumerate(counts) if i + 1 <= args.max_run}
    args.montage_count = args.count if (args.count and args.stage == "montages") else 6

    load_env()
    if (args.provider or "").lower() == "baseline":
        args.provider = "baseline"
    if args.min_window_score is None:  # rule-based scores sit around 2-3, AI scores around 4-5
        args.min_window_score = 0.0 if args.provider == "baseline" else 4.0
    try:
        pipe = Pipeline(args)
    except LLMError as e:
        print(f"Cannot start the model: {e}\nTip: --provider baseline runs without AI.")
        return 2
    if args.stage == "bible":
        pipe.bible(force=args.redo)
    elif args.stage == "moments":
        pipe.moments()
    elif args.stage == "rank":
        pipe.rank()
    elif args.stage == "scenes":
        pipe.scenes()
    elif args.stage == "montages":
        pipe.montages()
    elif args.stage == "recheck":
        pipe.recheck()
    elif args.stage == "judge":
        pipe.judge()
    else:
        args.redo_thin = True
        pipe.bible()
        pipe.moments()
        pipe.scenes()
        pipe.montages()
        pipe.recheck()
        pipe.judge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
