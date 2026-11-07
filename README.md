# Promo Finder: AI that finds social-media promos in microdrama screenplays

Kuku AI Generalist assignment, **Problem 1 (Find the Promos)**.

Give it a folder of episode screenplays. It returns ranked **20-45 second promos**, each with:
- a cut list (episode + start + end for every segment)
- a hook in the first 3 seconds
- a cliffhanger at the end
- no line of dialogue cut in half

It makes two kinds:
1. **Scene promos** from runs of 1-3 consecutive episodes, searched across the whole show
2. **Theme montages** joining the best moments of the top episodes

Every promo is then checked by code, tested on a "cold viewer" AI that has never seen the show, judged against the others, and ranked.

**Run it in Colab:** open `Promo_pipeline.ipynb` →
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deThorer/kuku-promo-finder/blob/main/Promo_pipeline.ipynb)
Upload a zip of the scripts, add a free Gemini key, and run the cells top to bottom.

> The input scripts and outputs are **not** in this repo: they are Kuku's confidential content (`input/` and `outputs/` are git-ignored).

## Make a single promo (`make_promo.py`)

You give it episode number(s), and it gives you one ready-to-cut promo:
- a **cut list**: episode, start and end for every segment
- the **hook** (inside the first 3 s)
- the **cliffhanger** (the last thing shown)
- a note on what a new viewer understands
- why it works
- a transcript of exactly what the viewer sees and hears

It reads the screenplays only. There is no video in the input pack, so the output is the cut list an editor (or ffmpeg) would use, as the brief asks.

### Run it locally

```bash
cp .env.example .env            # add your free Gemini key + model name
# put the show's folder of episode .txt scripts in input/, e.g. input/<Show Name>/

python make_promo.py --list                         # episodes, lengths, loglines
python make_promo.py --episodes 1                   # one episode
python make_promo.py --episodes 3 4                 # consecutive episodes
python make_promo.py --episodes 10-12
python make_promo.py --episodes "12@00:40-02:10"    # only part of an episode
python make_promo.py --episodes 1 --provider baseline   # no AI, no key (rule-based)
python tests/test_offline.py                        # tests, no key needed
```

You need Python 3.10+. There is nothing to install, because it uses only the standard library.

Results are written to:
- `outputs/promos/<id>.md`: readable
- `outputs/promos/<id>.json`: full detail, including every attempt
- `outputs/llm_logs/`: every prompt and raw model response


## Full pipeline: two kinds of promos

1. **Scene promos**: runs of 1, 2 and 3 consecutive episodes, searched across all 50 episodes. By default the best 7 singles, 5 pairs and 3 triples (15 promos).
2. **Theme montages**: the best moments of the **top 15 episodes**, joined across episodes by theme. 6 by default.

The easiest way to run it is `Promo_pipeline.ipynb` in Google Colab.

```bash
python run_pipeline.py all                  # everything, about 70-110 AI calls in total
# or one stage at a time (each stage saves its output; reruns resume where they stopped):
python run_pipeline.py bible                # 1 call: characters, themes, spoilers  -> outputs/show_bible.json
python run_pipeline.py moments --batch 3    # ~17 calls (50 with --batch 1): 5 best moments per episode -> outputs/moment_bank.json/.md
python run_pipeline.py rank                 # 0 calls: top 20 episodes + 147 windows -> outputs/rankings.md
python run_pipeline.py scenes --per-run 7,5,3   # 15-45 calls: 7 singles, 5 pairs, 3 triples -> outputs/promos/scene_*.md
python run_pipeline.py montages --count 6 --top 15   # 7-19 calls: theme montages from the top 15 episodes -> outputs/promos/montage_*.md
python run_pipeline.py recheck              # 0 calls: re-check failed promos (code fixes: extend short segments, trim small overruns)
python run_pipeline.py judge                # ~4 calls: cold-viewer test + comparative judge -> outputs/final_ranking.md
python run_pipeline.py all --provider baseline   # whole thing with rules only, no AI -> outputs_baseline/
```

| Stage | Who | What |
|---|---|---|
| Show bible | AI (1 call) | Characters and their speaker labels, 6-10 recurring themes, and big reveals (the spoiler list) |
| Moment bank | AI (1 call per episode) + code | The 5 strongest moments per episode: boundaries, peak, type, themes, characters, summary, open question, hook / cliffhanger / clarity / intensity scores (1-5), spoiler level. Code checks the IDs, snaps cuts to whole lines, adds timestamps and drops overlaps |
| Moment score | code | 0.35 hook + 0.20 cliffhanger + 0.25 clarity + 0.20 intensity - spoiler penalty |
| Episode ranking | code | Episode score = mean of its 3 best moments. The top 15 feed the montages |
| Scene windows | code | Every run of 1, 2 and 3 consecutive episodes (147) is scored by its best hook moment, then a cliffhanger moment. In 2-3 episode runs the hook must be in the first episode and the cliffhanger in the last. Ties (AI scores cluster at 5/5) are broken by the moments' overall scores. Picks are made per category (default 7 singles, 5 pairs, 3 triples), with no shared hook or cliffhanger and at most 3 per theme |
| Scene promos | AI + checker | The promo maker on each picked run |
| Montage pools | code | Moments from the top 15 episodes grouped by theme (spoiler-high moments excluded; a theme needs 3+ moments from 2+ episodes) |
| Montage plans | AI (1 call) + code | The AI joins 3-4 moments per theme: hook first (a cold open from a later episode is allowed), the rest in story order, cliffhanger last. Code rejects bad plans (wrong order, unknown IDs, one episode only, spoilers, reuse > 2, duplicates) and tops up with rule-based plans |
| Montage promos | AI + checker | Each plan is cut to 20-45 s by the same checker as scene promos |
| Recheck | code | Failed promos: a segment under 2 s is extended with the next whole lines; a promo up to 15 s too long gets a chunk cut from its middle at line boundaries (hook and cliffhanger kept). Bigger problems stay as failures |
| Cold-viewer test | AI, no show context | Reads each promo's transcript and says who and what it understood, what it wants to know, and what confused it |
| Judge | AI (1 call for all promos) | Scores every promo against the others, so scores spread out: hook, cliffhanger, clarity (from the cold viewer), spoiler risk, duplicates |
| Final ranking | code | 0.35 hook + 0.30 cliffhanger + 0.25 clarity + 0.10 keep-watching, minus penalties for spoilers, final-act promos (-1), failed rules (-5) and duplicates (-1.5). A duplicate means 40%+ shared footage, or the judge says so AND the promos share an episode. The judge alone once flagged two unrelated scenes as duplicates |

The full run makes about 70-110 AI calls. A full promo for all 147 windows would take 150-450 calls, so windows are scored in code from the moment bank, and only the best are made into promos. Use `--count` to make more.

All promos are listed in `outputs/promo_index.md`.

## How the promo maker works

| Step | Who | What |
|---|---|---|
| 1. Parse | code | Each script becomes numbered, timed events, e.g. `E03.017 [01:12.40-01:14.90] HEROINE (Furious): ...` |
| 2. Context | code | Episode 1's logline plus the 3 episodes before the selection, so the AI knows the story |
| 3. Pick | **AI** | Chooses 1-4 segments, the hook and the cliffhanger, **by event ID only** |
| 4. Check | code | Converts IDs to real timestamps and enforces the rules (below) |
| 5. Repair | **AI** | If a rule is broken, the checker's measurements go back to the AI (up to 3 tries) |
| 6. Output | code | Cut list, hook, cliffhanger, why it works, and a transcript in promo time |

The AI never writes a timestamp. It can only point at events that exist, and invented or out-of-range IDs are rejected. That's how the tool avoids made-up moments.

### Rules the checker enforces

| Rule | When broken |
|---|---|
| Total length 20-45 s (brief allows 60; `--max 60` to change) | sent back to the AI |
| Hook starts within the first 3 s | auto-trimmed if the hook is in segment 1, otherwise sent back |
| Promo ends on the cliffhanger (≤3 s of footage after it) | auto-trimmed or extended |
| No line of dialogue is cut in half | cut auto-moved to the edge of the line |
| Two segments of the same episode less than 3 s apart | auto-joined into one (no jump cuts) |
| Segments in story order, no overlap, each ≥2 s, at most 4 | sent back |
| A segment can't run across an episode boundary | auto-split into two segments |

Every automatic fix is logged in the output. If the rules still fail after 3 tries, the status is `failed_checks` and the problems are listed.

### What code can't check
Code can't tell whether the hook is actually gripping, whether the ending is really unresolved, or whether a stranger would follow it. Those judgements belong to the AI and the human editor. Part 2 adds a cold-viewer test and a judge for them.

## Swapping the model
Change `LLM_PROVIDER` and `LLM_MODEL` in `.env`. Supported providers:
- Gemini
- Groq
- OpenRouter
- OpenAI
- Anthropic
- Ollama (local)
- any OpenAI-compatible URL

All prompts are in `promo/prompts.py`.

## Size and cost
A prompt is about 3.5k tokens for 1 episode and about 10k for 4 episodes (about half with `--no-camera`). One promo takes 1-3 calls. That fits easily in free tiers.

## Files
```
make_promo.py          command line
promo/parser.py        script -> events (code)
promo/llm.py           one interface for every model provider
promo/prompts.py       the promo editor prompt + repair prompt
promo/checker.py       hard rules, auto-fixes, measurements (code)
promo/promo_maker.py   the loop: AI pick -> check -> repair -> output
promo/baseline.py      rule-based promo with no AI (offline testing + comparison)
promo/bible.py         show bible: characters, themes, spoilers
promo/moments.py       moment bank: 5 best moments per episode, validated and scored
promo/ranking.py       episode ranking, scene-window scoring, montage pools (code only)
promo/montage.py       montage planner + builder
promo/judge.py         cold-viewer test, comparative judge, final ranking
run_pipeline.py        the full pipeline, stage by stage
Promo_pipeline.ipynb   Google Colab notebook that runs everything in order
tests/test_offline.py  41 checks, uses a fake model
```


## Confidentiality
The input scripts and outputs (which quote them) are git-ignored. Share outputs through a private link.

## Change log
- v1: first version (limit 60 s).
- v2 (prompt): hook/cliffhanger must point at dialogue or action events; aim 30-45 s. Result: the hook improved, but the model ignored the length target (53-57 s).
- v3 (code): hard 45 s limit in the checker plus auto-joining segments less than 3 s apart. A rule the model must obey goes in code, not just in the prompt.

## Free-tier request limits
Some free models allow very few requests per day (gemini-2.5-flash allowed 20/day on our key). To cope:
- `--batch 5` on the moments stage sends 5 episodes per request, so 50 episodes take 10 requests (about 11k tokens each).
- Quotas are per model, so stages can use different models: `--provider` / `--model` work on every command.
- Every stage saves its progress, so a quota stop just means you rerun the same command later.
- In batch mode a model may return too few moments per episode. Any episode with fewer than 3 gets one follow-up call, and `--redo-thin` repairs an existing bank. `--batch 3` is more reliable than 5.
- `rank --rank-k 2` scores episodes on their best 2 moments, a zero-request fallback when episodes have different numbers of moments.
- `scenes` and `montages` skip promos that are already made, and montage plans are saved and reused, so rerunning after a quota stop costs nothing extra. `--redo` remakes them.
- When a model's daily quota is used up (HTTP 429, "per day"), the tool switches straight to the next model in `LLM_FALLBACK_MODELS` instead of retrying, and records the model actually used in each promo.

## Ranking rules (code)
- **Final act**: moments from the last 30% of the show (episode 36+ of 50) lose 0.75 points (`--final-act-from`, `--final-act-penalty`). Moments in the bible's big-reveal episodes count as at least a low spoiler. The AI flagged only 14 of 249 moments as spoilers, so code backs it up.
- **One promo per episode**: an episode appears in at most one scene promo (`--episode-reuse`). Otherwise a single episode and a pair/triple containing it would both cut the same best scene.
- **Triples are picked first**, then pairs, then singles. Windows scoring below 4.0 are skipped (`--min-window-score`).
- A model that stops answering (read timeout, default 90 s via `LLM_TIMEOUT`) is retried, then treated like an overloaded one and replaced by the next fallback model.
