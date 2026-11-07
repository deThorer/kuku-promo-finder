/* Promo Finder core, a faithful JavaScript port of the Python pipeline's promo maker
   (parser.py, checker.py, baseline.py, prompts.py, promo_maker.py). Runs entirely in the browser:
   scripts are read from the user's own disk and never uploaded anywhere except, if the user chooses,
   to the AI provider they give a key for. Works in Node too (module.exports) for testing. */
(function (root) {
  "use strict";
  const EPS = 0.05;

  // ------------------------------------------------------------------ parser
  const TS = "(\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\.\\d+)?)";
  const TIME_RE = new RegExp("^Start:\\s*\\(\\s*" + TS + "\\s*\\)\\s*\\|\\s*End:\\s*\\(\\s*" + TS + "\\s*\\)");
  const LABEL_RE = /^\*\*(.+?):\*\*\s*$/;
  const SCENE_RE = /^SCENE\s+(\d+)\s*$/i;
  const HEADING_RE = /^Scene heading:\s*(.*)$/i;
  const STRUCT_RE = /^(Scene (start|end):|Actions and dialogue:|---)/i;
  const HEADER_RE = /^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$/i;
  const FILE_EP_RE = /episode-(\d+)/i;
  const NON_DIALOGUE = { "CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SOUND", "SOUND": "SOUND", "SFX": "SOUND", "MUSIC": "SOUND" };

  function parseTs(text) {
    const parts = text.trim().split(":");
    const sec = parseFloat(parts[parts.length - 1]);
    const mins = parseInt(parts[parts.length - 2], 10);
    const hrs = parts.length === 3 ? parseInt(parts[0], 10) : 0;
    return hrs * 3600 + mins * 60 + sec;
  }
  function fmtTs(seconds) {
    const s0 = Math.max(0, Math.round(seconds * 100) / 100);
    const m = Math.floor(s0 / 60);
    const s = Math.round((s0 - m * 60) * 100) / 100;
    return String(m).padStart(2, "0") + ":" + s.toFixed(2).padStart(5, "0");
  }
  const pad2 = n => String(n).padStart(2, "0");
  const pad3 = n => String(n).padStart(3, "0");

  function labelOf(e) { return e.kind === "DIALOGUE" ? (e.emotion ? `${e.speaker} (${e.emotion})` : String(e.speaker)) : e.kind; }
  function lineOf(e) { return `${e.id} [${fmtTs(e.start)}-${fmtTs(e.end)}] ${labelOf(e)}: ${e.text}`; }

  function parseEpisode(text, filename) {
    const lines = text.split(/\r?\n/);
    let number = null, headerKey = null, inScript = false;
    const headers = { LOGLINE: "", CONTINUITY: "" };
    let scene = 0, heading = "", pending = null, current = null;
    const raw = [];
    const flush = () => { if (current) { current.text = current.text.join(" ").trim(); raw.push(current); current = null; } };
    for (const r of lines) {
      const line = r.trim();
      if (!inScript) {
        if (line.toUpperCase().startsWith("--- SCRIPT START")) { inScript = true; continue; }
        const m = line.match(HEADER_RE);
        if (m) {
          const key = m[1].toUpperCase(), val = m[2].trim();
          if (key === "EPISODE NUMBER") { const n = val.match(/\d+/); number = n ? parseInt(n[0], 10) : null; headerKey = null; }
          else { headerKey = key.includes("LOGLINE") ? "LOGLINE" : "CONTINUITY"; headers[headerKey] = val; }
        } else if (line && headerKey) { headers[headerKey] += " " + line; }
        continue;
      }
      let m;
      if ((m = line.match(SCENE_RE))) { flush(); scene = parseInt(m[1], 10); continue; }
      if ((m = line.match(HEADING_RE))) { flush(); heading = m[1].trim(); continue; }
      if (STRUCT_RE.test(line)) { flush(); continue; }
      if ((m = line.match(LABEL_RE))) { flush(); pending = m[1].trim(); continue; }
      if ((m = line.match(TIME_RE)) && pending !== null) {
        const label = pending; pending = null;
        let kind = NON_DIALOGUE[label.toUpperCase()] || null, speaker = null, emotion = null;
        if (!kind) {
          kind = "DIALOGUE";
          const groups = [...label.matchAll(/\(([^)]*)\)/g)].map(g => g[1].trim());
          speaker = label.split("(")[0].trim() || label;
          emotion = groups.length ? groups.join("; ") : null;
        }
        current = { kind, speaker, emotion, start: parseTs(m[1]), end: parseTs(m[2]), text: [], scene, heading };
        continue;
      }
      if (line && current) current.text.push(line);
    }
    flush();
    if (number === null) { const fm = (filename || "").match(FILE_EP_RE); number = fm ? parseInt(fm[1], 10) : 0; }
    const events = raw.map((r, i) => {
      let t = r.text;
      if (r.kind === "DIALOGUE") t = t.trim().replace(/^["“”]+|["“”]+$/g, "").trim();
      return { id: `E${pad2(number)}.${pad3(i + 1)}`, episode: number, index: i + 1, kind: r.kind, speaker: r.speaker,
               emotion: r.emotion, start: r.start, end: Math.max(r.end, r.start), text: t, scene: r.scene, heading: r.heading };
    });
    const duration = events.reduce((a, e) => Math.max(a, e.end), 0);
    return { number, logline: headers.LOGLINE.trim(), continuity: headers.CONTINUITY.trim(), events, duration, file: filename };
  }

  /** files: [{name, text}] -> show */
  function loadShow(files, title) {
    const episodes = {};
    for (const f of files.slice().sort((a, b) => a.name.localeCompare(b.name))) {
      if (!/\.txt$/i.test(f.name)) continue;
      const ep = parseEpisode(f.text, f.name.split("/").pop());
      if (ep.events.length) episodes[ep.number] = ep;
    }
    const nums = Object.keys(episodes).map(Number).sort((a, b) => a - b);
    return { title: title || "Show", episodes, numbers: nums };
  }

  // ------------------------------------------------------------------ selection
  function parseSelection(text) {
    const out = [];
    for (const part of String(text).split(/[,\s]+/).filter(Boolean)) {
      if (part.includes("@")) {
        const [ep, rng] = part.split("@");
        const [a, b] = rng.split("-");
        out.push({ episode: parseInt(ep, 10), start: a ? parseTs(a) : null, end: b ? parseTs(b) : null });
      } else if (part.includes("-")) {
        const [a, b] = part.split("-").map(x => parseInt(x, 10));
        for (let n = a; n <= b; n++) out.push({ episode: n, start: null, end: null });
      } else out.push({ episode: parseInt(part, 10), start: null, end: null });
    }
    return out.sort((x, y) => x.episode - y.episode);
  }
  function selLabel(s) {
    if (s.start === null && s.end === null) return `E${pad2(s.episode)}`;
    return `E${pad2(s.episode)}@${fmtTs(s.start || 0)}-${s.end !== null ? fmtTs(s.end) : "end"}`;
  }
  function selectedEvents(show, selection) {
    const out = [];
    for (const s of selection) {
      const ep = show.episodes[s.episode];
      if (!ep) throw new Error(`Episode ${s.episode} not found (have ${show.numbers[0]}-${show.numbers[show.numbers.length - 1]})`);
      for (const e of ep.events) {
        if (s.start !== null && e.end <= s.start) continue;
        if (s.end !== null && e.start >= s.end) continue;
        out.push(e);
      }
    }
    return out;
  }
  function storySoFar(show, firstEp, recent = 3) {
    const eps = show.numbers.filter(n => n < firstEp);
    if (!eps.length) return "";
    const keep = [...new Set([eps[0], ...eps.slice(-recent)])].sort((a, b) => a - b);
    const lines = [];
    keep.forEach((n, i) => { if (lines.length && n !== keep[i - 1] + 1) lines.push("..."); lines.push(`Ep ${n}: ${show.episodes[n].logline}`); });
    return lines.join("\n");
  }

  // ------------------------------------------------------------------ checker
  const DEFAULT_RULES = { minTotal: 20, maxTotal: 45, hookWindow: 3, cliffTail: 3, minSegment: 2, maxSegments: 4, mergeGap: 3 };
  const dur = s => s.end - s.start;
  const overlaps = (s, e) => e.episode === s.episode && e.start < s.end - EPS && e.end > s.start + EPS;
  const normId = x => String(x == null ? "" : x).trim().toUpperCase();
  const describe = s => `E${pad2(s.episode)} ${fmtTs(s.start)}-${fmtTs(s.end)} (${dur(s).toFixed(1)}s)`;

  function snapToLines(seg, epEvents, fixes, label) {
    let changed = true;
    while (changed) {
      changed = false;
      for (const e of epEvents) {
        if (e.kind !== "DIALOGUE") continue;
        if (e.start < seg.start - EPS && e.end > seg.start + EPS) {
          fixes.push(`${label}: start moved ${fmtTs(seg.start)} -> ${fmtTs(e.start)} so line ${e.id} is not cut.`); seg.start = e.start; changed = true;
        }
        if (e.start < seg.end - EPS && e.end > seg.end + EPS) {
          fixes.push(`${label}: end moved ${fmtTs(seg.end)} -> ${fmtTs(e.end)} so line ${e.id} is not cut.`); seg.end = e.end; changed = true;
        }
      }
    }
  }
  function mergeClose(segments, gap, fixes) {
    const merged = [];
    for (const seg of segments) {
      const prev = merged[merged.length - 1];
      if (prev && prev.episode === seg.episode && seg.start >= prev.start && seg.start - prev.end < gap) {
        const g = seg.start - prev.end;
        const what = g < 0 ? `overlapped by ${(-g).toFixed(1)}s` : `were only ${g.toFixed(1)}s apart`;
        fixes.push(`Joined two segments of episode ${seg.episode} that ${what} (${fmtTs(prev.end)} / ${fmtTs(seg.start)}); a cut that small looks like a jump.`);
        prev.end = Math.max(prev.end, seg.end); prev.lastEvent = seg.lastEvent;
        prev.purpose = [prev.purpose, seg.purpose].filter(Boolean).join(" + ");
      } else merged.push(seg);
    }
    return merged;
  }

  function checkPromo(promo, index, byEp, rules) {
    rules = Object.assign({}, DEFAULT_RULES, rules || {});
    const res = { segments: [], total: 0, hookOffset: null, cliffTail: null, fixes: [], issues: [] };
    const rawSegs = promo && promo.segments;
    if (!Array.isArray(rawSegs) || !rawSegs.length) { res.issues.push("No segments returned."); return finish(res); }
    rawSegs.forEach((s0, i0) => {
      const i = i0 + 1, s = s0 && typeof s0 === "object" ? s0 : {};
      const aId = normId(s.first_event || s.start_event), bId = normId(s.last_event || s.end_event);
      const a = index[aId], b = index[bId];
      if (!a || !b) {
        const bad = [[aId, a], [bId, b]].filter(x => !x[1]).map(x => x[0]);
        res.issues.push(`Segment ${i} uses event ID(s) not in the selected episodes: ${bad.join(", ") || "(empty)"}.`); return;
      }
      if (a.episode > b.episode || (a.episode === b.episode && a.start > b.start)) {
        res.issues.push(`Segment ${i}: first_event ${a.id} comes after last_event ${b.id}.`); return;
      }
      const purpose = String(s.purpose || "");
      if (a.episode === b.episode) res.segments.push({ episode: a.episode, start: a.start, end: Math.max(b.end, a.start), firstEvent: a.id, lastEvent: b.id, purpose });
      else {
        for (let ep = a.episode; ep <= b.episode; ep++) {
          const evs = (byEp[ep] || []).filter(e => index[e.id]);
          if (!evs.length) continue;
          const first = ep === a.episode ? a : evs.reduce((x, y) => (y.start < x.start ? y : x));
          const last = ep === b.episode ? b : evs.reduce((x, y) => (y.end > x.end ? y : x));
          res.segments.push({ episode: ep, start: first.start, end: last.end, firstEvent: first.id, lastEvent: last.id, purpose });
        }
        res.fixes.push(`Segment ${i} ran from episode ${a.episode} into ${b.episode}; split at the episode boundary.`);
      }
    });
    if (!res.segments.length) return finish(res);
    res.segments = mergeClose(res.segments, rules.mergeGap, res.fixes);

    const hookId = normId((promo.hook || {}).event), hook = index[hookId];
    let seg0 = res.segments[0];
    if (!hook) res.issues.push(`Hook event '${hookId}' is not an event in the selected episodes.`);
    else if (!overlaps(seg0, hook)) res.issues.push(`Hook ${hook.id} is not inside the first segment; the promo must open on the hook.`);
    else if (hook.start - seg0.start > rules.hookWindow) {
      res.fixes.push(`Hook ${hook.id} started ${(hook.start - seg0.start).toFixed(1)}s in; trimmed segment 1 to open on it.`); seg0.start = hook.start;
    }
    const cliffId = normId((promo.cliffhanger || {}).event), cliff = index[cliffId];
    let last = res.segments[res.segments.length - 1];
    if (!cliff) res.issues.push(`Cliffhanger event '${cliffId}' is not an event in the selected episodes.`);
    else if (!overlaps(last, cliff) && !(cliff.episode === last.episode && Math.abs(cliff.start - last.end) < EPS))
      res.issues.push(`Cliffhanger ${cliff.id} is not inside the last segment; the promo must end on it.`);
    else if (cliff.end > last.end + EPS) { res.fixes.push(`Extended the last segment to ${fmtTs(cliff.end)} to include the full cliffhanger ${cliff.id}.`); last.end = cliff.end; }
    else if (last.end - cliff.end > rules.cliffTail) { res.fixes.push(`Trimmed ${(last.end - cliff.end).toFixed(1)}s after cliffhanger ${cliff.id} so the promo ends on it.`); last.end = cliff.end; }

    res.segments.forEach((seg, i) => snapToLines(seg, byEp[seg.episode] || [], res.fixes, `Segment ${i + 1}`));
    res.segments = mergeClose(res.segments, rules.mergeGap, res.fixes);

    res.segments.forEach((seg, i) => {
      if (dur(seg) >= rules.minSegment) return;
      const evs = (byEp[seg.episode] || []).filter(e => index[e.id]).slice().sort((x, y) => x.start - y.start || x.index - y.index);
      const before = dur(seg);
      if (i < res.segments.length - 1) {
        const nxt = res.segments[i + 1];
        for (const e of evs) {
          if (dur(seg) >= rules.minSegment + 1) break;
          if (e.start >= seg.end - EPS && !(nxt.episode === seg.episode && e.end > nxt.start - rules.mergeGap)) seg.end = Math.max(seg.end, e.end);
        }
      } else {
        const prv = i ? res.segments[i - 1] : null;
        for (const e of evs.slice().reverse()) {
          if (dur(seg) >= rules.minSegment + 1) break;
          if (e.end <= seg.start + EPS && !(prv && prv.episode === seg.episode && e.start < prv.end + rules.mergeGap)) seg.start = Math.min(seg.start, e.start);
        }
      }
      if (dur(seg) > before) {
        snapToLines(seg, byEp[seg.episode] || [], res.fixes, `Segment ${i + 1}`);
        res.fixes.push(`Segment ${i + 1} was only ${before.toFixed(1)}s; extended to ${dur(seg).toFixed(1)}s with the next whole lines.`);
      }
    });
    seg0 = res.segments[0]; last = res.segments[res.segments.length - 1];

    res.total = res.segments.reduce((a, s) => a + dur(s), 0);
    if (hook && overlaps(seg0, hook)) {
      res.hookOffset = Math.max(0, hook.start - seg0.start);
      if (res.hookOffset > rules.hookWindow + EPS) res.issues.push(`Hook starts at second ${res.hookOffset.toFixed(1)}; it must start within ${rules.hookWindow}s.`);
    }
    if (cliff && last.episode === cliff.episode) {
      res.cliffTail = Math.max(0, last.end - cliff.end);
      if (res.cliffTail > rules.cliffTail + 1.5) res.issues.push(`${res.cliffTail.toFixed(1)}s of footage plays after the cliffhanger; end on it.`);
    }
    if (res.segments.length > rules.maxSegments) res.issues.push(`${res.segments.length} segments; use at most ${rules.maxSegments}.`);
    res.segments.forEach((s, i) => { if (dur(s) < rules.minSegment) res.issues.push(`Segment ${i + 1} is only ${dur(s).toFixed(1)}s; make it at least ${rules.minSegment}s or drop it.`); });
    for (let i = 1; i < res.segments.length; i++) {
      const p = res.segments[i - 1], c = res.segments[i];
      if (c.episode < p.episode || (c.episode === p.episode && c.start < p.end - EPS)) res.issues.push(`Segment ${i + 1} overlaps or comes before segment ${i}; keep segments in story order.`);
    }
    if (res.total < rules.minTotal) res.issues.push(`Total length ${res.total.toFixed(1)}s is under ${rules.minTotal}s; add setup or escalation.`);
    if (res.total > rules.maxTotal) res.issues.push(`Total length ${res.total.toFixed(1)}s is over ${rules.maxTotal}s; cut ${Math.round(res.total - rules.maxTotal)}s+ (skip slow middle parts using an extra segment, or start/end tighter).`);
    return finish(res);
  }
  function finish(res) { res.ok = !res.issues.length && res.segments.length > 0; return res; }
  function report(res) {
    const lines = res.segments.map((s, i) => `- Segment ${i + 1}: ${describe(s)}`);
    lines.push(`- Total length: ${res.total.toFixed(1)}s`);
    if (res.hookOffset !== null) lines.push(`- Hook starts at promo second ${res.hookOffset.toFixed(1)}`);
    if (res.cliffTail !== null) lines.push(`- Footage after the cliffhanger ends: ${res.cliffTail.toFixed(1)}s`);
    return lines.concat(res.issues.map(x => `- PROBLEM: ${x}`)).join("\n");
  }

  function cmpTuple(a, b) { for (let i = 0; i < a.length; i++) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; } return 0; }

  function tighten(promo, index, byEp, rules, maxCut = 15) {
    rules = Object.assign({}, DEFAULT_RULES, rules || {});
    const res = checkPromo(promo, index, byEp, rules);
    const excess = res.total - rules.maxTotal;
    if (!res.segments.length || excess <= 0 || excess > maxCut || res.segments.length >= rules.maxSegments) return null;
    const hook = index[normId((promo.hook || {}).event)], cliff = index[normId((promo.cliffhanger || {}).event)];
    let best = null;
    res.segments.forEach((seg, si) => {
      const evs = (byEp[seg.episode] || []).filter(e => index[e.id] && e.start >= seg.start - EPS && e.end <= seg.end + EPS);
      const dlg = (byEp[seg.episode] || []).filter(e => e.kind === "DIALOGUE");
      const clean = t => !dlg.some(d => d.start < t - EPS && d.end > t + EPS);
      let keepHead = rules.minSegment + 1;
      if (si === 0 && hook && overlaps(seg, hook)) keepHead = Math.max(keepHead, hook.end - seg.start);
      let keepTail = rules.minSegment + 1;
      if (si === res.segments.length - 1 && cliff && overlaps(seg, cliff)) keepTail = Math.max(keepTail, seg.end - cliff.start);
      const lo = seg.start + keepHead, hi = seg.end - keepTail;
      const ends = [...new Set(evs.filter(e => clean(e.end) && lo <= e.end).map(e => e.end))].sort((x, y) => x - y);
      const starts = [...new Set(evs.filter(e => clean(e.start) && e.start <= hi).map(e => e.start))].sort((x, y) => x - y);
      for (const a of ends) {
        const b = starts.find(t => t - a >= Math.max(excess, rules.mergeGap));
        if (b === undefined || b - a > maxCut + rules.mergeGap) continue;
        const cand = [b - a - excess, Math.abs((a + b) / 2 - (seg.start + seg.end) / 2), si, a, b];
        if (!best || cmpTuple(cand, best) < 0) best = cand;
      }
    });
    if (!best) return null;
    const [, , si, a, b] = best, seg = res.segments[si];
    const epEvents = ep => (byEp[ep] || []).filter(e => index[e.id]).slice().sort((x, y) => x.start - y.start || x.index - y.index);
    const firstAt = (ep, t) => epEvents(ep).filter(e => e.start >= t - EPS).reduce((x, y) => (cmpTuple([y.start, y.index], [x.start, x.index]) < 0 ? y : x));
    const lastAt = (ep, t, after) => epEvents(ep).filter(e => e.end <= t + EPS && e.start >= after - EPS).reduce((x, y) => (cmpTuple([y.end, -y.index], [x.end, -x.index]) > 0 ? y : x));
    const newSegs = [];
    res.segments.forEach((sg, i) => {
      const ep = sg.episode;
      if (i !== si) newSegs.push({ first_event: firstAt(ep, sg.start).id, last_event: lastAt(ep, sg.end, sg.start).id, purpose: sg.purpose });
      else {
        newSegs.push({ first_event: firstAt(ep, sg.start).id, last_event: lastAt(ep, a, sg.start).id, purpose: sg.purpose });
        newSegs.push({ first_event: firstAt(ep, b).id, last_event: lastAt(ep, sg.end, b).id, purpose: sg.purpose + " (after code cut)" });
      }
    });
    const out = Object.assign({}, promo, { segments: newSegs });
    if (!checkPromo(out, index, byEp, rules).ok) return null;
    out._code_cut = `code cut ${(b - a).toFixed(1)}s from the middle of segment ${si + 1} (episode ${seg.episode}, ${fmtTs(a)}-${fmtTs(b)}) at line boundaries because the model kept exceeding ${rules.maxTotal}s`;
    return out;
  }

  // ------------------------------------------------------------------ rule-based baseline (no AI)
  const KEYWORDS = {"slap": 6, "punch": 5, "strike": 4, "struck": 4, "shove": 4, "grab": 3, "kneel": 3, "blood": 4, "scream": 4, "gasp": 3, "shock": 3, "reveal": 4, "kiss": 3, "sword": 4, "dagger": 4, "pistol": 4, "gun": 4, "faint": 3, "collapse": 3, "throw": 3, "smash": 4, "shatter": 4, "crash": 3, "fury": 3, "betray": 4, "humiliat": 4, "pregnan": 5, "divorce": 4, "duke": 3, "dead": 3, "kill": 4, "poison": 4, "arrest": 4, "liar": 3, "how dare": 5, "never": 1, "!": 1};
  const EMOTIONS = {"furious": 3, "enraged": 3, "shocked": 3, "horrified": 3, "cruel": 3, "contempt": 2, "venom": 3, "menac": 3, "explosive": 3, "scream": 3, "desperate": 2, "mocking": 2, "cold": 1, "triumphant": 2, "threat": 3, "seething": 3, "disgust": 2, "outraged": 3};
  function intensity(e) {
    const text = e.text.toLowerCase();
    let score = 0;
    for (const [k, w] of Object.entries(KEYWORDS)) if (text.includes(k)) score += w;
    if (e.emotion) { const emo = e.emotion.toLowerCase(); for (const [k, w] of Object.entries(EMOTIONS)) if (emo.includes(k)) score += w; }
    if (e.kind === "CAMERA") score = Math.floor(score / 2);
    return score;
  }
  function baselinePromo(selected) {
    const events = selected.slice().sort((a, b) => a.episode - b.episode || a.start - b.start);
    const lastEp = events[events.length - 1].episode;
    const lastEpEvents = events.filter(e => e.episode === lastEp);
    let lines = lastEpEvents.filter(e => e.kind === "DIALOGUE");
    if (!lines.length) lines = lastEpEvents.filter(e => e.kind === "ACTION");
    const cliff = lines.length ? lines[lines.length - 1] : lastEpEvents[lastEpEvents.length - 1];
    const before = (e, d) => e.episode < cliff.episode || (e.episode === cliff.episode && e.start < cliff.start - d);
    let pool = events.filter(e => e.kind !== "CAMERA" && before(e, 20));
    if (!pool.length) pool = events.filter(e => e.kind !== "CAMERA" && before(e, 0));
    let hook = events[0];
    if (pool.length) hook = pool.reduce((x, y) => (cmpTuple([intensity(y), -y.episode, -y.start], [intensity(x), -x.episode, -x.start]) > 0 ? y : x));
    const span = hook.episode === cliff.episode ? cliff.end - hook.start : null;
    let segments;
    if (span !== null && span >= 20 && span <= 45) segments = [{ first_event: hook.id, last_event: cliff.id, purpose: "hook through to cliffhanger" }];
    else {
      const hookEp = events.filter(e => e.episode === hook.episode);
      const c1 = hookEp.filter(e => e.start >= hook.start && e.end <= hook.start + 20);
      const seg1Last = c1.length ? c1.reduce((x, y) => (y.end > x.end ? y : x)) : hook;
      const c2 = lastEpEvents.filter(e => e.start >= cliff.end - 20 && e.start < cliff.start);
      const seg2First = c2.length ? c2.reduce((x, y) => (y.start < x.start ? y : x)) : cliff;
      const notAfter = seg2First.episode < seg1Last.episode || (seg2First.episode === seg1Last.episode && seg2First.start <= seg1Last.end);
      segments = notAfter ? [{ first_event: hook.id, last_event: cliff.id, purpose: "hook through to cliffhanger" }]
        : [{ first_event: hook.id, last_event: seg1Last.id, purpose: "hook" }, { first_event: seg2First.id, last_event: cliff.id, purpose: "build to cliffhanger" }];
    }
    return {
      title: `Baseline: ${hook.text.slice(0, 40)}`, segments,
      hook: { event: hook.id, type: "other", description: `${labelOf(hook)}: ${hook.text}` },
      cliffhanger: { event: cliff.id, open_question: `(baseline cannot infer) last line: ${cliff.text}` },
      context_for_new_viewer: "(baseline does not judge this)",
      why_it_works: "Rule-based baseline: highest keyword/emotion score as hook, final line of the last episode as cliffhanger.",
      confidence: 1,
    };
  }

  // ------------------------------------------------------------------ prompts (same text as promo/prompts.py)
  const PROMO_SYSTEM = "You are a senior promo editor for vertical microdrama apps such as ReelShort and DramaBox.\nYou cut 20-45 second promos from a show's own footage. A promo is posted on Instagram, TikTok or YouTube\nto make a scrolling stranger stop, watch, and open the show to find out what happens next.\n\nYou receive the screenplay of one or more consecutive episodes as a list of timed events. Each event has:\n- an ID like E03.017 (episode 3, event 17)\n- a time range inside its own episode (timestamps restart at 00:00 every episode)\n- a type: CAMERA (shot description), ACTION, SOUND, or a speaking CHARACTER (with their emotion)\n- its text\n\nBuild ONE promo from these events, following every rule:\n\n1. HOOK FIRST. The first segment must open on the hook: the single most scroll-stopping moment\n   (a slap, a public humiliation, a shocking line, a reveal, a threat, a dramatic entrance).\n   The hook must begin within the first 3 seconds of the promo. Never open on a calm establishing shot.\n   Point \"hook\" and \"cliffhanger\" at the DIALOGUE or ACTION event itself, not at a CAMERA event.\n2. CLIFFHANGER LAST. The promo must end the moment a big question is raised and BEFORE it is answered.\n   The cliffhanger event (or its immediate reaction shot) must be the last thing in the promo.\n   Never include the answer or the resolution.\n3. MAKES SENSE ALONE. A stranger who has never seen the show must understand who is who and what is\n   at stake. Keep only the setup lines a stranger needs; skip moments that depend on earlier episodes.\n4. LENGTH. Total 20-45 seconds (hard limit, checked automatically); aim for 30-40. Shorter and punchier beats longer.\n5. SEGMENTS. 1 to 4 segments. Each segment is one continuous stretch, given by its first and last\n   event ID, inside a single episode. Segments must be in story order and must not overlap.\n   Use more than one segment only to skip slow parts or to join a setup to a payoff across episodes.\n   Each segment must be at least 3 seconds long. Skip at least 3 seconds between two segments of the\n   same episode; a smaller skip looks like a jump cut, so make it one segment instead.\n6. NO CUT LINES. Never start or end a segment in the middle of a line of dialogue.\n7. Use only event IDs that appear in the input. Never invent dialogue, events or IDs.\n\nReturn JSON only, exactly in this shape:\n{\n  \"title\": \"short working title for this promo\",\n  \"segments\": [\n    {\"first_event\": \"E03.017\", \"last_event\": \"E03.041\", \"purpose\": \"what this segment does (hook / setup / escalation / cliffhanger)\"}\n  ],\n  \"hook\": {\"event\": \"E03.017\", \"type\": \"slap | humiliation | shocking line | reveal | threat | entrance | other\",\n           \"description\": \"what the viewer sees and hears in the first 3 seconds\"},\n  \"cliffhanger\": {\"event\": \"E04.052\", \"open_question\": \"the question the viewer is left with\"},\n  \"context_for_new_viewer\": \"who is who and what is at stake, as a stranger would understand it from the promo alone\",\n  \"why_it_works\": \"one line\",\n  \"confidence\": 1\n}\nconfidence is 1-5: how strong you think this promo is. If the episodes contain nothing strong,\nstill return your best attempt and give it a low confidence.\n";
  function buildPromoPrompt(title, story, notes, eventsBlock) {
    return `SHOW: ${title}\n\nSTORY SO FAR (before the selected episodes):\n${story || "(this is the start of the show)"}\n\nSELECTED EPISODES:\n${notes}\n\nEVENTS:\n${eventsBlock}\n\nBuild the promo now. Return JSON only.`;
  }
  function buildRepairPrompt(original, previous, rep) {
    return `${original}\n\n---\nYOUR PREVIOUS ANSWER:\n${JSON.stringify(previous, null, 2)}\n\nTHE AUTOMATIC CHECKER MEASURED YOUR PROMO AND FOUND PROBLEMS:\n${rep}\n\nFix every problem and return the complete corrected JSON (same shape). Return JSON only.`;
  }
  function extractJson(text) {
    let t = String(text).trim().replace(/^```(?:json)?\s*|\s*```$/g, "");
    try { return JSON.parse(t); } catch (e) { /* fall through */ }
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    if (s !== -1 && e > s) { try { return JSON.parse(t.slice(s, e + 1)); } catch (err) { /* fall through */ } }
    throw new Error("Model did not return valid JSON: " + String(text).slice(0, 200));
  }

  // ------------------------------------------------------------------ Gemini (browser fetch, key never leaves the user's browser except to Google)
  async function callGemini(apiKey, model, system, user, temperature = 0.4) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    const body = { systemInstruction: { parts: [{ text: system }] }, contents: [{ role: "user", parts: [{ text: user }] }],
                   generationConfig: { temperature, responseMimeType: "application/json" } };
    let delay = 4000;
    for (let attempt = 1; attempt <= 4; attempt++) {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey }, body: JSON.stringify(body) });
      if (r.ok) {
        const data = await r.json();
        const parts = (((data.candidates || [])[0] || {}).content || {}).parts || [];
        return parts.filter(p => !p.thought).map(p => p.text || "").join("");
      }
      const detail = await r.text();
      if (r.status === 429 && /per day|perday|limit: 0/i.test(detail)) throw new Error(`Daily quota used up for ${model}. Try another model (e.g. a flash-lite one).`);
      if ([429, 500, 502, 503, 504].includes(r.status) && attempt < 4) { await new Promise(res => setTimeout(res, delay)); delay *= 2; continue; }
      throw new Error(`HTTP ${r.status} from Gemini: ${detail.slice(0, 300)}`);
    }
    throw new Error("No answer from Gemini");
  }

  // ------------------------------------------------------------------ the promo maker loop
  async function makePromo(show, selection, opts) {
    opts = opts || {};
    const rules = Object.assign({}, DEFAULT_RULES, opts.rules || {});
    const log = opts.onLog || (() => {});
    const events = selectedEvents(show, selection);
    const index = {}; events.forEach(e => { index[e.id] = e; });
    const byEp = {}; show.numbers.forEach(n => { byEp[n] = show.episodes[n].events; });
    const label = selection.map(selLabel).join("+");
    const attempts = [];
    let promo, result, provider;
    if (!opts.apiKey) {
      provider = "baseline (rules only, no AI)";
      promo = baselinePromo(events);
      result = checkPromo(promo, index, byEp, rules);
      attempts.push({ answer: promo, fixes: result.fixes, issues: result.issues });
    } else {
      provider = `gemini:${opts.model}`;
      const notes = selection.map(s => `Episode ${s.episode} (${fmtTs(show.episodes[s.episode].duration)} long${(s.start !== null || s.end !== null) ? ", using " + selLabel(s) : ""}): ${show.episodes[s.episode].logline}`).join("\n");
      const prompt = buildPromoPrompt(show.title, storySoFar(show, selection[0].episode), notes, events.map(lineOf).join("\n"));
      let user = prompt;
      const tries = opts.maxAttempts || 3;
      for (let t = 1; t <= tries; t++) {
        log(`Asking ${opts.model} (try ${t}/${tries}) ...`);
        promo = extractJson(await callGemini(opts.apiKey, opts.model, PROMO_SYSTEM, user));
        result = checkPromo(promo, index, byEp, rules);
        attempts.push({ answer: promo, fixes: result.fixes, issues: result.issues });
        if (result.ok) break;
        log(`Checker found ${result.issues.length} problem(s)${t < tries ? ": sending the measurements back to the model" : ""}.`);
        user = buildRepairPrompt(prompt, promo, report(result));
      }
    }
    if (!result.ok && result.issues.length && result.issues.every(i => i.includes("Total length") && i.includes("over"))) {
      const cut = tighten(promo, index, byEp, rules);
      if (cut) {
        const note = cut._code_cut; delete cut._code_cut;
        promo = cut; result = checkPromo(promo, index, byEp, rules); result.fixes.push(note);
        attempts.push({ answer: promo, fixes: result.fixes, issues: result.issues, by: "code" });
        log(note);
      }
    }
    return assemble(show, label, provider, promo, result, attempts, index);
  }

  function transcript(show, result) {
    const rows = []; let offset = 0;
    result.segments.forEach((seg, i) => {
      show.episodes[seg.episode].events.slice().sort((a, b) => a.start - b.start || a.index - b.index).forEach(e => {
        if (overlaps(seg, e)) rows.push({ segment: i + 1, promo_time: fmtTs(offset + Math.max(0, e.start - seg.start)), event: e.id, kind: e.kind, who: labelOf(e), text: e.text });
      });
      offset += dur(seg);
    });
    return rows;
  }
  function assemble(show, label, provider, promo, result, attempts, index) {
    const all = {}; show.numbers.forEach(n => show.episodes[n].events.forEach(e => { all[e.id] = e; }));
    const hook = (promo && promo.hook) || {}, cliff = (promo && promo.cliffhanger) || {};
    const he = all[normId(hook.event)], ce = all[normId(cliff.event)];
    return {
      promo_id: `${label}__${provider.split(":")[0].split(" ")[0]}`, status: result.ok ? "ok" : "failed_checks", selection: label, provider,
      attempts: attempts.length, title: (promo && promo.title) || "", total_seconds: Math.round(result.total * 100) / 100,
      segments: result.segments.map((s, i) => ({ order: i + 1, episode: s.episode, start: fmtTs(s.start), end: fmtTs(s.end), seconds: Math.round(dur(s) * 100) / 100, first_event: s.firstEvent, last_event: s.lastEvent, purpose: s.purpose })),
      hook: Object.assign({}, hook, { line: he ? lineOf(he) : null, starts_at_promo_second: result.hookOffset === null ? null : Math.round(result.hookOffset * 100) / 100 }),
      cliffhanger: Object.assign({}, cliff, { line: ce ? lineOf(ce) : null, seconds_after_it: result.cliffTail === null ? null : Math.round(result.cliffTail * 100) / 100 }),
      context_for_new_viewer: (promo && promo.context_for_new_viewer) || "", why_it_works: (promo && promo.why_it_works) || "",
      model_confidence: promo ? promo.confidence : null, checker: { fixes: result.fixes, issues: result.issues },
      attempt_history: attempts, transcript: transcript(show, result),
    };
  }

  const api = { parseTs, fmtTs, parseEpisode, loadShow, parseSelection, selectedEvents, storySoFar, checkPromo, tighten,
                baselinePromo, intensity, makePromo, extractJson, PROMO_SYSTEM, DEFAULT_RULES, lineOf, selLabel };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.PromoCore = api;
})(typeof window !== "undefined" ? window : globalThis);
