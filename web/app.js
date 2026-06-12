/* ============================================================
   Verdict — search → shortlist → candidate dossier with a
   plain-language, zoomable decision graph.
   ============================================================ */

import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const $ = (s, el = document) => el.querySelector(s);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

mermaid.initialize({
  startOnLoad: false,
  theme: "base",
  flowchart: { curve: "basis", htmlLabels: true, padding: 12 },
  themeVariables: {
    fontFamily: "'Instrument Sans', sans-serif",
    fontSize: "13px",
    primaryColor: "#FFFFFF",
    primaryBorderColor: "#141414",
    primaryTextColor: "#141414",
    lineColor: "#A8A29E",
    edgeLabelBackground: "#FFFFFF",
    clusterBkg: "#F5F5F4",
    clusterBorder: "#D9D6D2",
  },
});

let RUBRIC = null;          // fetched once: weights, gates, dampener catalog
const explainCache = new Map();
let lastRows = [], lastSecs = 0, lastQuery = "", lastRole = "";
let sortMode = "relevance"; // "relevance" = closest to the typed query · "fit" = best for the JD

/* ------------------------------------------------------------
   plain-language dictionaries
   ------------------------------------------------------------ */
const roleName = () => lastRole || RUBRIC?.meta?.role || "the hiring role";

const RULE_NAMES = {
  yoe_fit: "Right amount of experience",
  title_role_match: "Job title matches your role text",
  core_title_family: "Job title matches the hiring role",
  product_company_tenure: "Time spent at product companies",
  trajectory: "Career growth over time",
  stability: "Job stability — no job-hopping",
  location_fit: "Location works for the hiring role",
  external_validation: "Outside proof — GitHub, talks, papers",
  core_skill_coverage: "Covers the core skills",
  resume_text_match: "Resume matches the role text",
  shipped_search_ranking_reco: "Built search / recommendation systems",
  production_embeddings_retrieval: "Built embeddings / vector retrieval",
  ranking_evaluation: "Measured & improved result quality",
  ml_production_scale: "Ran ML in production at scale",
  nlp_ir_depth: "Deep NLP / search experience",
  llm_production: "Used LLMs in production",
};

const DAMP_PATTERNS = [
  [/research/i, "Pure research career — never shipped to production"],
  [/services|consult/i, "Career spent only at consulting / outsourcing firms"],
  [/title|hop|tenure ~/i, "Switches jobs too often"],
  [/manager|architect|lead roles/i, "Hasn't written production code recently"],
  [/cv\/|speech|vision|robot/i, "Computer-vision / speech background — not NLP or search"],
  [/langchain|framework/i, "Only recent LLM-tool experience — lacks real ML depth"],
  [/adjacent/i, "Nearby tech background, but not enough proven ML / search depth"],
  [/non-?technical/i, "Non-technical career with no ML evidence"],
  [/fit-shaped|not backed/i, "Looks right on paper, but no real ML / search evidence behind it"],
];

const FLAG_TITLES = {
  CONSISTENCY_DOUBT: "Claims don't add up",
  LONG_NOTICE: "Long notice period",
  LOW_RESPONSE: "Rarely replies to recruiters",
  STALE_PROFILE: "Profile not active recently",
  NOT_LOOKING: "Not actively looking",
};

const LOC_LABEL = { preferred: "preferred city", tier1: "tier-1 city", india_other: "india", abroad: "abroad" };

const pct = (v) => `${Math.min(100, Math.round(v * 100))}%`;
const cacheKey = (cid, role = roleName()) => `${cid}|${role}`;
const num = (v, fallback = 0) => Number.isFinite(+v) ? +v : fallback;

function locText(d) {
  const base = LOC_LABEL[d.location] || d.location || "—";
  return d.location === "abroad" && d.willing_to_relocate ? `${base} — will relocate` : base;
}

function friendlyRule(k) { return RULE_NAMES[k] || prettify(k); }

function level(v) {
  return v >= 0.85 ? "excellent" : v >= 0.6 ? "strong" : v >= 0.4 ? "partial" : v >= 0.15 ? "weak" : "missing";
}

function friendlyDamp(s) {
  const hit = DAMP_PATTERNS.find(([re]) => re.test(s));
  return { title: hit ? hit[1] : "Screened down per the job description", detail: s };
}

function friendlyFlag(s) {
  const m = String(s).match(/^([A-Z_]+):\s*(.*)$/);
  if (m) return { title: FLAG_TITLES[m[1]] || titleCase(m[1]), detail: m[2] };
  return { title: null, detail: s };
}

function matchWord(score) {
  return score >= 0.45 ? "Excellent match" : score >= 0.3 ? "Good match" : score >= 0.15 ? "Possible match" : "Weak match";
}

function searchWord(p) {
  return p >= 0.75 ? "Excellent" : p >= 0.5 ? "Good" : p >= 0.25 ? "Fair" : "Weak";
}

function rankWhy(r, topScore) {
  const chips = [];
  const shortlistPct = Math.round((num(r.relevance) / Math.max(topScore, 1e-6)) * 100);
  chips.push({ cls: shortlistPct >= 90 ? "good" : shortlistPct >= 75 ? "warn" : "bad", text: `${shortlistPct}% of #1 shortlist score` });

  const notice = num(r.notice_days, null);
  if (notice != null && notice > 90) chips.push({ cls: "bad", text: `${notice}d notice` });
  else if (notice != null && notice > 60) chips.push({ cls: "warn", text: `${notice}d notice` });

  const title = r.titleFit ?? r.familyFit;
  if (title != null && title < 0.8) chips.push({ cls: "warn", text: `title fit ${pct(title)}` });

  if (r.skillFit != null && r.skillFit < 0.5) chips.push({ cls: "bad", text: `proven skills ${pct(r.skillFit)}` });
  else if (r.skillFit != null && r.skillFit < 0.75) chips.push({ cls: "warn", text: `proven skills ${pct(r.skillFit)}` });

  if (r.textMatch != null && r.textMatch < 0.65) chips.push({ cls: "warn", text: `resume evidence ${pct(r.textMatch)}` });

  const rr = num(r.response_rate, null);
  if (rr != null && rr < 0.25) chips.push({ cls: "bad", text: `${pct(rr)} response rate` });
  else if (rr != null && rr < 0.5) chips.push({ cls: "warn", text: `${pct(rr)} response rate` });

  const avail = num(r.availability, null);
  if (avail != null && avail < 0.35) chips.push({ cls: "bad", text: `reachability ${pct(avail)}` });
  else if (avail != null && avail < 0.5) chips.push({ cls: "warn", text: `reachability ${pct(avail)}` });

  return chips.slice(0, 5);
}

/* ------------------------------------------------------------
   bootstrap
   ------------------------------------------------------------ */
init();

async function init() {
  intro();
  wireSearch();
  wireDossier();
  pollHealth();
  try { RUBRIC = await (await fetch("/rubric")).json(); } catch { /* tolerated; weights default */ }
}

function intro() {
  if (reduced || !window.gsap) return;
  const cp = { clearProps: "all" }; // always land in natural CSS state — never stuck invisible
  gsap.timeline({ defaults: { ease: "power4.out" } })
    .from(".brand .mark", { scale: 0, duration: .6, ease: "back.out(2.5)", ...cp })
    .from(".brand-word, .nav-links a, .health", { y: -12, autoAlpha: 0, stagger: .06, duration: .6, ...cp }, .1)
    .from(".hero-title .line-in", { yPercent: 110, duration: .9, stagger: .1, ...cp }, .2)
    .from("#heroSub", { y: 16, autoAlpha: 0, duration: .7, ...cp }, .6)
    .from(".console", { y: 24, autoAlpha: 0, duration: .8, ...cp }, .75)
    .from(".suggestions .sug-label, .suggestions .sug", { y: 10, autoAlpha: 0, stagger: .05, duration: .45, ...cp }, .95);
}

async function pollHealth() {
  const dot = $("#healthDot"), txt = $("#healthText");
  try {
    const h = await (await fetch("/healthz")).json();
    dot.className = "health-dot ok";
    txt.textContent = `${Number(h.candidates_indexed).toLocaleString()} candidates indexed`;
  } catch {
    dot.className = "health-dot bad";
    txt.textContent = "index offline — start uvicorn api:app";
  }
}

/* ------------------------------------------------------------
   search
   ------------------------------------------------------------ */
function wireSearch() {
  $("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  $("#suggestions").addEventListener("click", (e) => {
    const b = e.target.closest(".sug");
    if (!b) return;
    $("#q").value = b.dataset.q;
    runSearch();
  });
  $("#sortCtl").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (!b || b.dataset.mode === sortMode) return;
    sortMode = b.dataset.mode;
    $("#sortCtl .on")?.classList.remove("on");
    b.classList.add("on");
    if (lastRows.length) renderLedger(lastRows, lastSecs, { noScroll: true });
  });
}

async function runSearch() {
  const q = $("#q").value.trim();
  const role = $("#roleInput").value.trim();
  if (!q && !role) { $("#roleInput").focus(); return; }

  // pasting a candidate id goes straight to the dossier
  if (/^cand[_-]?\d+$/i.test(q)) {
    lastRole = role || lastRole;
    openDossier(q.toUpperCase().replace(/-/, "_"), null);
    return;
  }

  const btn = $("#goBtn");
  btn.classList.add("busy");
  $("#goBtn .btn-label").textContent = "Scoring…";
  showState(`<div class="deliberating"><span class="spin"></span>Scoring evidence and weighing credibility across the index…</div>`);
  $("#results").hidden = true;

  const presetSel = $("#preset").value;
  const p = new URLSearchParams({
    query: q || role,
    preset: presetSel === "all" ? "ai_ml" : presetSel,
    top: $("#topN").value,
    min_yoe: $("#minYoe").value || 0,
    max_yoe: $("#maxYoe").value || 50,
    location: $("#location").value,
  });
  if (role) p.set("role", role);
  if (presetSel === "all" && !role) {
    // union of every preset's families/categories — no role filtered out
    p.set("families", "ml_engineer,applied_scientist,nlp_engineer,search_engineer,data_scientist,mlops_engineer,backend,swe,fullstack,devops,data_engineer,analyst");
    p.set("categories", "ml_core,mlops,llm,nlp,search,ranking,embeddings,vector_db,backend,data_eng,cloud,devops,analytics");
  }
  // Location "India" also welcomes abroad candidates willing to relocate.
  if ($("#location").value === "india") p.set("relocation_ok", "true");
  // Toggles are hard filters, not just boosts — what you tick is what you get.
  if ($("#availability").checked) {
    p.set("availability", "true");        // boost reachable profiles in ranking
    p.set("max_notice_days", "60");       // and drop anyone slower than 60 days
    p.set("min_response_rate", "0.25");   // or who ignores recruiters
  }
  if ($("#goodCompanies").checked) {
    p.set("good_companies", "true");      // boost product-company careers
    p.set("min_product_share", "0.5");    // and require ≥50% of career at product companies
  }

  const t0 = performance.now();
  try {
    const res = await fetch(`/search?${p}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const data = await res.json();
    let rows = data.results;

    // second pass: judge every result against the job description, so the
    // Match column always agrees with the dossier's "overall match".
    lastQuery = q || role;
    lastRole = role || q;
    const relMax = Math.max(...rows.map(r => +r.score || 0), 1e-6);
    rows.forEach((r, i) => {
      r.relevance = +r.score || 0;
      r.searchOrder = i;
      r.relPct = r.relevance / relMax;            // relative to the best result
      r.textMatch = r.text_match != null ? +r.text_match : null;
      r.titleFit = r.title_fit != null ? +r.title_fit : null;
      r.familyFit = r.family_fit != null ? +r.family_fit : null;
      r.skillFit = r.skill_fit != null ? +r.skill_fit : null;
    });
    if (rows.length) {
      showState(`<div class="deliberating"><span class="spin"></span>Found ${rows.length} candidates — now judging each one against ${esc(roleName())}…</div>`);
      try {
        const vp = new URLSearchParams({
          ids: rows.map(r => r.candidate_id).join(","),
          role: roleName(),
          preset: presetSel === "all" ? "ai_ml" : presetSel,
          min_yoe: $("#minYoe").value || 0,
          max_yoe: $("#maxYoe").value || 50,
        });
        if ($("#availability").checked) vp.set("availability", "true");
        const vres = await fetch(`/verdicts?${vp}`);
        if (vres.ok) {
          const verds = await vres.json();
          const byId = new Map(verds.map(v => [v.candidate_id, v]));
          rows.forEach(r => {
            const v = byId.get(r.candidate_id);
            if (v) { r.verdict = v.score; explainCache.set(cacheKey(v.candidate_id), v); }
          });
        }
      } catch { /* tolerated — Match falls back to relative search score */ }
    }

    hideState();
    lastRows = rows; lastSecs = (performance.now() - t0) / 1000;
    renderLedger(rows, lastSecs);
  } catch (err) {
    showState(`<div class="err">Search failed.\n${esc(String(err.message || err))}</div>`);
  } finally {
    btn.classList.remove("busy");
    $("#goBtn .btn-label").textContent = "Search";
  }
}

function renderLedger(rows, secs, opts = {}) {
  const ledger = $("#ledger");
  ledger.innerHTML = "";
  const judged = rows.some(r => r.verdict != null);

  // order by the active sort mode — both scores live on every row
  rows = [...rows];
  if (sortMode === "fit" && judged) {
    rows.sort((a, b) => (b.verdict ?? -1) - (a.verdict ?? -1));
  } else {
    rows.sort((a, b) => a.searchOrder - b.searchOrder);
  }

  $("#resultsStats").textContent =
    `${rows.length} results · ${secs.toFixed(1)}s · ` +
    (sortMode === "fit" && judged
      ? `ordered by overall fit to ${roleName()}`
      : `ordered by closeness to what you searched — the % still shows fit to ${roleName()}`) +
    ` · click a row for the full story`;
  $("#resultsStats").textContent =
    `${rows.length} results | fit target: ${roleName()} | ${secs.toFixed(1)}s | ` +
    (sortMode === "fit" && judged
      ? `ordered by overall fit to ${roleName()}`
      : `ordered by shortlist score; % shows job fit to ${roleName()}`) +
    ` | row chips explain why a high-fit candidate can rank lower`;
  $("#results").hidden = false;

  if (!rows.length) {
    showState(`<div class="err">No candidate survived the filters. Loosen them and try again.</div>`);
    return;
  }

  const maxScore = Math.max(...rows.map(r => r.relevance || 0), 1e-6);
  const fitOf = r => r.verdict != null ? Math.min(1, r.verdict) : (r.relevance / maxScore);
  const fitMax = Math.max(...rows.map(fitOf), 1e-6);
  let pos = 0;
  for (const r of rows) {
    r.rank = ++pos;
    const li = document.createElement("li");
    li.className = "row";
    li.dataset.id = r.candidate_id;
    const reason = String(r.reasoning || "");
    const concernIdx = reason.indexOf("Concern:");
    const reasonHtml = concernIdx >= 0
      ? `${esc(reason.slice(0, concernIdx))}<span class="concern">${esc(reason.slice(concernIdx))}</span>`
      : esc(reason);
    const why = rankWhy(r, maxScore);
    const whyHtml = why.length
      ? `<div class="rank-why"><b>Why shortlist #${r.rank}</b>${why.map(x => `<span class="why-chip ${x.cls}">${esc(x.text)}</span>`).join("")}</div>`
      : "";
    li.innerHTML = `
      <span class="rank">${String(r.rank).padStart(2, "0")}</span>
      <div class="who">
        <span class="cid">${esc(r.candidate_id)}</span>
        <span class="name">${esc(r.title || "—")}</span>
        <p class="reason">${reasonHtml}</p>
        ${whyHtml}
      </div>
      <span class="cell-yoe">${(+r.yoe).toFixed(1)}y</span>
      <span class="cell-loc">${esc(LOC_LABEL[r.location] || r.location || "—")}</span>
      <div class="cell-score">
        <span class="num ${r.verdict != null ? (r.verdict >= 0.3 ? "g" : r.verdict >= 0.15 ? "a" : "r") : ""}">${r.verdict != null ? pct(r.verdict) : Math.round(fitOf(r) * 100)}</span>
        <div class="scorebar"><i class="${r.verdict != null ? (r.verdict >= 0.3 ? "hi" : r.verdict >= 0.15 ? "mid" : "lo") : ""}"></i></div>
      </div>`;
    li.addEventListener("click", () => openDossier(r.candidate_id, r));
    ledger.appendChild(li);
  }

  // staggered reveal + score bars sweep
  const items = [...ledger.children];
  const bars = items.map(li => $(".scorebar i", li));
  const widths = rows.map(r => `${(fitOf(r) / fitMax) * 100}%`);
  if (!reduced && window.gsap) {
    gsap.fromTo(items, { y: 22, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .55, stagger: .045, ease: "power3.out", clearProps: "all" });
    items.forEach((li, i) => gsap.to(bars[i], {
      width: widths[i], duration: .9, delay: .15 + i * .045, ease: "power4.out",
    }));
  } else {
    items.forEach((li, i) => bars[i].style.width = widths[i]);
  }
  if (!opts.noScroll) $("#results").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
}

function showState(html) { const s = $("#stateBox"); s.innerHTML = html; s.hidden = false; }
function hideState() { $("#stateBox").hidden = true; }

/* ------------------------------------------------------------
   dossier panel
   ------------------------------------------------------------ */
function wireDossier() {
  $("#dossierClose").addEventListener("click", closeDossier);
  $("#veil").addEventListener("click", closeDossier);
  addEventListener("keydown", (e) => { if (e.key === "Escape") closeDossier(); });
}

function openDossier(cid, ctx) {
  // ctx = the shortlist row this dossier was opened from (carries search-fit data)
  ctx = ctx ?? lastRows.find(r => r.candidate_id === cid) ?? null;
  const panel = $("#dossier"), veil = $("#veil");
  veil.hidden = false;
  panel.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  if (!reduced && window.gsap) {
    gsap.to(veil, { opacity: 1, duration: .4 });
    gsap.fromTo(panel, { x: "102%" }, { x: "0%", duration: .65, ease: "power4.out" });
  } else {
    veil.style.opacity = 1; panel.style.transform = "translateX(0)";
  }
  loadDossier(cid, ctx);
}

function closeDossier() {
  const panel = $("#dossier"), veil = $("#veil");
  if (panel.getAttribute("aria-hidden") === "true") return;
  panel.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  const done = () => { veil.hidden = true; };
  if (!reduced && window.gsap) {
    gsap.to(veil, { opacity: 0, duration: .35, onComplete: done });
    gsap.to(panel, { x: "102%", duration: .5, ease: "power3.in" });
  } else { panel.style.transform = ""; done(); }
}

async function loadDossier(cid, ctx) {
  const body = $("#dossierBody");
  body.innerHTML = `
    <div class="d-kicker"><span class="mark mark-sm"></span> ${esc(cid)}</div>
    <div style="margin-top:26px">
      ${`<div class="skel skel-line" style="width:62%"></div>`.repeat(1)}
      ${`<div class="skel skel-line"></div>`.repeat(4)}
      <div class="skel" style="height:220px;margin-top:22px"></div>
    </div>`;

  try {
    let d = explainCache.get(cacheKey(cid));
    if (!d) {
      const ep = new URLSearchParams({
        role: roleName(),
        preset: $("#preset").value === "all" ? "ai_ml" : $("#preset").value,
        min_yoe: $("#minYoe").value || 0,
        max_yoe: $("#maxYoe").value || 50,
      });
      if ($("#availability").checked) ep.set("availability", "true");
      const res = await fetch(`/explain/${encodeURIComponent(cid)}?${ep}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      d = await res.json();
      explainCache.set(cacheKey(cid), d);
    }
    await renderDossier(d, ctx);
  } catch (err) {
    body.innerHTML = `
      <div class="d-kicker"><span class="mark mark-sm"></span> ${esc(cid)}</div>
      <div class="state" style="padding:30px 0"><div class="err">No verdict on file.\n${esc(String(err.message || err))}</div></div>`;
  }
}

async function renderDossier(d, ctx) {
  const body = $("#dossierBody");
  const verdict = classifyVerdict(d);
  const hasSearch = !!(ctx && lastQuery);

  const dims = [];
  if (hasSearch) {
    if (ctx.textMatch != null) dims.push({ tag: "your search", name: "Resume mentions what you searched", desc: `how strongly their story matches “${lastQuery}”`, v: ctx.textMatch });
    if (ctx.titleFit != null) dims.push({ tag: "your role", name: "Title matches the role", desc: `how closely their title matches ${roleName()}`, v: ctx.titleFit });
    if (ctx.familyFit != null) dims.push({ tag: "your search", name: "Right kind of role", desc: "their job titles fit the role type you picked", v: ctx.familyFit });
    if (ctx.skillFit != null) dims.push({ tag: "your search", name: "Has the skills you asked for", desc: "proven skills in the categories you searched", v: ctx.skillFit });
  }
  dims.push(
    { tag: "hiring role", name: "Skills & experience fit", desc: "how well their career matches the hiring role", v: d.J },
    { tag: "hiring role", name: "Resume trustworthiness", desc: "do their claims add up?", v: d.C },
    { tag: "hiring role", name: "Ease of hiring", desc: "responds to recruiters, notice period, logistics", v: d.A },
  );

  body.innerHTML = `
    <div class="d-kicker"><span class="mark mark-sm"></span> ${esc(d.candidate_id)}</div>
    <h2 class="d-name">${esc(d.title || "Untitled career")}</h2>
    <div class="d-meta">
      <span class="chip">${esc(prettify(d.family || "—"))}</span>
      <span class="chip">${(+d.yoe).toFixed(1)} years experience</span>
      <span class="chip">${esc(locText(d))}</span>
    </div>

    <div class="verdict-block">
      <span class="stamp ${verdict.cls}">${verdict.label}</span>
      <div class="verdict-score">
        ${hasSearch ? `
        <div class="vs-label">Match for your search</div>
        <div class="vs-num" id="vsNum">0%</div>
        <div class="vs-word">${searchWord(ctx.relPct)} for “${esc(trunc(lastQuery, 26))}”</div>
        <div class="vs-divider"></div>
        <div class="vs-label">Fit to the hiring role</div>
        <div class="vs-num vs-num-sm ${verdict.cls === "green" ? "" : verdict.cls === "red" ? "r" : "a"}">${pct(d.score)}</div>
        <div class="vs-word ${verdict.cls}">${matchWord(d.score)}</div>
        ` : `
        <div class="vs-label">Fit to the hiring role</div>
        <div class="vs-num" id="vsNum">0%</div>
        <div class="vs-word ${verdict.cls}">${matchWord(d.score)}</div>
        `}
      </div>
      <div class="verdict-dims">
        ${dims.map(x => `
          <div class="dim-row">
            <div class="dim-info"><b>${x.name} <i class="dim-tag">${x.tag}</i></b><span>${x.desc}</span></div>
            <div class="dim-bar"><i class="${x.v >= 0.6 ? "hi" : x.v >= 0.35 ? "mid" : "lo"}" style="--w:${pct(x.v)}"></i></div>
            <span class="dim-pct">${pct(x.v)}</span>
          </div>`).join("")}
      </div>
    </div>
    ${hasSearch ? "" : `<p class="d-explain" style="margin-top:10px">Judged against the hiring role: <b>${esc(roleName())}</b>.</p>`}

    <section class="d-section">
      <div class="d-section-head"><h3>How the decision was made</h3><span class="hint">drag to move · scroll to zoom</span></div>
      <p class="d-explain">Follow the arrows top to bottom. <b class="g">Green</b> helped this candidate, <b class="r">red</b> counted against them.</p>
      <div class="graph-wrap" id="graphWrap">
        <div class="graph-canvas" id="graphCanvas"></div>
        <div class="graph-controls">
          <button type="button" id="zoomIn" title="zoom in">+</button>
          <button type="button" id="zoomOut" title="zoom out">−</button>
          <button type="button" id="zoomFit" title="fit to view">⌖</button>
        </div>
      </div>
      <div class="graph-legend">
        <span class="k"><span class="sw g"></span>helped them</span>
        <span class="k"><span class="sw r"></span>hurt them</span>
        <span class="k"><span class="sw a"></span>caution</span>
        <span class="k"><span class="sw s"></span>checkpoint</span>
      </div>
    </section>

    <section class="d-section" id="scorecardSection">
      <div class="d-section-head"><h3>The hiring-role scorecard</h3><span class="hint">how the fit score was earned</span></div>
      <div id="scorecardSummary" class="d-explain"></div>
      <div class="rules-grid" id="rulesGrid"></div>
    </section>

    ${d.evidence?.length ? `
    <section class="d-section">
      <div class="d-section-head"><h3>Proof from their resume</h3><span class="n">${d.evidence.length} item${d.evidence.length > 1 ? "s" : ""}</span></div>
      <div class="quote-list">${d.evidence.map(renderEvidence).join("")}</div>
    </section>` : ""}

    <section class="d-section">
      <div class="d-section-head"><h3>Red flags</h3><span class="hint">screens from the job description</span></div>
      ${d.dampeners?.length
        ? `<div class="flag-list">${d.dampeners.map(x => {
            const f = friendlyDamp(x);
            return `<div class="flag"><span class="fmark">✗</span><div><b class="ftitle">${esc(f.title)}</b><span class="fdetail">${esc(f.detail)}</span></div></div>`;
          }).join("")}</div>`
        : `<div class="allclear">No red flags — nothing in this career matched a screen-out.</div>`}
    </section>

    <section class="d-section">
      <div class="d-section-head"><h3>Trust &amp; logistics</h3></div>
      ${d.credibility_flags?.length
        ? `<div class="flag-list">${d.credibility_flags.map(x => {
            const f = friendlyFlag(x);
            return `<div class="flag"><span class="fmark">!</span><div>${f.title ? `<b class="ftitle">${esc(f.title)}</b>` : ""}<span class="fdetail">${esc(f.detail)}</span></div></div>`;
          }).join("")}</div>`
        : `<div class="allclear">Their resume claims are internally consistent.</div>`}
      <div style="height:10px"></div>
      ${d.availability_flags?.length
        ? `<div class="flag-list">${d.availability_flags.map(x => {
            const f = friendlyFlag(x);
            return `<div class="flag amber"><span class="fmark">~</span><div>${f.title ? `<b class="ftitle">${esc(f.title)}</b>` : ""}<span class="fdetail">${esc(f.detail)}</span></div></div>`;
          }).join("")}</div>`
        : `<div class="allclear">Easy to reach — no hiring-logistics concerns.</div>`}
    </section>`;

  renderRules(d);
  await renderGraph(d, ctx);

  // entrance choreography + score count-up
  const numEl = $("#vsNum");
  const vsTarget = hasSearch ? ctx.relPct : d.score;
  if (!reduced && window.gsap) {
    gsap.from("#dossierBody > *", { y: 24, autoAlpha: 0, duration: .6, stagger: .07, ease: "power3.out", clearProps: "all" });
    const o = { v: 0 };
    gsap.to(o, { v: Math.min(1, vsTarget), duration: 1.4, ease: "power3.out", onUpdate: () => numEl.textContent = pct(o.v) });
    gsap.fromTo(".stamp", { scale: 1.6, autoAlpha: 0 }, { scale: 1, autoAlpha: 1, duration: .5, delay: .5, ease: "back.out(2)" });
    gsap.from(".dim-bar i, .rule-bar i", { scaleX: 0, transformOrigin: "left", duration: .8, stagger: .03, delay: .3, ease: "power4.out", clearProps: "transform" });
  } else {
    numEl.textContent = pct(vsTarget);
  }
  $(".dossier-scroll").scrollTop = 0;
}

function locationOK(d) {
  // the JD gate only rejects abroad candidates who won't relocate
  return d.location !== "abroad" || !!d.willing_to_relocate;
}

function classifyVerdict(d) {
  const g = RUBRIC?.gates || { yoe_min: 2, yoe_max: 15 };
  const gated = d.yoe < g.yoe_min || d.yoe > g.yoe_max || !locationOK(d);
  if (gated) return { cls: "red", label: "fails basic requirements" };
  if (d.dampeners?.length) return { cls: "red", label: `${d.dampeners.length} red flag${d.dampeners.length > 1 ? "s" : ""}` };
  if (d.credibility_flags?.length) return { cls: "amber", label: "needs review" };
  if (d.J >= 0.55) return { cls: "green", label: "strong profile" };
  return { cls: "amber", label: "weak fit" };
}

/* ---- scorecard ---- */
function ruleWeight(k) {
  return RUBRIC?.crisp_rules?.[k]?.weight ?? RUBRIC?.fuzzy_predicates?.[k]?.weight ?? 1;
}

function renderRules(d) {
  const grid = $("#rulesGrid");
  const rows = Object.entries(d.rules)
    .map(([k, v]) => ({ k, v, w: ruleWeight(k) }))
    .sort((a, b) => (b.w * b.v) - (a.w * a.v));

  // weights sum to ~1, so ×100 reads as "points out of 100"
  const fmt = (n) => (Math.round(n * 10) / 10).toString();
  const earned = rows.reduce((s, r) => s + r.v * r.w * 100, 0);
  const total = rows.reduce((s, r) => s + r.w * 100, 0);
  const damped = d.dampeners?.length;
  $("#scorecardSummary").innerHTML =
    `They earned <b>${fmt(earned)}</b> of <b>${fmt(total)}</b> possible points` +
    (damped
      ? ` — then red flags cut the fit score down to <b class="r">${pct(d.J)}</b>.`
      : ` — a fit score of <b>${pct(d.J)}</b>.`);

  grid.innerHTML = rows.map(({ k, v, w }) => `
    <div class="rule-row">
      <span class="rname" title="${esc(prettify(k))}">${esc(friendlyRule(k))}</span>
      <div class="rule-bar"><i class="${v >= 0.7 ? "hi" : v >= 0.4 ? "mid" : "lo"}" style="width:${(v * 100).toFixed(0)}%"></i></div>
      <span class="rlevel ${v >= 0.7 ? "g" : v >= 0.4 ? "" : "r"}">${level(v)}</span>
      <span class="rcontrib">${fmt(v * w * 100)} / ${fmt(w * 100)} pts</span>
    </div>`).join("");
}

/* ---- evidence quotes ---- */
function renderEvidence(note) {
  const m = String(note).match(/^\[([\w_]+)\]\s*"?([\s\S]*?)"?$/);
  if (m) {
    return `<div class="quote"><span class="qtag">${esc(friendlyRule(m[1]))}</span><em>“${esc(m[2])}…”</em></div>`;
  }
  return `<div class="quote plain">${esc(note)}</div>`;
}

/* ------------------------------------------------------------
   the decision graph — plain-language, built from /explain
   ------------------------------------------------------------ */
let graphSeq = 0;

async function renderGraph(d, ctx) {
  const canvas = $("#graphCanvas");
  try {
    const { svg } = await mermaid.render(`verdictGraph${++graphSeq}`, mermaidSource(d, ctx));
    canvas.innerHTML = svg;
    enableZoom($("#graphWrap"), canvas);
  } catch (e) {
    canvas.innerHTML = `<div style="font-size:13px;color:#8C1D13;padding:16px">decision graph unavailable: ${esc(String(e.message || e))}</div>`;
  }
}

function mermaidSource(d, ctx) {
  const g = RUBRIC?.gates || { yoe_min: 2, yoe_max: 15 };
  const yoeOK = d.yoe >= g.yoe_min && d.yoe <= g.yoe_max;
  const locOK = locationOK(d);
  const basicsOK = yoeOK && locOK;

  const entries = Object.entries(d.rules).map(([k, v]) => ({ k, v, w: ruleWeight(k) }));
  const wins = entries.filter(e => e.v >= 0.7).sort((a, b) => b.v * b.w - a.v * a.w).slice(0, 3);
  const cuts = entries.filter(e => e.v < 0.40).sort((a, b) => b.w - a.w).slice(0, 3);

  const L = [];
  const q = s => `"${String(s).replace(/"/g, "'").replace(/[\[\]{}|#]/g, " ")}"`;
  const node = (id, label, shape, cls) => {
    const [o, c] = shape === "diamond" ? ["{", "}"] : shape === "round" ? ["([", "])"] : ["[", "]"];
    L.push(`${id}${o}${q(label)}${c}:::${cls}`);
  };

  L.push("flowchart TD");

  /* 1 — who */
  node("IN", `${d.title || d.candidate_id}<br/>${(+d.yoe).toFixed(1)} yrs · ${locText(d)}`, "round", "spine");

  /* 1b — how your search found them (only when opened from a shortlist) */
  const hasSearch = !!(ctx && lastQuery);
  if (hasSearch) {
    const rp = ctx.relPct ?? 0;
    node("Q", `Your search: '${trunc(lastQuery, 34)}'<br/>${pct(rp)} match — ${searchWord(rp).toLowerCase()}`, "box", rp >= 0.5 ? "pass" : rp >= 0.25 ? "warn" : "fail");
    L.push("IN --> Q");
    const sParts = [
      ["QT", "Resume mentions it", ctx.textMatch],
      ["QF", "Right kind of role", ctx.familyFit],
      ["QS", "Has the skills asked", ctx.skillFit],
    ].filter(([, , v]) => v != null);
    if (sParts.length) {
      L.push(`subgraph QP[${q("Why your search found them")}]`);
      L.push("direction TB");
      sParts.forEach(([id, label, v]) => node(id, `${label}<br/>${pct(v)}`, "box", v >= 0.5 ? "pass" : v >= 0.25 ? "warn" : "fail"));
      L.push("end");
      L.push("Q --- QP");
    }
  }

  /* 2 — basics */
  node("BASIC", "Meets the basic requirements?", "diamond", basicsOK ? "pass" : "fail");
  L.push(`${hasSearch ? "Q" : "IN"} -->|${q("now judged for the hiring role")}| BASIC`);
  if (!basicsOK) {
    node("OUT", "✗ Screened out before scoring", "box", "verdictR");
    L.push(`BASIC -->|${q(`experience ${yoeOK ? "✓" : "✗"} · location ${locOK ? "✓" : "✗"}`)}| OUT`);
  }

  /* 3 — skills check with helped/hurt clusters */
  node("FIT", `Skills &amp; experience check<br/>${pct(d.J)} fit`, "box", d.J >= 0.55 ? "pass" : d.J >= 0.3 ? "warn" : "fail");
  L.push(`BASIC -->|${q(basicsOK ? "yes — experience ✓ · location ✓" : "scored anyway, for the record")}| FIT`);

  if (wins.length) {
    L.push(`subgraph GOOD[${q("✓ What helped them")}]`);
    L.push("direction TB");
    wins.forEach((e, i) => node(`S${i}`, `${friendlyRule(e.k)}<br/>${level(e.v)}`, "box", "pass"));
    L.push("end");
    L.push("FIT --- GOOD");
  }
  if (cuts.length) {
    L.push(`subgraph BAD[${q("✗ What hurt them")}]`);
    L.push("direction TB");
    cuts.forEach((e, i) => node(`R${i}`, `${friendlyRule(e.k)}<br/>${level(e.v)}`, "box", "fail"));
    L.push("end");
    L.push("FIT --- BAD");
  }

  /* 4 — red flags */
  const nd = d.dampeners?.length || 0;
  node("FLAG", "Any red flags?", "diamond", nd ? "fail" : "pass");
  L.push("FIT --> FLAG");
  d.dampeners?.slice(0, 3).forEach((x, i) => {
    node(`D${i}`, friendlyDamp(x).title, "box", "fail");
    L.push(`FLAG --- D${i}`);
  });

  /* 5 — trust + reachability */
  node("TRUST", `Can we trust the resume?<br/>${pct(d.C)}`, "box", d.credibility_flags?.length ? "warn" : "pass");
  L.push(`FLAG -->|${q(nd ? `yes — ${nd} found, score reduced` : "none")}| TRUST`);
  (d.credibility_flags || []).slice(0, 2).forEach((x, i) => {
    node(`CF${i}`, friendlyFlag(x).title || trunc(x, 40), "box", "fail");
    L.push(`TRUST --- CF${i}`);
  });

  node("REACH", `How easy to hire?<br/>${pct(d.A)}`, "box", d.A >= 0.45 ? "pass" : "warn");
  L.push("TRUST --> REACH");
  (d.availability_flags || []).slice(0, 2).forEach((x, i) => {
    node(`AF${i}`, friendlyFlag(x).title || trunc(x, 40), "box", "warn");
    L.push(`REACH --- AF${i}`);
  });

  /* 6 — final */
  const v = classifyVerdict(d);
  node("V", `Final: ${pct(d.score)} match<br/>${matchWord(d.score)}`, "box", v.cls === "green" ? "verdictG" : v.cls === "red" ? "verdictR" : "verdictA");
  L.push("REACH --> V");

  /* palette */
  L.push("classDef spine fill:#FFFFFF,stroke:#141414,stroke-width:1.4px,color:#141414");
  L.push("classDef pass fill:#EAF4EE,stroke:#16794C,stroke-width:1.4px,color:#11593A");
  L.push("classDef fail fill:#FAECEA,stroke:#B3261E,stroke-width:1.4px,color:#8C1D13");
  L.push("classDef warn fill:#F8F1DF,stroke:#946300,stroke-width:1.4px,color:#6E4A00");
  L.push("classDef verdictG fill:#16794C,stroke:#11593A,stroke-width:2px,color:#FFFFFF");
  L.push("classDef verdictR fill:#B3261E,stroke:#8C1D13,stroke-width:2px,color:#FFFFFF");
  L.push("classDef verdictA fill:#946300,stroke:#6E4A00,stroke-width:2px,color:#FFFFFF");

  return L.join("\n");
}

/* ------------------------------------------------------------
   zoom / pan canvas for the graph
   ------------------------------------------------------------ */
function enableZoom(wrap, canvas) {
  const svg = canvas.querySelector("svg");
  if (!svg) return;
  const vb = svg.viewBox.baseVal;
  const W = vb?.width || svg.getBoundingClientRect().width || 800;
  const H = vb?.height || svg.getBoundingClientRect().height || 600;
  svg.style.width = `${W}px`;
  svg.style.height = `${H}px`;
  svg.style.maxWidth = "none";

  let s = 1, tx = 0, ty = 0;
  const apply = () => { canvas.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`; };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const fit = () => {
    s = clamp(Math.min(wrap.clientWidth / W, wrap.clientHeight / H) * 0.95, 0.2, 1.6);
    tx = (wrap.clientWidth - W * s) / 2;
    ty = Math.max((wrap.clientHeight - H * s) / 2, 8);
    apply();
  };
  fit();

  const zoomAt = (mx, my, k) => {
    const ns = clamp(s * k, 0.2, 4);
    tx = mx - (mx - tx) * (ns / s);
    ty = my - (my - ty) * (ns / s);
    s = ns;
    apply();
  };

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = wrap.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, { passive: false });

  let drag = null;
  wrap.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".graph-controls")) return;
    drag = { x: e.clientX, y: e.clientY, tx, ty };
    wrap.classList.add("dragging");
    wrap.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener("pointermove", (e) => {
    if (!drag) return;
    tx = drag.tx + (e.clientX - drag.x);
    ty = drag.ty + (e.clientY - drag.y);
    apply();
  });
  const endDrag = (e) => {
    drag = null;
    wrap.classList.remove("dragging");
    if (e.pointerId != null) { try { wrap.releasePointerCapture(e.pointerId); } catch { /* noop */ } }
  };
  wrap.addEventListener("pointerup", endDrag);
  wrap.addEventListener("pointercancel", endDrag);
  wrap.addEventListener("dblclick", (e) => {
    if (e.target.closest(".graph-controls")) return;
    const r = wrap.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, 1.5);
  });

  const center = () => [wrap.clientWidth / 2, wrap.clientHeight / 2];
  $("#zoomIn").onclick = () => zoomAt(...center(), 1.3);
  $("#zoomOut").onclick = () => zoomAt(...center(), 1 / 1.3);
  $("#zoomFit").onclick = fit;
}

/* ------------------------------------------------------------
   utils
   ------------------------------------------------------------ */
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function prettify(s) { return String(s).replace(/_/g, " "); }
function titleCase(s) { s = prettify(s).toLowerCase(); return s.charAt(0).toUpperCase() + s.slice(1); }
function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
