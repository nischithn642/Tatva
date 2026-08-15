/*
 * TATVA Optimization Studio — frontend.
 *
 * One file, deliberately. The previous version split the same functions between an
 * inline <script> in index.html and this file, with this file loaded last, so its
 * definitions silently overwrote the inline ones -- including a model handler that
 * never called the Python bridge and wrote to element IDs that did not exist. Anything
 * the page does now, it does from here.
 *
 * Two rules run through the whole file:
 *   1. Nothing is loaded from the network. No CDN, no font host, no analytics.
 *   2. No number is displayed unless the backend measured it. When the bridge is
 *      absent, the UI says so; it does not fall back to a plausible-looking figure.
 */
'use strict';

/* ============================================================ the five stages
 * These are the stages promised in the product deck, in order. Sidebar, rail and
 * overview table are all generated from this one array, which is why the numbering
 * can no longer disagree with itself the way it did before (step 2, then step 4).
 */
const STAGES = [
  { id: 's1', no: '01', name: 'Input',    icon: 'i-file',    tag: 'Model + target',
    desc: 'Trained model plus target details.' },
  { id: 's2', no: '02', name: 'Analyze',  icon: 'i-scan',    tag: 'Graph + operators',
    desc: 'Graph, operators, requirements.' },
  { id: 's3', no: '03', name: 'Map',      icon: 'i-map',     tag: 'Chip capability',
    desc: 'Matched against chip capability.' },
  { id: 's4', no: '04', name: 'Optimize', icon: 'i-sliders', tag: 'Passes',
    desc: 'Memory, operators, compute.' },
  { id: 's5', no: '05', name: 'Generate', icon: 'i-box',     tag: 'Build + measure',
    desc: 'Deployable RISC-V artifacts.' },
];

const VIEW_TITLES = {
  overview: 'Overview', report: 'Benchmark Report', diag: 'Diagnostics',
  assistant: 'Assistant', scaffold: 'Project Scaffolding',
  artifacts: 'Generated Artifacts', validation: 'Validation', evidence: 'Engineering Evidence',
};
STAGES.forEach(s => { VIEW_TITLES[s.id] = `${s.no} ${s.name}`; });

/* ============================================================ state */
const S = {
  view: 'overview',
  model: null,          // { path, filename, size_mb, framework, sha256, layer_count }
  target: null,         // one entry from list_targets()
  targets: [],
  analysis: null,       // analyze_model() result
  mapping: null,        // map_operators() result
  passes: { fuse: true, quantize: false },
  result: null,         // run_pipeline() result, only when measured
  // The id the backend gave the last pipeline execution. Artifacts, Validation and
  // Evidence all fetch by this id rather than holding their own copy of the run, so
  // three pages cannot end up describing three different builds.
  runId: '',
  busy: false,
  done: { s1: false, s2: false, s3: false, s4: false, s5: false },
};

/* Every page that shows a run resets together. Leaving one of them populated after
 * the model or the target changed is how a screenshot of a stale artifact list ends
 * up attached to a different build. */
function clearRunViews() {
  S.result = null;
  S.runId = '';
  show('report-body', false); show('report-empty', true);
  show('artifacts-body', false); show('artifacts-empty', true);
  show('val-body', false); show('val-empty', true);
  show('ev-body', false); show('ev-empty', true);
  show('s5-outcome', false);
  show('btn-s5-artifacts', false);
  markDone('s5', false);
  const next = $('btn-s5-next');
  if (next) next.disabled = true;
  conReports();
}

/* ============================================================ tiny helpers */
const $ = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const show = (id, on) => { const n = $(id); if (n) n.classList.toggle('hidden', !on); };
const text = (id, v) => { const n = $(id); if (n) n.textContent = v; };
const esc = s => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const icon = (name, w) =>
  `<svg viewBox="0 0 24 24" ${w ? `width="${w}" height="${w}"` : ''} fill="none" stroke="currentColor"
        stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><use href="#${name}"/></svg>`;

/* Milliseconds, printed at a precision that suits the magnitude.
 *
 * A small model under QEMU finishes in well under a millisecond. Fixed 2-decimal
 * formatting turned a real baseline-vs-optimized gap into "0.06 ms" against
 * "0.06 ms" -- two identical numbers, and two identical-looking bars, for a run
 * that did measure a difference. Small values get more decimals; large ones get
 * fewer, so a 197 ms figure is not padded with noise digits.
 */
function fmtMs(v) {
  const n = Number(v);
  if (!isFinite(n)) return '—';
  const a = Math.abs(n);
  const d = a === 0 ? 2 : a < 0.01 ? 4 : a < 1 ? 3 : a < 100 ? 2 : 1;
  return n.toFixed(d) + ' ms';
}

/* File size in whatever unit does not read as zero. */
function fmtSize(bytes, mb) {
  const b = Number(bytes);
  if (isFinite(b) && b > 0) {
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(2) + ' MB';
  }
  return mb != null ? `${mb} MB` : '—';
}

/* The mean squared error between the RISC-V output and the host ONNX Runtime
 * reference, at three significant figures.
 *
 * The backend hands this over as a raw double, and printing it verbatim gave
 * "0.000029431902811034336" -- twenty-two characters, of which about four carry
 * information. Everything past the third digit is float noise, and at that width
 * it pushed the verdict badge off the edge of the table.
 */
function fmtMse(v) {
  const n = Number(v);
  if (!isFinite(n)) return '—';
  if (n === 0) return '0';
  const a = Math.abs(n);
  // Below a thousandth, decimal notation is all leading zeros; use exponent form.
  return a < 1e-3 || a >= 1e5 ? n.toExponential(2) : String(Number(n.toPrecision(3)));
}

/* The percentage change between the two builds, as text plus the class that
 * colours it. Exactly 0.00% is neither a win nor a regression, and painting it
 * green read as "optimization succeeded" on a run where nothing changed.
 */
function deltaText(pct) {
  const p = Number(pct);
  if (!isFinite(p)) return { text: '—', cls: '', faster: false };
  if (p === 0) return { text: '0.00%', cls: '', faster: false };
  return { text: (p > 0 ? '−' : '+') + Math.abs(p).toFixed(2) + '%', cls: p > 0 ? 'ok' : 'warn', faster: p > 0 };
}

/* Which badge colour a backend status word gets.
 *
 * The vocabulary is the backend's -- SUCCESS / PARTIAL / BLOCKED / FAILED for a run,
 * PASSED / FAILED / SKIPPED / NOT_IMPLEMENTED for a check. Mapping it in one place
 * means the studio cannot quietly paint a PARTIAL run green on one page and amber on
 * another, and nothing here invents a status the backend did not send.
 */
const BADGE_CLASS = {
  SUCCESS: 'available', PASSED: 'available', Supported: 'available',
  PARTIAL: 'experimental', Partial: 'experimental',
  BLOCKED: 'blocked', FAILED: 'blocked',
  NOT_IMPLEMENTED: 'roadmap', 'Coming Soon': 'roadmap',
  SKIPPED: 'info', NOT_RUN: 'info', RUNNING: 'info',
};
const badgeCls = s => BADGE_CLASS[s] || 'info';
/* NOT_IMPLEMENTED is shown as "Coming Soon" and nothing else is relabelled: a check
 * that failed says FAILED, in the same word the backend used. */
const statusLabel = s => (s === 'NOT_IMPLEMENTED' ? 'Coming Soon' : String(s || '—'));
const badge = (s, label) =>
  `<span class="badge ${badgeCls(s)}">${esc(label != null ? label : statusLabel(s))}</span>`;

/* The notice box shares the badge's semantics but has its own class names. */
const NOTICE_CLASS = {
  SUCCESS: 'ok', PASSED: 'ok', PARTIAL: 'warn', BLOCKED: 'err', FAILED: 'err',
  NOT_RUN: 'info', RUNNING: 'info',
};
function setNotice(id, status, message) {
  const box = $(id);
  if (!box) return;
  const kind = NOTICE_CLASS[status] || 'info';
  box.className = 'notice mt ' + kind;
  const use = box.querySelector('use');
  // The icon moves with the verdict. A tick beside "code generation did not start"
  // reads as success at a glance, whatever the sentence says.
  if (use) use.setAttribute('href', kind === 'ok' ? '#i-check' : kind === 'info' ? '#i-info' : '#i-alert');
  const span = box.querySelector('span');
  if (span) span.textContent = message || '';
}

/* The tick that appears in the corner of a selected .pick card. */
const CHECK =
  `<svg class="check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6"
        stroke-linecap="round" stroke-linejoin="round"><use href="#i-check"/></svg>`;

/* ============================================================ the bridge
 * pywebview injects window.pywebview.api. In a plain browser it is simply absent,
 * and every call below is expected to fail loudly rather than quietly pretend.
 */
const bridge = () => (window.pywebview && window.pywebview.api) || null;
const connected = () => bridge() !== null;

async function call(method, ...args) {
  const api = bridge();
  if (!api || typeof api[method] !== 'function') {
    throw new Error(
      'Compiler backend not connected. This page is being viewed outside the TATVA ' +
      'desktop app, so nothing can be compiled or measured from here.');
  }
  return await api[method](...args);
}

/* ============================================================ chrome */
function buildNav() {
  const host = $('nav-stages');
  host.innerHTML = '';
  STAGES.forEach(s => {
    const b = el('button', 'nav-item');
    b.id = `nav-${s.id}`;
    b.dataset.view = s.id;
    b.innerHTML = `<span class="step-no">${s.no}</span>${esc(s.name)}`;
    host.appendChild(b);
  });
}

function buildRail() {
  const host = $('rail');
  host.innerHTML = '';
  STAGES.forEach(s => {
    const b = el('button', 'rail-step');
    b.id = `rail-${s.id}`;
    b.dataset.view = s.id;
    b.title = s.desc;
    b.innerHTML =
      `<span class="n">${s.no}</span>` +
      `<span class="t"><b>${esc(s.name)}</b><i>${esc(s.tag)}</i></span>`;
    host.appendChild(b);
  });
}

function buildOverviewTable() {
  const tb = $('ov-stage-table');
  tb.innerHTML = '';
  STAGES.forEach(s => {
    const tr = el('tr');
    tr.innerHTML =
      `<td class="mono gold">${s.no}</td>` +
      `<td><b>${esc(s.name)}</b></td>` +
      `<td class="muted">${esc(s.desc)}</td>` +
      `<td id="ovs-${s.id}"><span class="badge">Pending</span></td>`;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => switchView(s.id));
    tb.appendChild(tr);
  });
}

function switchView(view) {
  S.view = view;
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  const target = $(`view-${view}`);
  if (target) target.classList.remove('hidden');

  document.querySelectorAll('.nav-item').forEach(n =>
    n.classList.toggle('active', n.dataset.view === view));
  document.querySelectorAll('.rail-step').forEach(n =>
    n.classList.toggle('active', n.dataset.view === view));

  text('crumb-view', VIEW_TITLES[view] || view);
  const scroller = document.querySelector('.view-scroll');
  if (scroller) scroller.scrollTop = 0;

  if (view === 's4') renderPlan();
  if (view === 'report') requestAnimationFrame(drawChart);
  if (view === 'diag') loadHealth();
  // Fetched on arrival rather than cached in the page. These read a finished run out
  // of the backend registry, so a file deleted since the build shows as gone instead
  // of lingering in a list the frontend is holding.
  if (view === 'artifacts') loadArtifacts();
  if (view === 'validation') loadValidation();
  if (view === 'evidence') loadEvidence();
}

/* Marks a stage complete and refreshes every place that shows stage progress.
 * Doing it in one function is the point: three separate progress indicators that
 * each maintained their own idea of "done" is how a stage went missing before. */
function markDone(stage, done) {
  S.done[stage] = done;
  STAGES.forEach(s => {
    const nav = $(`nav-${s.id}`);
    if (nav) nav.classList.toggle('done', !!S.done[s.id]);
    const rail = $(`rail-${s.id}`);
    if (rail) rail.classList.toggle('done', !!S.done[s.id]);
    const cell = $(`ovs-${s.id}`);
    if (cell) {
      cell.innerHTML = S.done[s.id]
        ? '<span class="badge available">Done</span>'
        : '<span class="badge">Pending</span>';
    }
  });

  const next = STAGES.find(s => !S.done[s.id]) || STAGES[STAGES.length - 1];
  text('ov-stage', next.no);
  text('ov-stage-n', next.name);
}

/* ============================================================ theme */
function applyTheme(mode) {
  document.documentElement.className = mode;
  try { localStorage.setItem('tatva-theme', mode); } catch (_) { /* private mode */ }
  const dark = mode === 'dark';
  $('btn-theme').innerHTML = icon(dark ? 'i-sun' : 'i-moon');
  $('brand-logo').src = dark ? 'assets/logo-dark.png' : 'assets/logo-light.png';
  if (S.result) requestAnimationFrame(drawChart);   // canvas colours are theme-bound
}

/* ============================================================ console dock
 *
 * The dock is a view onto output the app already produces, not a second source of it.
 * Every line in Log is a line term() printed; every row in Messages is one of those
 * lines that carried a severity class; every row in Reports is a file the backend found
 * on disk. Nothing here is generated for the sake of filling a panel, so a session that
 * has run nothing shows three empty tabs, which is the truth.
 */

/* term() tags lines with a colour class. Those classes are the only severity signal in
 * the codebase, so the mapping is read off them rather than by pattern-matching text --
 * guessing at severity from wording would invent classifications the app never made. */
const SEV_OF = { 'c-err': 'ERROR', 'c-warn': 'WARNING', 'c-ok': 'INFO', 'c-gold': 'INFO' };
const CON = { msgs: 0, err: 0, warn: 0, log: 0 };

function conCount(id, n, cls) {
  const e = $(id);
  if (!e) return;
  e.textContent = String(n);
  e.className = 'ctab-count' + (cls || '');
}

function conSync() {
  conCount('cnt-log', CON.log);
  conCount('cnt-messages', CON.msgs,
    CON.err ? ' has-err' : CON.warn ? ' has-warn' : '');
  show('cmsg-empty', CON.msgs === 0);
}

/* Mirror one term() line into the dock. */
function conLine(line, cls) {
  const log = $('console-log');
  if (log) {
    log.appendChild(el('div', cls || '', esc(line)));
    log.scrollTop = log.scrollHeight;
    CON.log++;
  }

  const sev = SEV_OF[cls];
  const body = String(line == null ? '' : line).trim();
  // A blank spacer line carries a class sometimes; it is not a message.
  if (!sev || !body) { conSync(); return; }

  const list = $('cmsg-list');
  if (list) {
    const row = el('div', 'cmsg ' + sev.toLowerCase());
    row.innerHTML = `<span class="sev">${sev}</span><span class="txt">${esc(body)}</span>`;
    list.appendChild(row);
    CON.msgs++;
    if (sev === 'ERROR') CON.err++;
    else if (sev === 'WARNING') CON.warn++;
  }
  conSync();
}

function conClear() {
  const log = $('console-log'); if (log) log.innerHTML = '';
  const list = $('cmsg-list'); if (list) list.innerHTML = '';
  CON.msgs = CON.err = CON.warn = CON.log = 0;
  conSync();
}

/* Reports is the artifact inventory the backend reads off disk -- the same call the
 * Artifacts page makes. If a run wrote nothing, this stays empty rather than listing
 * filenames a build is "supposed" to produce. */
async function conReports() {
  const list = $('creport-list');
  if (!list) return;
  list.innerHTML = '';
  let n = 0;

  if (S.runId) {
    let r;
    try { r = await call('get_artifacts', S.runId); }
    catch (_) { r = null; }
    (r && r.success ? r.builds || [] : []).forEach(b => {
      (b.artifacts || []).forEach(a => {
        const row = el('div', 'cmsg info');
        row.innerHTML =
          `<span class="sev">${esc(b.config)}</span>` +
          `<span class="txt mono">${esc(a.name)}</span>` +
          `<span class="at">${esc(a.stage)} · ${fmtSize(a.size_bytes)}</span>`;
        row.title = 'Open in Generated Artifacts';
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => switchView('artifacts'));
        list.appendChild(row);
        n++;
      });
    });
  }

  conCount('cnt-reports', n);
  show('creport-empty', n === 0);
}

function wireConsole() {
  document.querySelectorAll('.console-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.console-tab').forEach(t => {
        const on = t === tab;
        t.classList.toggle('active', on);
        t.setAttribute('aria-selected', String(on));
      });
      document.querySelectorAll('.console-pane').forEach(p => {
        p.classList.toggle('active', p.id === 'cpane-' + tab.dataset.cpane);
      });
      // Opening the tab is the moment to re-read the disk, so a file deleted since the
      // build does not linger in the list.
      if (tab.dataset.cpane === 'reports') conReports();
    });
  });

  const clear = $('console-clear');
  if (clear) clear.addEventListener('click', conClear);

  const col = $('console-collapse');
  if (col) {
    col.addEventListener('click', () => {
      const dock = $('console');
      const now = dock.classList.toggle('collapsed');
      col.setAttribute('aria-expanded', String(!now));
      col.innerHTML = now ? '&#9652;' : '&#9662;';
      col.title = now ? 'Expand the console' : 'Collapse the console';
    });
  }
}

/* ============================================================ terminal */
function term(line, cls) {
  conLine(line, cls);
  const t = $('terminal');
  if (!t) return;
  const d = el('div', cls || '', esc(line));
  t.appendChild(d);
  t.scrollTop = t.scrollHeight;
}
/* Clearing a stage's own terminal does not clear the dock: the dock is the transcript of
 * the whole session, and stage 05 clears its pane at the start of every run. */
const termClear = () => { const t = $('terminal'); if (t) t.innerHTML = ''; };

// The beta scope calls plain-English diagnostics the key differentiator: on failure the
// user should get a cause and a recommendation, not a raw compiler error. The backend
// runs the rule engine and returns the prose; this prints it under its own heading so it
// reads as an explanation rather than more error output. Numbered mitigation steps are
// indented so the block scans as a list in a fixed-width terminal.
function termDiagnosis(text) {
  const body = String(text || '').trim();
  if (!body) return;
  term('WHAT THIS MEANS', 'c-gold');
  body.split('\n').forEach(line => {
    const l = line.trimEnd();
    if (!l) { term(''); return; }
    term(/^\d+\./.test(l.trim()) ? '  ' + l.trim() : l, 'c-dim');
  });
  term('');
}

/* ============================================================ STAGE 01 — input */
async function loadTargets() {
  const grid = $('target-grid');
  let list = [];
  try {
    list = await call('list_targets');
  } catch (_) {
    grid.innerHTML =
      `<div class="notice warn" style="grid-column:1/-1">${icon('i-alert')}
        <span>Target list unavailable — the compiler backend is not connected.</span></div>`;
    return;
  }
  if (!list || !list.length) {
    grid.innerHTML =
      `<div class="notice err" style="grid-column:1/-1">${icon('i-alert')}
        <span>The compiler returned no targets. Reinstall may be incomplete.</span></div>`;
    return;
  }

  S.targets = list;
  grid.innerHTML = '';
  list.forEach(t => {
    const b = el('button', 'pick');
    b.dataset.target = t.name;
    b.innerHTML =
      CHECK +
      `<div class="row" style="gap:8px">
         <span class="name">${esc(t.name)}</span>
         ${t.experimental ? '<span class="badge experimental">Experimental</span>'
                          : '<span class="badge available">Available</span>'}
         ${t.default ? '<span class="badge info">Default</span>' : ''}
       </div>
       <div class="flags">${esc(t.march)} · ${esc(t.mabi)} · ${t.bitness}-bit</div>
       ${t.notes ? `<div class="desc">${esc(t.notes)}</div>` : ''}`;
    b.addEventListener('click', () => selectTarget(t.name));
    grid.appendChild(b);
  });

  const def = list.find(t => t.default) || list[0];
  selectTarget(def.name);
}

function selectTarget(name) {
  const t = S.targets.find(x => x.name === name);
  if (!t) return;
  S.target = t;

  document.querySelectorAll('#target-grid .pick').forEach(p =>
    p.classList.toggle('selected', p.dataset.target === name));

  text('ov-target', t.name);
  text('ov-target-n', `${t.march} · ${t.bitness}-bit`);

  const warn = t.experimental
    ? `${t.name} is experimental. ${t.notes || ''}`.trim()
    : '';
  show('target-warn', !!warn);
  if (warn) text('target-warn-text', warn);

  // The chip changed, so the mapping and the measurement no longer describe
  // what is selected. Invalidating is safer than leaving a stale verdict on screen.
  if (S.mapping) { S.mapping = null; show('s3-result', false); show('s3-empty', true); markDone('s3', false); }
  if (S.result) clearRunViews();
  // The capability table is per-target, so the one on screen now belongs to the chip
  // that was selected a moment ago.
  show('s3-caps', false);
  text('s3-cap-target', t.name);
  $('btn-s3-next').disabled = true;
  $('btn-s5-next').disabled = true;
  renderPlan();
}

/* ---- model formats (§17-§18) ------------------------------------------------
 * The registry, rendered. Nothing here is a claim the page makes on its own: the
 * status of every row is computed in the backend against the packages actually
 * importable in this install, so a format cannot be advertised as working because
 * someone typed "Supported" into the HTML.
 */
async function loadFormats() {
  const tb = $('s1-formats');
  if (!tb) return;
  let r;
  try { r = await call('get_model_formats'); }
  catch (e) {
    tb.innerHTML = `<tr><td colspan="3" class="muted">${esc(e.message)}</td></tr>`;
    text('s1-formats-note', 'The backend is not connected, so format availability cannot be checked here.');
    return;
  }
  const formats = (r && r.formats) || [];
  if (!formats.length) {
    tb.innerHTML = `<tr><td colspan="3" class="muted">${esc((r && r.error) || 'The format registry returned nothing.')}</td></tr>`;
    return;
  }
  tb.innerHTML = '';
  formats.forEach(f => {
    const tr = el('tr');
    tr.innerHTML =
      `<td><b>${esc(f.label)}</b><div class="tiny muted mono">${esc(f.extensions_label)}</div></td>` +
      `<td>${badge(f.status)}</td>` +
      `<td class="muted">${esc(f.reason || f.notes || '')}</td>`;
    tb.appendChild(tr);
  });
  const usable = formats.filter(f => f.status === 'Supported').map(f => f.label);
  text('s1-formats-note', usable.length
    ? `Compilable in this install: ${usable.join(', ')}. Everything else is listed so you can see `
      + 'exactly what is missing rather than discovering it at stage 05.'
    : 'No format is fully available in this install. Check Diagnostics.');
}

/* ---- what the selected model actually is ------------------------------------
 * Read out of the file by the frontend adapter. The family line is allowed to say
 * "Unable to determine" -- a confident wrong answer about the model someone just
 * loaded is worse than no answer.
 */
async function loadModelDetail(path) {
  const tb = $('s1-detail-table');
  if (!tb) return;
  show('s1-detail', false);

  let info;
  try { info = await call('import_model_info', path); }
  catch (_) { return; }               // the summary above is already populated
  if (!info || !info.ok) return;

  const rows = [];
  const fam = info.family || {};
  if (fam.family) {
    const conf = fam.confidence && fam.confidence !== 'none' ? ` (${fam.confidence} confidence)` : '';
    const signals = (fam.signals || []).length ? `<div class="tiny muted">From: ${esc(fam.signals.join(', '))}</div>` : '';
    rows.push(['Model family', `${esc(fam.family)}${esc(conf)}${signals}`]);
  }
  if (info.framework) rows.push(['Framework', esc(info.framework) + ' · ' + esc(info.format)]);
  if (info.parameter_count) {
    rows.push(['Parameters', `${info.parameter_count.toLocaleString()} <span class="muted">(${fmtSize(info.parameter_bytes)} of weights)</span>`]);
  }
  if (info.precision) rows.push(['Precision', esc(info.precision)]);
  if (info.op_count) rows.push(['Operators', `${info.op_count} calls · ${info.distinct_op_count} distinct kinds`]);

  const shapeRow = list => (list || []).map(v =>
    `<div class="mono tiny">${esc(v.name)} <span class="muted">${esc(v.shape_label)} ${esc(v.dtype || '')}</span></div>`).join('');
  if ((info.inputs || []).length) rows.push(['Inputs', shapeRow(info.inputs)]);
  if ((info.outputs || []).length) rows.push(['Outputs', shapeRow(info.outputs)]);
  if (info.has_dynamic_shapes) {
    rows.push(['Dynamic shapes',
      '<span class="badge experimental">Present</span> <span class="muted">Symbolic dimensions are '
      + 'resolved at import; a shape that stays symbolic will stop code generation.</span>']);
  }
  if (info.opset) rows.push(['ONNX opset', esc(info.opset)]);
  if (info.producer) rows.push(['Exported by', esc(info.producer)]);

  if (!rows.length) return;
  tb.innerHTML = '';
  rows.forEach(([k, v]) => {
    const tr = el('tr');
    tr.innerHTML = `<td><b>${esc(k)}</b></td><td>${v}</td>`;
    tb.appendChild(tr);
  });
  show('s1-detail', true);
}

async function browseModel() {
  try {
    const res = await call('select_model_file');
    const path = (res && res.path) || '';
    if (path) { $('model-path').value = path; await validateModel(path); }
  } catch (e) { stageError('s1', e.message); }
}

async function loadSamples() {
  let list = [];
  try { list = await call('list_sample_models'); } catch (e) { stageError('s1', e.message); return; }
  if (!list || !list.length) { stageError('s1', 'No sample models are bundled with this install.'); return; }

  switchView('s1');

  // Show the choice rather than making it. The previous version loaded list[0] and
  // said nothing, so the transformer fixture -- the one model here with an attention
  // pattern, and so the only one where softmax fusion can change anything -- was
  // unreachable from the UI. A run on the MLP reports 0.00%, correctly, and that
  // looked like the optimizer was broken.
  const grid = $('s1-sample-grid');
  grid.innerHTML = '';
  list.forEach(s => {
    const card = el('button', 'pick');
    card.innerHTML =
      `<div class="name">${esc(s.label || s.name)}</div>` +
      `<div class="desc">${esc(s.note || '')}</div>` +
      `<div class="flags">${esc(s.name)} · ${s.size_kb} KB</div>` + CHECK;
    card.addEventListener('click', async () => {
      grid.querySelectorAll('.pick').forEach(n => n.classList.remove('selected'));
      card.classList.add('selected');
      $('model-path').value = s.path;
      await validateModel(s.path);
    });
    grid.appendChild(card);
  });
  show('s1-samples', true);

  // Load the first one anyway so the stage is not left empty; the cards above make
  // it obvious that it is a default and not the only option.
  if (!S.model) {
    grid.firstChild.click();
  }
}

// The second argument is the backend's plain-English diagnosis, which the beta scope
// lists as the key differentiator. When present it goes under the raw message rather
// than replacing it: the raw text is what a user pastes into a search or an issue, the
// diagnosis is what tells them what to do. The block is rebuilt on every call so a
// second failure never shows the previous failure's advice.
function stageError(stage, msg, diagnosis) {
  show(`${stage}-error`, true);
  text(`${stage}-error-text`, msg);

  const host = $(`${stage}-error`);
  if (!host) return;
  const old = host.querySelector('.diag-body');
  if (old) old.remove();

  const body = String(diagnosis || '').trim();
  if (!body) return;
  const d = el('div', 'diag-body tiny');
  d.style.cssText = 'margin-top:8px;white-space:pre-wrap;opacity:.85';
  d.textContent = body;
  host.querySelector(`#${stage}-error-text`).appendChild(d);
}
function clearStageError(stage) { show(`${stage}-error`, false); }

async function validateModel(path) {
  clearStageError('s1');
  if (!path) return;

  let v;
  try { v = await call('validate_model_file', path); }
  catch (e) { stageError('s1', e.message); return; }

  if (!v || !v.valid) {
    stageError('s1', (v && v.error) || 'That file could not be read as a model.');
    show('s1-summary', false); show('s1-empty', true);
    S.model = null; markDone('s1', false);
    $('btn-s1-next').disabled = true;
    return;
  }

  S.model = Object.assign({ path }, v);
  show('s1-empty', false);
  show('s1-summary', true);

  text('s1-name', v.filename || path.split(/[\\/]/).pop());
  text('s1-format', v.framework || 'ONNX');
  text('s1-size', fmtSize(v.size_bytes, v.size_mb));
  text('s1-ops', v.layer_count != null ? String(v.layer_count) : '—');
  text('s1-hash', v.sha256 ? String(v.sha256).slice(0, 12) : '—');

  text('ov-model', v.filename || '—');
  text('ov-model-n', `${fmtSize(v.size_bytes, v.size_mb)} · ${v.framework || 'ONNX'}`);

  markDone('s1', true);
  $('btn-s1-next').disabled = false;

  // A new model invalidates everything downstream of it.
  S.analysis = null; S.mapping = null;
  show('s2-result', false); show('s2-empty', true); markDone('s2', false);
  show('s3-result', false); show('s3-empty', true); markDone('s3', false);
  clearRunViews();
  $('btn-s2-next').disabled = true;
  $('btn-s3-next').disabled = true;
  renderPlan();

  // Deeper detail than validate_model_file returns: parameter count, shapes, opset,
  // family. Fired after the stage is already usable, so a slow read cannot delay the
  // Continue button.
  loadModelDetail(path);
}

/* ============================================================ STAGE 02 — analyze */
async function runAnalyze() {
  if (!S.model) { switchView('s1'); stageError('s1', 'Load a model first.'); return; }
  clearStageError('s2');
  const btn = $('btn-analyze');
  btn.disabled = true; btn.classList.add('loading');
  show('s2-empty', true); show('s2-result', false);

  try {
    const a = await call('analyze_model', S.model.path);
    if (a && a.error) throw Object.assign(new Error(a.error), { diagnosis: a.diagnosis });
    S.analysis = a;

    const hist = a.op_histogram || {};
    const entries = Object.entries(hist).sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]));

    text('s2-total', a.total_ops != null ? String(a.total_ops) : '—');
    text('s2-distinct', String(entries.length));
    text('s2-bottleneck', a.has_transformer_bottleneck ? 'Found' : 'None');
    $('s2-bottleneck').className = 'v ' + (a.has_transformer_bottleneck ? 'gold' : '');

    text('s2-note-text', a.has_transformer_bottleneck
      ? 'A softmax feeding a matrix multiply was found. That is the attention pattern, '
        + 'and it is exactly what the fusion pass in stage 04 rewrites — so this model '
        + 'should show a measurable gain.'
      : 'No attention pattern in this graph. The softmax fusion pass will still run, but '
        + 'with little to fuse the measured difference is likely to be small.');

    const host = $('s2-histogram');
    host.innerHTML = '';
    const max = entries.length ? entries[0][1] : 1;
    entries.forEach(([op, n]) => {
      const row = el('div', 'hbar');
      row.innerHTML =
        `<span class="hbar-l mono">${esc(op)}</span>
         <span class="hbar-t"><i style="width:${Math.max(2, (n / max) * 100).toFixed(1)}%"></i></span>
         <span class="hbar-n mono">${n}</span>`;
      host.appendChild(row);
    });

    show('s2-empty', false); show('s2-result', true);
    markDone('s2', true);
    $('btn-s2-next').disabled = false;
  } catch (e) {
    stageError('s2', e.message, e.diagnosis);
    markDone('s2', false);
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

/* ============================================================ STAGE 03 — map */
async function runMap() {
  if (!S.model) { switchView('s1'); stageError('s1', 'Load a model first.'); return; }
  if (!S.target) { switchView('s1'); stageError('s1', 'Pick a target first.'); return; }
  clearStageError('s3');
  const btn = $('btn-map');
  btn.disabled = true; btn.classList.add('loading');
  show('s3-empty', true); show('s3-result', false);

  try {
    const m = await call('map_operators', S.model.path, S.target.name);
    if (!m || !m.success) {
      throw Object.assign(new Error((m && m.error) || 'Mapping failed.'), { diagnosis: m && m.diagnosis });
    }
    S.mapping = m;

    text('s3-target', `${m.target} (${m.march})`);

    const verdict = $('s3-verdict');
    verdict.className = 'notice ' + (m.ready ? 'ok' : 'err');
    // The icon has to move with the verdict. A green tick sitting next to "compilation
    // will stop at code generation" reads as success at a glance, which is the opposite
    // of what the sentence says.
    const vicon = verdict.querySelector('use');
    if (vicon) vicon.setAttribute('href', m.ready ? '#i-check' : '#i-alert');
    // "Stage 05 can compile this model" overpromised: mapping proves operator coverage,
    // not that a cross-compiler is installed. Stage 05 does its own preflight for that.
    text('s3-verdict-text', m.ready
      ? `All ${m.distinct_ops} operator kinds in this graph have a lowering on ${m.target}. `
        + 'Nothing in this graph will block stage 05.'
      : `${m.unsupported.length} operator kind(s) have no lowering on ${m.target}: `
        + `${m.unsupported.join(', ')}. Compilation will stop at code generation until `
        + 'these are removed from the model or added to TATVA.');

    const warns = $('s3-warnings');
    warns.innerHTML = '';
    (m.warnings || []).forEach(w => {
      warns.appendChild(el('div', 'notice warn', `${icon('i-alert')}<span>${esc(w)}</span>`));
    });

    renderMapTable(m);
    renderFixOffer(m);

    show('s3-empty', false); show('s3-result', true);
    markDone('s3', true);
    $('btn-s3-next').disabled = false;
  } catch (e) {
    stageError('s3', e.message, e.diagnosis);
    markDone('s3', false);
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

/* §5: the mapping table as a diagnostic rather than a checklist.
 *
 * The old four-column version said an operator was unsupported and stopped there, which
 * left the one question that matters unanswered: what do I do about it. Every column
 * here comes from the capability database -- none of it is inferred in the page. */
function renderMapTable(m) {
  const tb = $('s3-table');
  tb.innerHTML = '';
  (m.operators || []).forEach(o => {
    const kindBadge = !o.supported ? badge('BLOCKED', 'No lowering')
      : o.kind === 'fused' ? badge('SUCCESS', 'Fused')
      : o.kind === 'hot' ? badge('PARTIAL', 'Hot path')
      : badge('SUCCESS', 'Mapped');

    const support = o.supported
      ? `<span class="t-ok">Lowering exists</span>`
      : `<span class="t-err">None on ${esc(m.target)}</span>`;
    const constraints = (o.constraints || []).length
      ? `<div class="tiny muted">${esc(o.constraints.join('; '))}</div>` : '';

    const fix = o.auto_fix_available
      ? badge('SUCCESS', 'Rewrite available') +
        (o.auto_fix_summary ? `<div class="tiny muted">${esc(o.auto_fix_summary)}</div>` : '')
      : o.supported ? '<span class="muted">Not needed</span>'
      : badge('BLOCKED', 'No rewrite') +
        '<div class="tiny muted">TATVA has no validated decomposition for this operator.</div>';

    const tr = el('tr');
    tr.innerHTML =
      `<td class="mono">${esc(o.op)}</td>` +
      `<td class="num mono">${o.count}</td>` +
      `<td>${badge(o.status === 'MAPPED' ? 'SUCCESS' : 'BLOCKED', o.status)}</td>` +
      `<td>${support}${constraints}</td>` +
      `<td class="muted">${esc(o.lowering)}` +
        (o.optimization ? `<div class="tiny gold">${esc(o.optimization)}</div>` : '') + `</td>` +
      `<td class="muted">${esc(o.reason || '—')}</td>` +
      `<td class="muted">${esc(o.impact || '—')}</td>` +
      `<td>${fix}</td>`;
    tb.appendChild(tr);
  });
}

/* The Attempt Auto-Fix control exists only when the repair engine actually has a rule
 * for something in this graph (§33 -- no button that does nothing). When operators are
 * unmapped and none is repairable, the box still appears, says so, and carries no
 * button. */
function renderFixOffer(m) {
  const unsupported = m.unsupported || [];
  const fixable = m.fixable || [];
  const blocking = m.blocking || [];
  show('s3-fix-result', false);
  $('s3-fix-result').innerHTML = '';

  if (!unsupported.length) { show('s3-fix-offer', false); return; }

  show('s3-fix-offer', true);
  const parts = [];
  if (fixable.length) {
    parts.push(`${fixable.join(', ')} ${fixable.length === 1 ? 'has' : 'have'} no lowering on `
      + `${m.target}, but TATVA has an exact rewrite for ${fixable.length === 1 ? 'it' : 'them'}. `
      + 'The rewrite is checked structurally and by running both graphs on the host before it is kept.');
  }
  if (blocking.length) {
    parts.push(`${blocking.join(', ')} ${blocking.length === 1 ? 'has' : 'have'} no lowering and no `
      + 'validated rewrite, so code generation will stop there.');
  }
  text('s3-fix-text', parts.join(' '));
  show('btn-autofix', fixable.length > 0);
}

async function runAutoFix() {
  if (!S.model || !S.target) return;
  const btn = $('btn-autofix');
  btn.disabled = true; btn.classList.add('loading');

  let r;
  try { r = await call('attempt_auto_fix', S.model.path, S.target.name); }
  catch (e) { r = { success: false, error: e.message }; }
  finally { btn.disabled = false; btn.classList.remove('loading'); }

  const host = $('s3-fix-result');
  host.innerHTML = '';
  show('s3-fix-result', true);

  if (!r || !r.success) {
    host.appendChild(el('div', 'notice err',
      `${icon('i-alert')}<span>${esc((r && r.error) || 'The repair engine did not run.')}</span>`));
    return;
  }

  // REPAIRED | PARTIAL | BLOCKED | NO_REPAIR_NEEDED | DISCARDED, straight from the
  // engine. DISCARDED is the interesting one: a rewrite was produced and then thrown
  // away because it failed validation, and that has to read as a non-event, not a win.
  const kind = r.status === 'REPAIRED' ? 'ok'
    : r.status === 'PARTIAL' ? 'warn'
    : r.status === 'NO_REPAIR_NEEDED' ? 'info' : 'err';
  const ico = kind === 'ok' ? 'i-check' : kind === 'info' ? 'i-info' : 'i-alert';
  host.appendChild(el('div', 'notice ' + kind,
    `${icon(ico)}<span><b>${esc(r.status)}</b> — ${esc(r.message || '')}</span>`));

  const records = r.records || [];
  if (records.length) {
    const card = el('div', 'card');
    card.innerHTML =
      '<h2>What was rewritten</h2>' +
      '<p class="sub">Each rewrite with the identity it relies on and the checks that accepted it.</p>' +
      '<div class="table-wrap"><table><thead><tr>' +
      '<th style="width:140px">Operator</th><th class="num" style="width:64px">Sites</th>' +
      '<th style="width:200px">Rewritten to</th><th>Why this is the same function</th>' +
      '<th style="width:190px">Checks</th>' +
      '</tr></thead><tbody></tbody></table></div>';
    const body = card.querySelector('tbody');
    records.forEach(rec => {
      const checks =
        `<div>${esc(rec.structural_validation)} <span class="muted">structural</span></div>` +
        `<div>${esc(rec.numerical_validation)} <span class="muted">numerical</span></div>` +
        (rec.max_abs_diff != null
          ? `<div class="tiny muted mono">max |Δ| ${fmtMse(rec.max_abs_diff)}</div>` : '');
      const tr = el('tr');
      tr.innerHTML =
        `<td class="mono">${esc(rec.original_op)}</td>` +
        `<td class="num mono">${rec.occurrences}</td>` +
        `<td class="mono tiny">${esc((rec.replacement_ops || []).join(' → ') || '—')}</td>` +
        `<td class="muted">${esc(rec.identity || rec.reason || '')}</td>` +
        `<td class="tiny">${checks}</td>`;
      body.appendChild(tr);
    });
    host.appendChild(card);
  }

  // Re-map so the table shows the graph as it now stands. The repaired module is held
  // in the backend against this model and target, and the pipeline compiles that one --
  // so the table and the build agree.
  if (r.applied) await runMap();
}

/* §22: the chip's own operator table, with no model involved. */
async function loadCapabilities() {
  if (!S.target) return;
  const panel = $('s3-caps');
  const open = !panel.classList.contains('hidden');
  if (open) { show('s3-caps', false); text('btn-caps', 'Show capabilities'); return; }

  const tb = $('s3-caps-table');
  tb.innerHTML = `<tr><td colspan="4" class="muted">Reading the capability database…</td></tr>`;
  show('s3-caps', true);
  $('btn-caps').textContent = 'Hide capabilities';

  let r;
  try { r = await call('get_target_capabilities', S.target.name); }
  catch (e) { r = { success: false, error: e.message }; }

  if (!r || !r.success) {
    tb.innerHTML = `<tr><td colspan="4" class="muted">${esc((r && r.error) || 'Unavailable.')}</td></tr>`;
    return;
  }
  tb.innerHTML = '';
  (r.operators || []).forEach(o => {
    const tr = el('tr');
    tr.innerHTML =
      `<td class="mono">${esc(o.op)}</td>` +
      `<td>${badge(o.supported ? 'SUCCESS' : 'BLOCKED', o.status)}</td>` +
      `<td class="muted">${esc(o.lowering)}</td>` +
      `<td>${o.auto_fix_available ? badge('SUCCESS', 'Rewrite available') : '<span class="muted">—</span>'}</td>`;
    tb.appendChild(tr);
  });
  const repairable = (r.repairable || []).map(x => x.op);
  text('s3-caps-note',
    `${(r.operators || []).length} operators have a lowering on ${r.target} (${r.march} / ${r.mabi}). `
    + `${repairable.length} further operator kind(s) can be rewritten into this set: `
    + `${repairable.join(', ') || 'none'}. Anything outside both lists stops code generation.`);
}

/* ============================================================ STAGE 04 — optimize */
function togglePass(name) {
  S.passes[name] = !S.passes[name];
  $(name === 'fuse' ? 'pass-fuse' : 'pass-quant').classList.toggle('selected', S.passes[name]);
  show('s4-quant-warn', S.passes.quantize);
  markDone('s4', true);
  if (S.result) { S.result = null; show('report-body', false); show('report-empty', true); markDone('s5', false); $('btn-s5-next').disabled = true; }
  renderPlan();
}

// Apply a pass state directly rather than through togglePass, which flips. Shares the
// invalidation logic: any change to the passes makes an existing measurement describe a
// build that is no longer selected.
function setPass(name, on) {
  if (S.passes[name] === on) return false;
  S.passes[name] = on;
  $(name === 'fuse' ? 'pass-fuse' : 'pass-quant').classList.toggle('selected', on);
  return true;
}

/* ---- natural-language configuration ---------------------------------------
 * Listed as experimental in the beta scope, and the honest reading of that word here
 * is: it is a phrase matcher, not a language model. It runs offline, it is
 * deterministic, and it can only set the same two switches and the target that the
 * cards on this page already set. It shows its reasoning for exactly that reason --
 * the user has to be able to see the mapping and overrule it before a build.
 */
async function applyNlConfig(raw) {
  const input = $('nl-input');
  const textIn = String(raw != null ? raw : (input ? input.value : '')).trim();
  if (!textIn) { show('nl-result', false); return; }

  const btn = $('btn-nl');
  if (btn) { btn.disabled = true; btn.classList.add('loading'); }

  let r;
  try { r = await call('interpret_config', textIn, S.target ? S.target.name : '', S.passes.fuse, S.passes.quantize); }
  catch (e) { r = { success: false, error: e.message }; }
  finally { if (btn) { btn.disabled = false; btn.classList.remove('loading'); } }

  const box = $('nl-result');
  const out = $('nl-result-text');
  if (!box || !out) return;
  show('nl-result', true);
  out.innerHTML = '';

  if (!r || !r.success) {
    box.className = 'notice err mt';
    out.textContent = (r && r.error) || 'That could not be read as a configuration.';
    return;
  }

  if (!r.matched) {
    // Saying nothing changed is more useful than silently leaving the switches alone
    // and letting the user assume the sentence was understood.
    box.className = 'notice warn mt';
    out.textContent = 'Nothing in that matched a known priority, target or pass, so the '
      + 'settings below are unchanged. Try naming a target (RV32IMC, RV64GC), a pass '
      + '(fusion, INT8), or what you are optimising for (size, latency, accuracy).';
    return;
  }

  let changed = false;
  if (r.target && S.target && r.target !== S.target.name) { selectTarget(r.target); changed = true; }
  if (setPass('fuse', !!r.fuse)) changed = true;
  if (setPass('quantize', !!r.quantize)) changed = true;

  show('s4-quant-warn', S.passes.quantize);
  markDone('s4', true);
  if (changed && S.result) {
    S.result = null;
    show('report-body', false); show('report-empty', true);
    markDone('s5', false); $('btn-s5-next').disabled = true;
  }
  renderPlan();

  box.className = 'notice ' + (r.conflicts && r.conflicts.length ? 'warn' : 'ok') + ' mt';
  const head = el('div');
  head.innerHTML = `<b>${esc(r.summary || '')}</b>`;
  out.appendChild(head);
  (r.reasons || []).concat(r.conflicts || []).forEach(line => {
    const d = el('div', 'tiny');
    d.style.cssText = 'margin-top:5px;opacity:.85';
    d.textContent = '· ' + line;
    out.appendChild(d);
  });
  const foot = el('div', 'tiny');
  foot.style.cssText = 'margin-top:7px;opacity:.7';
  foot.textContent = 'Change any of it with the cards below — this only set the switches.';
  out.appendChild(foot);
}

function renderPlan() {
  const p = $('s4-plan');
  if (!p) return;
  const t = S.target;
  const on = [];
  if (S.passes.fuse) on.push('softmax fusion');
  if (S.passes.quantize) on.push('INT8 quantization');

  const lines = [
    ['c-head', '# Stage 05 will run:'],
    ['c-dim', ''],
    ['', `model    ${S.model ? S.model.path : '(none selected — go back to stage 01)'}`],
    ['', `target   ${t ? `${t.name}  ${t.march} / ${t.mabi}` : '(none selected)'}`],
    ['', `passes   ${on.length ? on.join(' + ') : 'none — baseline only, nothing to compare against'}`],
    ['c-dim', ''],
    ['c-dim', '1. build baseline    TVM Relax -> C99 -> riscv-none-elf-gcc'],
    ['c-dim', '2. measure baseline  QEMU system mode, rdcycle on target'],
  ];
  if (on.length) {
    lines.push(['c-dim', '3. build optimized   same path, with the passes above applied']);
    lines.push(['c-dim', '4. measure optimized QEMU system mode, rdcycle on target']);
    lines.push(['c-dim', '5. compare outputs   against host ONNX Runtime, MSE tolerance check']);
  } else {
    lines.push(['c-warn', '   no optimized build — with no passes there is nothing to compare']);
  }

  p.innerHTML = '';
  lines.forEach(([cls, txt]) => p.appendChild(el('div', cls, esc(txt) || '&nbsp;')));
}

/* ============================================================ STAGE 05 — generate */
async function runBuild() {
  if (!S.model) { switchView('s1'); stageError('s1', 'Load a model first.'); return; }
  if (!S.target) { switchView('s1'); stageError('s1', 'Pick a target first.'); return; }
  if (S.busy) return;

  // Warn, but do not block: an unmapped operator is a compile failure, and the user
  // is entitled to see the real error if they want it.
  if (S.mapping && !S.mapping.ready) {
    show('s5-blocked', true);
    text('s5-blocked-text',
      `Stage 03 found ${S.mapping.unsupported.length} operator(s) with no lowering on `
      + `${S.mapping.target}. This build is expected to fail at code generation.`);
  } else {
    show('s5-blocked', false);
  }

  // Preflight. Stage 05 is the only stage that shells out to gcc and QEMU, and without
  // them it fails several seconds in with a message about a binary the user has never
  // heard of. Checking first costs one call and turns that into an instruction.
  try {
    const h = await call('get_toolchain_health');
    const missing = [];
    if (!h.gcc) missing.push('the RISC-V cross-compiler');
    if (!h.qemu) missing.push('the QEMU system emulator');
    if (missing.length) {
      termClear();
      term('Cannot build — ' + missing.join(' and ') + ' ' +
        (missing.length > 1 ? 'are' : 'is') + ' not installed.', 'c-err');
      term('');
      term('Stage 05 compiles the generated C for the target and runs it under emulation,', 'c-dim');
      term('so both tools have to be present.', 'c-dim');
      term('');
      // The packaged app ships these, so on that build a miss almost always means the exe
      // got dragged out of its folder. A source checkout has no bundle and needs the
      // installer, and the same code runs in both -- so name both rather than guess.
      term('The app ships them in a "toolchain" folder next to TATVA.exe. If TATVA.exe', 'c-gold');
      term('was moved on its own, put it back beside that folder.', 'c-gold');
      term('Running from source instead? Open Diagnostics and press "Install toolchain".', 'c-gold');
      show('s5-toolchain', true);
      return;
    }
    show('s5-toolchain', false);
  } catch {
    // Backend unreachable is a different problem; let the real run report it.
  }

  S.busy = true;
  const btn = $('btn-build');
  btn.disabled = true; btn.classList.add('loading');
  $('s5-bar').style.display = 'block';
  $('s5-bar').classList.add('indet');

  termClear();
  term('$ tatva compile ' + (S.model.filename || S.model.path)
    + ' --target ' + S.target.name
    + (S.passes.fuse ? ' --fuse' : '')
    + (S.passes.quantize ? ' --quantize' : ''), 'c-gold');
  term('');
  term('Running. Two builds and two emulated runs — this takes minutes, not seconds.', 'c-dim');
  term('The backend reports once the whole run finishes; it does not stream.', 'c-dim');
  term('');

  const started = Date.now();
  let r = null;
  try {
    r = await call('run_pipeline', S.model.path, S.target.name, S.passes.fuse, S.passes.quantize);
  } catch (e) {
    r = { success: false, measured: false, error: e.message, status: 'ERROR' };
  }

  const secs = ((Date.now() - started) / 1000).toFixed(1);
  $('s5-bar').classList.remove('indet');
  $('s5-bar').style.display = 'none';
  btn.disabled = false; btn.classList.remove('loading');
  S.busy = false;

  if (r && r.config_digest) {
    String(r.config_digest).split('\n').forEach(l => term(l, 'c-dim'));
    term('');
  }

  // The run id exists for every outcome, including the ones that produced nothing.
  // Validation and Evidence are worth reading precisely when a build did not happen.
  S.runId = (r && r.run_id) || '';

  if (!r || !r.success) {
    // BLOCKED is not an error and must not be reported as one. The backend stopped on
    // purpose and said why; printing "Unknown failure" over the top of a reason it
    // already supplied would be the studio inventing a worse story than the truth.
    const blocked = r && r.status === 'BLOCKED';
    const msg = String((r && (r.error || r.status_reason)) || 'Unknown failure.');
    term((blocked ? 'BLOCKED after ' : 'FAILED after ') + secs + 's', blocked ? 'c-warn' : 'c-err');
    term(msg, blocked ? 'c-warn' : 'c-err');
    term('');
    if (blocked) {
      term('Nothing was compiled, so nothing is being reported as measured.', 'c-dim');
      term('Stage 03 lists each operator and whether a rewrite exists for it.', 'c-dim');
      term('');
    }
    renderOutcome(r);
    // The preflight above should have caught a missing toolchain, but it is allowed to
    // fail open -- a health-check bug must not block a machine that can actually build.
    // If the run then dies on a missing binary anyway, say what to do about it rather
    // than leaving "cross-compiler binary not found" as the last word.
    if (/cross-compiler binary not found|emulator binary not found/i.test(msg)) {
      term('That binary is part of the RISC-V toolchain the app ships in its "toolchain"', 'c-dim');
      term('folder. Check that folder is still next to TATVA.exe — moving the exe out on', 'c-gold');
      term('its own leaves it behind. From a source checkout, use Install toolchain.', 'c-gold');
      term('');
      show('s5-toolchain', true);
    }
    termDiagnosis(r && r.diagnosis);
    term('Nothing was measured, so the report page stays empty.', 'c-dim');
    S.result = null;
    markDone('s5', false);
    $('btn-s5-next').disabled = true;
    show('report-body', false); show('report-empty', true);
    // Validation and the audit trail were still recorded, and they are where a blocked
    // run explains itself. Hide the artifacts route, though: there are no artifacts.
    show('btn-s5-artifacts', false);
    conReports();
    return;
  }

  S.result = r;
  term(`baseline    ${fmtMs(r.base_ms)}`, '');
  if (r.has_optimized) {
    const d = deltaText(r.speedup);
    term(`optimized   ${fmtMs(r.opt_ms)}`, '');
    term(`change      ${d.text.replace('−', '-')} latency`, d.cls ? (d.faster ? 'c-ok' : 'c-warn') : 'c-dim');
    if (r.speedup === 0) {
      term('            no measurable difference — the selected passes had', 'c-dim');
      term('            nothing to change in this graph', 'c-dim');
    }
  } else {
    term('optimized   not built — no passes were selected', 'c-warn');
  }
  // Parity is checked on whatever was measured, including a baseline-only run, so this
  // line is not conditional on a pass being selected. It used to be, and a run with no
  // passes showed no parity line at all while the report claimed the check had passed.
  term(`parity MSE  ${fmtMse(r.mse)} vs ${r.accuracy_reference || 'host ONNX Runtime'}`,
    r.accuracy_ok ? 'c-ok' : 'c-err');
  if ((r.repair && r.repair.repaired_ops || []).length) {
    term(`repaired    ${r.repair.repaired_ops.join(', ')}`, 'c-gold');
  }
  // The status word is the backend's own verdict, coloured by that verdict rather than
  // by the parity flag alone -- a PARTIAL run with clean parity is not a green result.
  term(`status      ${r.status}`,
    r.status === 'SUCCESS' ? 'c-ok' : r.status === 'PARTIAL' ? 'c-warn' : 'c-err');
  if (r.status_reason) term(`            ${r.status_reason}`, 'c-dim');
  term(`artifacts   ${r.artifact_count} file(s) written`, 'c-dim');
  term('');
  // A build that compiles and runs but misses parity is still a result the user has to
  // act on, so the diagnosis belongs here too, not only on the exception path.
  if (!r.accuracy_ok) termDiagnosis(r.diagnosis);
  term(`Completed in ${secs}s. Full numbers on the Benchmark Report page.`, 'c-gold');

  renderOutcome(r);
  markDone('s5', true);
  $('btn-s5-next').disabled = false;
  show('btn-s5-artifacts', (r.artifact_count || 0) > 0);
  renderReport();
  conReports();
}

/* The run's verdict on the page that produced it, in the backend's own vocabulary.
 * Separate from the terminal because terminal output scrolls, and the one line that
 * says whether this build can be trusted should not be something you have to scroll
 * back to find. */
function renderOutcome(r) {
  if (!r || !r.status) { show('s5-outcome', false); return; }
  show('s5-outcome', true);
  const b = $('s5-outcome-badge');
  b.className = 'badge ' + badgeCls(r.status);
  b.textContent = r.status;
  text('s5-outcome-run', r.run_id ? `Run ${r.run_id}` : '');
  setNotice('s5-outcome-note', r.status,
    r.status_reason || r.error || 'The backend recorded no reason for this outcome.');
}

/* ============================================================ REPORT */
function renderReport() {
  const r = S.result;
  if (!r || !r.measured) {
    show('report-empty', true);
    show('report-body', false);
    return;
  }
  show('report-empty', false);
  show('report-body', true);

  // What the backend built, not what the checkboxes on stage 04 say. A pass that had
  // nothing to apply to still leaves its checkbox ticked, and reading the checkbox is
  // how a report ends up with an "Optimized" column for a build that was never made.
  const hasOpt = !!r.has_optimized;
  const passLabel = [S.passes.fuse ? 'fusion' : null, S.passes.quantize ? 'INT8' : null]
    .filter(Boolean).join(' + ');

  // §9: the verdict, above the numbers.
  text('r-status-head', {
    SUCCESS: 'Compiled, measured and validated',
    PARTIAL: 'Produced real output, with reservations',
    BLOCKED: 'Stopped before code generation',
    FAILED: 'The run failed',
  }[r.status] || 'Run outcome');
  const sb = $('r-status-badge');
  sb.className = 'badge ' + badgeCls(r.status);
  sb.textContent = r.status || '—';
  text('r-status-run', [r.run_id, S.model && S.model.filename, S.target && S.target.name]
    .filter(Boolean).join(' · '));
  setNotice('r-status-note', r.status, r.status_reason || r.validation_summary || '');

  text('r-base', fmtMs(r.base_ms));
  text('r-opt', hasOpt ? fmtMs(r.opt_ms) : '—');
  text('r-opt-n', hasOpt ? passLabel : 'no optimized build in this run');

  const d = deltaText(r.speedup);
  const delta = $('r-delta');
  delta.textContent = hasOpt ? d.text : '—';
  delta.className = 'v ' + (hasOpt ? d.cls : '');
  text('r-delta-n', !hasOpt
    ? 'no optimized build to compare'
    : r.speedup === 0
      ? 'no measurable change'
      : d.faster ? 'faster than baseline' : 'slower than baseline');

  text('r-parity', r.accuracy_ok ? 'Within tolerance' : 'Out of tolerance');
  $('r-parity').className = 'v ' + (r.accuracy_ok ? 'ok' : 'err');
  // The parity check runs against the host reference on whatever was measured, so a
  // baseline-only run has a real MSE too. "Nothing to compare" here was wrong: it was
  // shown next to a verdict that had in fact been computed.
  text('r-mse', `MSE ${fmtMse(r.mse)} vs ${r.accuracy_reference || 'host ONNX Runtime'}`
    + (r.tolerance != null ? ` · tolerance ${fmtMse(r.tolerance)}` : ''));

  const tb = $('r-table');
  tb.innerHTML = '';
  const row = (name, ms, change, mse, ok) => {
    const tr = el('tr');
    tr.innerHTML =
      `<td><b>${esc(name)}</b></td>` +
      `<td class="num mono">${ms}</td>` +
      `<td class="num mono">${change}</td>` +
      `<td class="num mono">${mse}</td>` +
      `<td>${ok}</td>`;
    tb.appendChild(tr);
  };
  // The baseline is what parity is measured against, so its own error is not a
  // number -- printing "0" there invited the reading that it had been checked.
  if (hasOpt) {
    row('Baseline (FP32, no passes)', fmtMs(r.base_ms), 'reference', '—', badge('SUCCESS', 'Reference'));
    row('Optimized (' + passLabel + ')', fmtMs(r.opt_ms), d.text, fmtMse(r.mse),
      r.accuracy_ok ? badge('PASSED', 'Pass') : badge('FAILED', 'Fail'));
  } else {
    // With no optimized build the baseline is the only thing measured, and its parity
    // against the host reference is a real result -- not a "reference" with no error.
    row('Baseline (FP32, no passes)', fmtMs(r.base_ms), '—', fmtMse(r.mse),
      r.accuracy_ok ? badge('PASSED', 'Pass') : badge('FAILED', 'Fail'));
  }

  const dg = $('r-digest');
  dg.innerHTML = '';
  String(r.config_digest || '').split('\n').forEach(l => dg.appendChild(el('div', 'c-dim', esc(l) || '&nbsp;')));

  loadEffort();
  requestAnimationFrame(drawChart);
}

/* §12: engineering effort, fetched from the backend rather than computed here.
 *
 * The number on this card and the number in engineering_effort.json are the same
 * number, from the same call. Doing the arithmetic in JavaScript would produce a second
 * figure that agrees with the first only until someone edits one of them. */
async function loadEffort() {
  let e;
  try { e = await call('get_effort', S.runId); }
  catch (err) { e = { success: false, reason: err.message }; }

  const ok = e && e.success && e.available;
  show('r-effort-body', !!ok);
  show('r-effort-empty', !ok);
  show('r-effort-how', false);
  text('btn-effort-how', 'How calculated?');

  if (!ok) {
    text('r-effort-reason', (e && (e.reason || e.error))
      || 'No estimate for this run.');
    return;
  }

  text('r-effort-hours', String(e.total_hours));
  text('r-effort-days', `${e.total_days} working days at ${e.hours_per_day} h/day`);
  text('r-effort-kind', e.measured ? 'Measured' : 'Estimate');
  text('r-effort-model', e.model_version || '—');
  text('r-effort-source', e.model_source || '');
  text('r-effort-disclaimer', e.disclaimer || '');

  const tb = $('r-effort-table');
  tb.innerHTML = '';
  (e.breakdown || []).forEach(b => {
    const tr = el('tr');
    tr.innerHTML =
      `<td>${esc(b.label)}<div class="tiny muted">per ${esc(b.unit)}</div></td>` +
      `<td class="num mono">${b.quantity}</td>` +
      `<td class="tiny muted">${esc(b.quantity_source || '')}</td>` +
      `<td class="num mono">${b.hours_per_unit}</td>` +
      `<td class="num mono">${b.hours}</td>`;
    tb.appendChild(tr);
  });
  const total = el('tr');
  total.innerHTML =
    `<td><b>Total</b></td><td class="num"></td><td></td><td class="num"></td>` +
    `<td class="num mono gold"><b>${e.total_hours}</b></td>`;
  tb.appendChild(total);

  // Every rate is an assumption and each one says so in its own words. Printing them
  // under the table is what makes the total arguable rather than authoritative.
  const basis = $('r-effort-basis');
  basis.innerHTML = '';
  (e.breakdown || []).forEach(b => {
    if (!b.basis) return;
    const d = el('div', 'tiny muted');
    d.innerHTML = `<b class="mono">${esc(b.key)}</b> — ${esc(b.basis)}`;
    basis.appendChild(d);
  });
}

/* ============================================================ ARTIFACTS (§3)
 *
 * The list is whatever the backend finds in the build directory when this page is
 * opened. It is not the manifest replayed, and it is not a fixed list of filenames a
 * build is expected to write -- a file that was never produced simply is not here, and
 * one deleted since the build disappears on the next visit.
 */
async function loadArtifacts() {
  let r;
  try { r = await call('get_artifacts', S.runId); }
  catch (e) { r = { success: false, error: e.message }; }

  const host = $('a-sections');
  show('a-viewer', false);

  if (!r || !r.success) {
    show('artifacts-body', false); show('artifacts-empty', true);
    const empty = $('artifacts-empty');
    const msg = empty.querySelector('div');
    if (msg) msg.textContent = (r && r.error) || 'No run to show yet.';
    return;
  }

  const builds = r.builds || [];
  if (!builds.length) {
    show('artifacts-body', false); show('artifacts-empty', true);
    const msg = $('artifacts-empty').querySelector('div');
    // A blocked run reaches here with a reason. Saying "nothing generated yet" over the
    // top of it would hide the one sentence that explains the empty page.
    if (msg) msg.textContent = r.reason || 'This run produced no files.';
    return;
  }

  show('artifacts-empty', false); show('artifacts-body', true);
  const totals = r.totals || {};
  text('a-count', String(totals.file_count != null ? totals.file_count : '—'));
  text('a-bytes', fmtSize(totals.total_bytes));
  text('a-builds', String(builds.length));
  text('a-builds-n', builds.map(b => b.config).join(' + '));
  text('a-note-text',
    `Read from disk just now for run ${r.run_id || S.runId}. Each build directory also carries `
    + `${r.manifest_filename || 'artifact_manifest.json'}, which records the same inventory with `
    + 'sizes and SHA-256 hashes so it can be checked outside TATVA.');

  host.innerHTML = '';
  builds.forEach(b => {
    const card = el('div', 'card');
    card.innerHTML =
      `<div class="row between">
         <div>
           <h2 style="margin-bottom:0">${esc(b.config)} build</h2>
           <p class="sub mono" style="margin-bottom:0">${esc(b.build_dir)}</p>
         </div>
         <span class="badge ${b.exists ? 'available' : 'blocked'}">${b.exists ? b.file_count + ' files' : 'directory missing'}</span>
       </div>
       <div class="table-wrap mt"><table><thead><tr>
         <th style="width:170px">File</th><th style="width:130px">Type</th>
         <th style="width:190px">Produced by</th><th>What it is</th>
         <th class="num" style="width:90px">Size</th><th style="width:90px"></th>
       </tr></thead><tbody></tbody></table></div>`;
    const body = card.querySelector('tbody');

    (b.artifacts || []).forEach(a => {
      const tr = el('tr');
      const lines = a.line_count ? `<div class="tiny muted">${a.line_count} lines</div>` : '';
      tr.innerHTML =
        `<td class="mono">${esc(a.name)}<div class="tiny muted" title="${esc(a.sha256)}">${esc(String(a.sha256).slice(0, 12))}</div></td>` +
        `<td class="muted">${esc(a.type)}</td>` +
        `<td class="muted">${esc(a.stage)}</td>` +
        `<td class="muted">${esc(a.description)}</td>` +
        `<td class="num mono">${fmtSize(a.size_bytes)}${lines}</td>` +
        `<td></td>`;
      const open = el('button', 'btn small', 'Open');
      open.addEventListener('click', () => openArtifact(a));
      tr.lastElementChild.appendChild(open);
      body.appendChild(tr);
    });

    host.appendChild(card);
  });
}

async function openArtifact(a) {
  show('a-viewer', true);
  text('a-viewer-name', a.name);
  text('a-viewer-path', a.path);
  const body = $('a-viewer-body');
  body.innerHTML = '';
  body.appendChild(el('div', 'c-dim', 'Reading…'));
  $('a-viewer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  let r;
  try { r = await call('read_artifact', a.path); }
  catch (e) { r = { success: false, error: e.message }; }

  body.innerHTML = '';
  if (!r || !r.success) {
    body.appendChild(el('div', 'c-err', esc((r && r.error) || 'Could not read that file.')));
    return;
  }
  if (r.binary) {
    // No fake disassembly. The ELF is real and its size and hash are listed; pretending
    // to show its contents would be inventing output.
    body.appendChild(el('div', 'c-warn', esc(r.note)));
    body.appendChild(el('div', 'c-dim', esc(`${fmtSize(r.size_bytes)} on disk.`)));
    return;
  }
  String(r.content || '').split('\n').forEach(l =>
    body.appendChild(el('div', 'c-dim', esc(l) || '&nbsp;')));
  if (r.truncated) {
    body.appendChild(el('div', 'c-warn',
      esc(`— truncated; the file is ${fmtSize(r.size_bytes)} and only the first part is shown here.`)));
  }
}

/* ============================================================ VALIDATION (§8)
 *
 * Two tables on purpose. The first is what TATVA checked; the second is what it does
 * not check at all. Merging them would let a page of green ticks imply coverage that
 * does not exist, which is the specific failure this stage was added to prevent.
 */
async function loadValidation() {
  let v;
  try { v = await call('get_validation', S.runId); }
  catch (e) { v = { success: false, error: e.message }; }

  if (!v || !v.success) {
    show('val-body', false); show('val-empty', true);
    const msg = $('val-empty').querySelector('div');
    if (msg) msg.textContent = (v && v.error) || 'No run to validate yet.';
    return;
  }
  show('val-empty', false); show('val-body', true);

  const b = $('val-badge');
  b.className = 'badge ' + badgeCls(v.verdict);
  b.textContent = v.verdict || '—';
  text('val-run', [v.run_id, v.model, v.target].filter(Boolean).join(' · '));
  setNotice('val-note', v.verdict, v.summary || '');

  text('val-passed', String(v.passed != null ? v.passed : '—'));
  text('val-failed', String(v.failed != null ? v.failed : '—'));
  text('val-skipped', String(v.skipped != null ? v.skipped : '—'));
  text('val-todo', String(v.not_implemented != null ? v.not_implemented : '—'));
  $('val-failed').className = 'v ' + (v.failed ? 'err' : '');

  const checks = v.checks || [];
  const ran = checks.filter(c => c.status !== 'NOT_IMPLEMENTED');
  const todo = checks.filter(c => c.status === 'NOT_IMPLEMENTED');

  const fill = (id, rows, evidenceKey) => {
    const tb = $(id);
    tb.innerHTML = '';
    if (!rows.length) {
      tb.innerHTML = '<tr><td colspan="3" class="muted">Nothing in this category for this run.</td></tr>';
      return;
    }
    rows.forEach(c => {
      const tr = el('tr');
      tr.innerHTML =
        `<td><b>${esc(c.name)}</b><div class="tiny muted mono">${esc(c.category)}</div></td>` +
        `<td>${badge(c.status)}</td>` +
        `<td class="muted">${esc(c.detail)}` +
          (c[evidenceKey] ? `<div class="tiny mono" style="margin-top:4px;opacity:.8">${esc(c[evidenceKey])}</div>` : '') +
        `</td>`;
      tb.appendChild(tr);
    });
  };
  fill('val-table', ran, 'evidence');
  fill('val-todo-table', todo, 'evidence');
}

/* ============================================================ EVIDENCE (§24)
 *
 * The audit trail is the point of this page: what happened, in order, including the
 * events that went badly. A trail that can only record success is a brochure.
 */
async function loadEvidence() {
  // Fetched independently. A failure in one of the three should show up as a gap in
  // that section, not turn the whole page into "no run recorded".
  const grab = async (name) => {
    try { return await call(name, S.runId); }
    catch (e) { return { success: false, error: e.message }; }
  };
  const h = await grab('get_optimization_history');
  const a = await grab('get_audit_trail');
  const run = await grab('get_run');

  if (!h || !h.success) {
    show('ev-body', false); show('ev-empty', true);
    const msg = $('ev-empty').querySelector('div');
    if (msg) msg.textContent = (h && h.error) || 'No run recorded in this session yet.';
    return;
  }
  show('ev-empty', false); show('ev-body', true);

  const tb = $('ev-history');
  tb.innerHTML = '';
  const history = h.history || [];
  if (!history.length) {
    tb.innerHTML = '<tr><td colspan="5" class="muted">No transformation was applied in this run.</td></tr>';
  }
  history.forEach((e, i) => {
    const tr = el('tr');
    tr.innerHTML =
      `<td class="num mono muted">${i + 1}</td>` +
      `<td><b>${esc(e.name)}</b><div class="tiny muted">${esc(e.stage)}</div></td>` +
      `<td class="muted">${esc(e.stage)}</td>` +
      `<td class="muted"><div><span class="muted">from</span> ${esc(e.before)}</div>` +
        `<div><span class="muted">to</span> ${esc(e.after)}</div>` +
        (e.reason ? `<div class="tiny muted" style="margin-top:4px">${esc(e.reason)}</div>` : '') +
        (e.impact ? `<div class="tiny gold" style="margin-top:2px">${esc(e.impact)}</div>` : '') + `</td>` +
      `<td class="tiny muted">${esc(e.validation || 'Not recorded.')}` +
        (e.detail ? `<div class="mono" style="margin-top:4px;opacity:.75">${esc(e.detail)}</div>` : '') + `</td>`;
    tb.appendChild(tr);
  });

  const repair = (run && run.success && run.repair) || {};
  const records = repair.records || [];
  show('ev-repair-empty', !records.length);
  show('ev-repair-wrap', records.length > 0);
  if (!records.length && repair.status && repair.status !== 'NO_REPAIR_NEEDED') {
    const msg = $('ev-repair-empty').querySelector('div');
    if (msg) msg.textContent = repair.message || 'No rewrite was applied.';
  }
  const rb = $('ev-repair');
  rb.innerHTML = '';
  // A rewrite that was validated and kept, one that was thrown away because validation
  // did not hold, and one whose operator is still unmapped are three different outcomes.
  // Collapsing them into pass/fail would make a discarded rewrite look like a fix.
  const REWRITE = {
    MAPPED: ['SUCCESS', 'Mapped'],
    DISCARDED: ['FAILED', 'Discarded — validation did not hold'],
    UNMAPPED: ['PARTIAL', 'Still unmapped'],
  };
  records.forEach(rec => {
    const tr = el('tr');
    const [st, label] = REWRITE[rec.mapping_result] || ['SKIPPED', rec.mapping_result || '—'];
    tr.innerHTML =
      `<td class="mono">${esc(rec.original_op)}</td>` +
      `<td class="num mono">${rec.occurrences}</td>` +
      `<td class="mono tiny">${esc((rec.replacement_ops || []).join(' → ') || '—')}</td>` +
      `<td class="muted">${esc(rec.identity || '')}` +
        (rec.reason ? `<div class="tiny muted" style="margin-top:4px">${esc(rec.reason)}</div>` : '') +
        `<div class="tiny muted" style="margin-top:4px">structural: ${esc(rec.structural_validation)} · ` +
        `numerical: ${esc(rec.numerical_validation)}` +
        (rec.max_abs_diff != null ? ` · max |Δ| ${fmtMse(rec.max_abs_diff)}` : '') + `</div></td>` +
      `<td>${badge(st, label)}</td>`;
    rb.appendChild(tr);
  });

  const log = $('ev-audit');
  log.innerHTML = '';
  const events = (a && a.events) || [];
  if (!events.length) {
    log.appendChild(el('div', 'c-dim', 'No audit events were recorded for this run.'));
    return;
  }
  const CLS = { ok: 'c-ok', warn: 'c-warn', error: 'c-err', blocked: 'c-warn', info: 'c-dim' };
  events.forEach(ev => {
    const head = `${String(ev.seq).padStart(2, '0')}  +${Number(ev.elapsed_s).toFixed(1)}s  `
      + `${String(ev.stage).padEnd(10)} ${ev.event}`;
    log.appendChild(el('div', CLS[ev.outcome] || 'c-dim', esc(head)));
    if (ev.detail) log.appendChild(el('div', 'c-dim', esc('      ' + ev.detail)));
    const evidence = ev.evidence || {};
    Object.keys(evidence).forEach(k => {
      const val = evidence[k];
      if (val === null || val === undefined || val === '') return;
      log.appendChild(el('div', 'c-dim',
        esc(`      ${k}: ${typeof val === 'object' ? JSON.stringify(val) : val}`)));
    });
  });
}

/* ============================================================ the chart
 * Drawn by hand on a 2D context. The previous page loaded Chart.js from a CDN and
 * then never called it -- `latencyChartInstance` was declared and never assigned, so
 * the canvas sat empty forever. That is the "no graph" the report described. Doing it
 * here means it also works with no network, which is the machine TATVA targets.
 */
function drawChart() {
  const canvas = $('chart-latency');
  const r = S.result;
  if (!canvas || !r || !r.measured) return;

  const box = canvas.parentElement;
  const cssW = Math.max(320, box.clientWidth);
  const cssH = Math.max(200, box.clientHeight || 260);
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';

  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  // Colours come from the stylesheet, so the chart follows the theme toggle without
  // keeping a second copy of the palette here.
  const cs = getComputedStyle(document.documentElement);
  const c = n => cs.getPropertyValue(n).trim();
  const ink = c('--ink'), dim = c('--ink-3'), line = c('--line'), gold = c('--gold');
  const okc = c('--ok'), warnc = c('--warn');

  const hasOpt = S.passes.fuse || S.passes.quantize;
  const bars = [{ label: 'Baseline', ms: r.base_ms, fill: dim }];
  if (hasOpt) {
    bars.push({ label: 'Optimized', ms: r.opt_ms, fill: r.speedup > 0 ? gold : warnc });
  }

  // padT leaves room for the delta bracket and its label above the tallest bar.
  // At 18 the bracket was drawn on top of the topmost gridline and its own text ran
  // off the canvas, so the one number the run exists to produce was the least legible
  // thing on the chart.
  const padL = 68, padR = 20, padT = 52, padB = 40;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;

  // Round the axis up to something readable rather than to the tallest bar, so the
  // two bars are comparable by eye and not just by label.
  const peak = Math.max.apply(null, bars.map(b => b.ms));
  const step = Math.pow(10, Math.floor(Math.log10(peak || 1))) / 2;
  const top = Math.ceil((peak * 1.15) / step) * step || 1;

  // Enough decimals that the ticks are distinguishable from each other. A sub-
  // millisecond run at fixed 1 dp printed "0.1 / 0.1 / 0.0 / 0.0 / 0.0 ms" up the
  // side -- five labels, three of them the same, none of them the actual value.
  const tickStep = top / 4;
  const tickDp = tickStep >= 10 ? 0 : tickStep >= 1 ? 1 : tickStep >= 0.1 ? 2 : tickStep >= 0.01 ? 3 : 4;

  g.font = '11px ui-monospace, Consolas, monospace';
  g.textBaseline = 'middle';
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = (top / ticks) * i;
    const y = padT + plotH - (v / top) * plotH;
    g.strokeStyle = line;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(padL, Math.round(y) + 0.5);
    g.lineTo(padL + plotW, Math.round(y) + 0.5);
    g.stroke();
    g.fillStyle = dim;
    g.textAlign = 'right';
    g.fillText(v.toFixed(tickDp) + (i === ticks ? ' ms' : ''), padL - 10, y);
  }

  const slot = plotW / bars.length;
  const bw = Math.min(130, slot * 0.46);

  bars.forEach((b, i) => {
    const cx = padL + slot * i + slot / 2;
    const h = Math.max(2, (b.ms / top) * plotH);
    const x = cx - bw / 2;
    const y = padT + plotH - h;

    const rad = Math.min(7, bw / 2, h);
    g.fillStyle = b.fill;
    g.beginPath();
    g.moveTo(x, y + h);
    g.lineTo(x, y + rad);
    g.quadraticCurveTo(x, y, x + rad, y);
    g.lineTo(x + bw - rad, y);
    g.quadraticCurveTo(x + bw, y, x + bw, y + rad);
    g.lineTo(x + bw, y + h);
    g.closePath();
    g.fill();

    g.fillStyle = ink;
    g.textAlign = 'center';
    g.font = '600 13px ui-monospace, Consolas, monospace';
    g.fillText(fmtMs(b.ms), cx, y - 12);

    g.fillStyle = dim;
    g.font = '12px system-ui, -apple-system, Segoe UI, sans-serif';
    g.fillText(b.label, cx, padT + plotH + 20);
  });

  // Delta bracket between the two bars — the number the whole run exists to produce.
  if (bars.length === 2) {
    const x1 = padL + slot * 0.5, x2 = padL + slot * 1.5;
    // Clamped so the label always lands inside the canvas: without the clamp a bar
    // that reaches the top of the plot pushes the text above y=0 and it vanishes.
    const yTop = Math.max(22, padT + plotH - (Math.max(bars[0].ms, bars[1].ms) / top) * plotH - 34);
    const d = deltaText(r.speedup);
    const col = d.cls === 'ok' ? okc : d.cls === 'warn' ? warnc : dim;
    g.strokeStyle = col;
    g.lineWidth = 1.4;
    g.beginPath();
    g.moveTo(x1, yTop + 8); g.lineTo(x1, yTop); g.lineTo(x2, yTop); g.lineTo(x2, yTop + 8);
    g.stroke();
    g.fillStyle = col;
    g.textAlign = 'center';
    g.font = '600 12px ui-monospace, Consolas, monospace';
    g.fillText(
      r.speedup === 0 ? 'no measurable change' : d.text + ' latency',
      (x1 + x2) / 2, yTop - 9);
  }
}

/* ============================================================ DIAGNOSTICS */
async function loadHealth() {
  const tb = $('diag-table');
  let h;
  try { h = await call('get_toolchain_health'); }
  catch (e) {
    setHealthPill('warn', 'Backend not connected');
    if (tb) tb.innerHTML = `<tr><td colspan="4" class="muted">${esc(e.message)}</td></tr>`;
    return;
  }

  // Only GCC and QEMU are required. CMake and Make are used by the project-scaffolding
  // page, not by the compile-and-measure pipeline -- counting them made a machine that
  // could run all five stages report "2 component(s) missing", which reads as broken.
  const rows = [
    ['RISC-V GCC', h.gcc, h.gcc_name || 'riscv-none-elf-gcc', h.gcc_path, true, 'compiles the generated C for the target'],
    ['QEMU (system)', h.qemu, h.qemu_name || 'qemu-system-riscv64', h.qemu_path, true, 'runs the binary and reads the cycle counter'],
    ['CMake', h.cmake, 'cmake', '', false, 'only for the Project Scaffolding page'],
    ['Make', h.make, 'make', '', false, 'only for the Project Scaffolding page'],
  ];

  if (tb) {
    tb.innerHTML = '';
    rows.forEach(([label, ok, name, path, required, note]) => {
      const tr = el('tr');
      // The "what it is for" line lives with the component name, not in the Required
      // column. In the Required column it had a badge's worth of width to wrap into
      // and turned every row four lines tall.
      tr.innerHTML =
        `<td><b>${esc(label)}</b>` +
        `<div class="tiny muted mono">${esc(name)}</div>` +
        `<div class="tiny muted">${esc(note)}</div></td>` +
        `<td>${required
          ? '<span class="badge info">Required</span>'
          : '<span class="badge roadmap">Optional</span>'}</td>` +
        `<td>${ok ? '<span class="badge available">Found</span>' : '<span class="badge blocked">Missing</span>'}</td>` +
        `<td class="mono tiny muted">${esc(path || (ok ? 'on PATH' : '—'))}</td>`;
      tb.appendChild(tr);
    });
  }

  const missingReq = rows.filter(r => r[4] && !r[1]).map(r => r[0]);
  show('diag-ready', missingReq.length === 0);
  show('diag-missing', missingReq.length > 0);
  if (missingReq.length) {
    text('diag-missing-text',
      `${missingReq.join(' and ')} ${missingReq.length > 1 ? 'are' : 'is'} not installed. ` +
      // 01-04 need nothing external; 05 is the only stage that shells out. This used to
      // say 04 and 05, which sent people looking for an install they did not need.
      'Stages 01–04 still work; only 05 needs it. The packaged app ships one in its ' +
      '"toolchain" folder — if that is missing, TATVA.exe has been moved away from it.');
  }

  if (!missingReq.length) setHealthPill('ok', 'Toolchain ready');
  else if (h.gcc || h.qemu) setHealthPill('warn', `${missingReq.join(' + ')} missing`);
  else setHealthPill('err', 'Toolchain not installed');

  loadToolchainPlan(missingReq.length > 0);
}

/* ---- in-app toolchain install --------------------------------------------
 * Everything below exists so that whoever receives the zip can reach stage 05.
 * Without it they get through analyze, map and optimize and then hit "RISC-V GCC
 * cross-compiler binary not found", with the documented fix being a terminal
 * command from a repository they do not have.
 */
async function loadToolchainPlan(needed) {
  const tb = $('tc-plan');
  const card = $('diag-install-card');

  // This card is a recovery path, not a setup step. The packaged app already ships the
  // toolchain, so on that build the health check comes back green and a prominent
  // "Install the RISC-V toolchain" button would be inviting a half-gigabyte download
  // for something already sitting next to the exe. Show it only when something is
  // actually missing -- i.e. a source checkout or a pip install that never ran setup.
  if (!needed) {
    if (card) card.classList.add('hidden');
    return;
  }

  let p;
  try { p = await call('get_toolchain_plan'); }
  catch { return; }

  if (!p || !p.supported) {
    if (card) card.classList.add('hidden');
    return;
  }
  if (card) card.classList.remove('hidden');

  if (tb) {
    tb.innerHTML = '';
    p.components.forEach(c => {
      const tr = el('tr');
      tr.innerHTML =
        `<td><b>${esc(c.label)}</b><div class="tiny muted mono">v${esc(c.version)}</div></td>` +
        // Size is shown either way. Replacing it with the badge left the SIZE column
        // header standing over a column that no longer had a size in it.
        `<td class="mono tiny">~${c.size_mb} MB` +
        (c.installed ? '<div class="tiny"><span class="badge available">Installed</span></div>' : '') +
        `</td>` +
        `<td class="mono tiny muted" style="word-break:break-all">${esc(c.url)}</td>`;
      tb.appendChild(tr);
    });
  }
  text('tc-dest', `Installs into ${p.tools_dir} — no admin rights, nothing added to PATH.`);

  const allIn = p.components.every(c => c.installed);
  const btn = $('btn-install-tc');
  if (btn) {
    btn.disabled = false;
    btn.lastChild.textContent = allIn ? ' Re-install' : ' Install toolchain';
  }
}

async function startToolchainInstall() {
  const btn = $('btn-install-tc');
  if (btn) btn.disabled = true;
  show('tc-progress', true);
  $('tc-log').innerHTML = '';

  let res;
  try { res = await call('start_toolchain_install', false); }
  catch (e) {
    $('tc-log').appendChild(el('div', 'c-err', esc(e.message)));
    if (btn) btn.disabled = false;
    return;
  }
  if (!res || !res.started) {
    $('tc-log').appendChild(el('div', 'c-err', esc((res && res.error) || 'Could not start.')));
    if (btn) btn.disabled = false;
    return;
  }
  pollInstall(0);
}

function pollInstall(shown) {
  call('get_toolchain_install_status').then(s => {
    if (!s) return;

    const log = $('tc-log');
    (s.log || []).slice(shown).forEach(line => {
      const cls = /^FAILED/.test(line) ? 'c-err' : /ready|Installed/.test(line) ? 'c-ok' : 'c-dim';
      log.appendChild(el('div', cls, esc(line)));
    });
    log.scrollTop = log.scrollHeight;
    const seen = (s.log || []).length;

    text('tc-stage', s.label || (s.running ? 'Working…' : 'Done'));
    const bar = $('tc-bar');
    if (s.total > 0) {
      const pct = Math.min(100, Math.round((s.read / s.total) * 100));
      text('tc-pct', `${pct}%  (${Math.round(s.read / 1048576)} / ${Math.round(s.total / 1048576)} MB)`);
      bar.parentElement.classList.remove('indet');
      bar.style.width = pct + '%';
    } else if (s.running) {
      // Unpacking and verifying report no byte count, so an indeterminate sweep is
      // the honest thing to show rather than a percentage nothing is measuring.
      text('tc-pct', 'working…');
      bar.parentElement.classList.add('indet');
    }

    if (s.running) { setTimeout(() => pollInstall(seen), 400); return; }

    bar.parentElement.classList.remove('indet');
    bar.style.width = '100%';
    text('tc-pct', s.ok ? 'complete' : 'failed');
    $('btn-install-tc').disabled = false;
    // Re-probe so the table, the pill and the stage-05 button all agree.
    loadHealth();
  }).catch(() => {
    $('btn-install-tc').disabled = false;
  });
}

function setHealthPill(kind, msg) {
  const dot = $('health-dot');
  if (dot) dot.className = 'dot ' + kind;
  text('health-text', msg);
}

/* ============================================================ ASSISTANT */
function chatBubble(who, body, cls) {
  const box = $('chat');
  const n = el('div', 'notice ' + (cls || 'info'));
  n.innerHTML = `<span><b>${esc(who)}</b><br>${esc(body).replace(/\n/g, '<br>')}</span>`;
  box.appendChild(n);
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  const input = $('chat-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  chatBubble('You', q, '');

  const btn = $('btn-chat');
  btn.disabled = true; btn.classList.add('loading');
  try {
    const model = $('scaf-model').value || '';
    const key = $('scaf-key').value || '';
    const r = await call('ask_assistant', q, model, key);
    if (r && r.success) {
      chatBubble(`TATVA · ${r.model || 'assistant'}`, r.reply, 'ok');
    } else {
      chatBubble('Not answered', (r && r.error) || 'No LLM is configured or reachable.', 'warn');
    }
  } catch (e) {
    chatBubble('Not answered', e.message, 'warn');
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

/* ============================================================ SCAFFOLDING */
async function syncModels() {
  const provider = $('scaf-provider').value;
  const sel = $('scaf-model');
  const note = t => { show('scaf-status', true); text('scaf-status-text', t); };

  try {
    let names = [];
    if (provider.startsWith('Local')) {
      names = await call('get_ollama_models');
      note(names && names.length
        ? `${names.length} local model(s) found on the Ollama server.`
        : 'No local Ollama server responded on the default port.');
    } else if (provider.startsWith('NVIDIA')) {
      const key = $('scaf-key').value.trim();
      if (!key) { note('Enter an NVIDIA API key first.'); return; }
      const r = await call('fetch_nvidia_models', key);
      if (!r || !r.success) { note((r && r.error) || 'The catalog request failed.'); return; }
      names = r.models || [];
      note(`${names.length} model(s) available on this key.`);
    } else {
      note(`${provider} is not wired up in this build. Local Ollama and NVIDIA NIM are.`);
      return;
    }

    sel.innerHTML = '';
    (names.length ? names : ['']).forEach(n => {
      const o = document.createElement('option');
      o.value = n; o.textContent = n || '—';
      sel.appendChild(o);
    });
  } catch (e) { note(e.message); }
}

async function runScaffold() {
  const btn = $('btn-scaf-run');
  btn.disabled = true; btn.classList.add('loading');
  show('scaf-status', true);
  text('scaf-status-text', 'Generating…');
  try {
    const r = await call('run_autonomous_loop',
      $('scaf-prompt').value, S.target ? S.target.name : 'RV64GC', $('scaf-model').value || '');
    text('scaf-status-text', (r && (r.summary || r.status || r.error)) || 'Finished.');
  } catch (e) {
    text('scaf-status-text', e.message);
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

/* ============================================================ splash */
function hideSplash() {
  const s = $('splash');
  if (!s || s.classList.contains('gone')) return;
  s.classList.add('gone');
  setTimeout(() => { if (s.parentNode) s.parentNode.removeChild(s); }, 600);
}

/* ============================================================ wiring */
function wire() {
  document.body.addEventListener('click', ev => {
    const nav = ev.target.closest('[data-view]');
    if (nav) { switchView(nav.dataset.view); return; }
    const go = ev.target.closest('[data-goto]');
    if (go) { switchView(go.dataset.goto); }
  });

  // Flow Navigator groups collapse. The expanded state is the truth in the DOM -- the
  // header's aria-expanded and the group's hidden attribute are set from the same
  // toggle, so a screen reader and the caret never disagree.
  document.querySelectorAll('.nav-head').forEach(head => {
    head.addEventListener('click', () => {
      const group = $(head.getAttribute('aria-controls'));
      if (!group) return;
      const open = head.getAttribute('aria-expanded') !== 'true';
      head.setAttribute('aria-expanded', String(open));
      group.hidden = !open;
    });
  });

  wireConsole();

  $('btn-theme').addEventListener('click', () =>
    applyTheme(document.documentElement.className === 'dark' ? 'light' : 'dark'));

  $('btn-browse').addEventListener('click', browseModel);
  $('btn-load-sample').addEventListener('click', loadSamples);
  $('btn-sample-2').addEventListener('click', loadSamples);
  $('model-path').addEventListener('change', e => validateModel(e.target.value.trim()));
  $('btn-s1-next').addEventListener('click', () => switchView('s2'));

  $('btn-analyze').addEventListener('click', runAnalyze);
  $('btn-s2-next').addEventListener('click', () => switchView('s3'));

  $('btn-map').addEventListener('click', runMap);
  $('btn-autofix').addEventListener('click', runAutoFix);
  $('btn-caps').addEventListener('click', loadCapabilities);
  $('btn-s3-next').addEventListener('click', () => switchView('s4'));

  $('pass-fuse').addEventListener('click', () => togglePass('fuse'));
  $('pass-quant').addEventListener('click', () => togglePass('quantize'));

  $('btn-nl').addEventListener('click', () => applyNlConfig());
  $('nl-input').addEventListener('keydown', e => { if (e.key === 'Enter') applyNlConfig(); });
  // The examples are the documentation. Someone who has never seen this field has no
  // idea what it accepts, and a placeholder shows only one shape of answer.
  document.querySelectorAll('.nl-eg').forEach(b => b.addEventListener('click', () => {
    $('nl-input').value = b.textContent.trim();
    applyNlConfig();
  }));
  $('btn-s4-next').addEventListener('click', () => { markDone('s4', true); switchView('s5'); });

  $('btn-build').addEventListener('click', runBuild);
  // Two exits from stage 05, and neither is a gate on the other. The spec is explicit
  // that the artifact list must not become a checkpoint the user has to pass through
  // to reach the benchmark report.
  $('btn-s5-artifacts').addEventListener('click', () => switchView('artifacts'));
  $('btn-s5-next').addEventListener('click', () => switchView('report'));

  $('btn-effort-how').addEventListener('click', () => {
    const open = $('r-effort-how').classList.contains('hidden');
    show('r-effort-how', open);
    text('btn-effort-how', open ? 'Hide the method' : 'How calculated?');
  });
  $('btn-a-close').addEventListener('click', () => show('a-viewer', false));

  $('btn-recheck').addEventListener('click', loadHealth);
  $('btn-install-tc').addEventListener('click', startToolchainInstall);
  $('btn-goto-install').addEventListener('click', () => {
    switchView('diag');
    // The install card is below the health table; scroll to it so the button the
    // user was just sent to find is on screen when they arrive.
    requestAnimationFrame(() => $('diag-install-card').scrollIntoView({ behavior: 'smooth', block: 'center' }));
  });
  $('health-btn').addEventListener('click', () => switchView('diag'));

  $('btn-chat').addEventListener('click', sendChat);
  $('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });

  $('btn-scaf-sync').addEventListener('click', syncModels);
  $('scaf-provider').addEventListener('change', syncModels);
  $('btn-scaf-run').addEventListener('click', runScaffold);

  // Drag and drop onto the stage-01 card. pywebview hands over a real path on
  // Windows; in a browser it does not, and we say so rather than failing silently.
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach(t =>
    drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t =>
    drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    const path = f.path || f.name;
    if (path && path !== f.name) { $('model-path').value = path; validateModel(path); }
    else stageError('s1', `Dropped "${f.name}", but this window cannot see its full path. Use Browse… instead.`);
  });
  drop.addEventListener('click', browseModel);

  let rt;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (S.view === 'report') drawChart(); }, 120);
  });

  // Keyboard: 1-5 jump between stages, so the numbering is usable as well as visible.
  document.addEventListener('keydown', e => {
    if (e.target.matches('input, textarea, select')) return;
    const i = '12345'.indexOf(e.key);
    if (i >= 0) switchView(STAGES[i].id);
  });
}

async function boot() {
  buildNav();
  buildRail();
  buildOverviewTable();
  wire();

  // Light is the default, the way Vivado ships. Dark is still one click away and is
  // remembered per machine once chosen.
  let theme = 'light';
  try { theme = localStorage.getItem('tatva-theme') || 'light'; } catch (_) { /* private mode */ }
  applyTheme(theme);

  markDone('s1', false);
  switchView('overview');
  renderPlan();

  if (!connected()) {
    // Still run the loaders: each one reports "not connected" in its own panel, which
    // is more use than a silently empty page.
    text('mode-badge', 'Preview — no backend');
    text('build-badge', '—');
    await loadTargets();
    await loadHealth();
    return;
  }

  text('mode-badge', 'Local · offline');
  try {
    const b = await call('get_build_info');
    // b.version is already display-ready ("2.1"). Do not decorate it: prefixing
    // a "v" here is how the badge once ended up reading "vv0.3.0".
    text('build-badge', b.version || b.label || '—');
    const badgeNode = $('build-badge');
    if (badgeNode && b.label) badgeNode.title = b.label;   // full label, incl. the exact build
  } catch (_) { text('build-badge', '—'); }

  await loadTargets();
  await loadHealth();
  // The importable-format table is a property of the build, not of any one model, so it
  // is filled once here rather than being re-fetched every time a file is selected.
  await loadFormats();
}

/* pywebview injects the bridge after the document loads, so the page waits for its
 * ready event. In a plain browser that event never fires, hence the timeout — the
 * splash must not be able to trap the window either way. */
document.addEventListener('DOMContentLoaded', () => {
  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    boot().finally(() => setTimeout(hideSplash, 260));
  };
  window.addEventListener('pywebviewready', start);
  if (window.pywebview) start();
  setTimeout(start, 1500);
});
