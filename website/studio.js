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
  busy: false,
  done: { s1: false, s2: false, s3: false, s4: false, s5: false },
};

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

/* ============================================================ terminal */
function term(line, cls) {
  const t = $('terminal');
  if (!t) return;
  const d = el('div', cls || '', esc(line));
  t.appendChild(d);
  t.scrollTop = t.scrollHeight;
}
const termClear = () => { const t = $('terminal'); if (t) t.innerHTML = ''; };

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
  if (S.result) { S.result = null; show('report-body', false); show('report-empty', true); markDone('s5', false); }
  $('btn-s3-next').disabled = true;
  $('btn-s5-next').disabled = true;
  renderPlan();
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

function stageError(stage, msg) {
  show(`${stage}-error`, true);
  text(`${stage}-error-text`, msg);
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
  S.analysis = null; S.mapping = null; S.result = null;
  show('s2-result', false); show('s2-empty', true); markDone('s2', false);
  show('s3-result', false); show('s3-empty', true); markDone('s3', false);
  show('report-body', false); show('report-empty', true); markDone('s5', false);
  $('btn-s2-next').disabled = true;
  $('btn-s3-next').disabled = true;
  $('btn-s5-next').disabled = true;
  renderPlan();
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
    if (a && a.error) throw new Error(a.error);
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
    stageError('s2', e.message);
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
    if (!m || !m.success) throw new Error((m && m.error) || 'Mapping failed.');
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

    const tb = $('s3-table');
    tb.innerHTML = '';
    (m.operators || []).forEach(o => {
      const badge = !o.supported ? '<span class="badge blocked">No lowering</span>'
        : o.kind === 'fused' ? '<span class="badge available">Fused</span>'
        : o.kind === 'hot' ? '<span class="badge experimental">Hot path</span>'
        : '<span class="badge available">Mapped</span>';
      const tr = el('tr');
      tr.innerHTML =
        `<td class="mono">${esc(o.op)}</td>` +
        `<td class="num mono">${o.count}</td>` +
        `<td class="muted">${esc(o.lowering)}</td>` +
        `<td>${badge}</td>`;
      tb.appendChild(tr);
    });

    show('s3-empty', false); show('s3-result', true);
    markDone('s3', true);
    $('btn-s3-next').disabled = false;
  } catch (e) {
    stageError('s3', e.message);
    markDone('s3', false);
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
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
      term('so both tools have to be present. TATVA does not bundle them: together they', 'c-dim');
      term('are about half a gigabyte and licensed separately.', 'c-dim');
      term('');
      term('Open Diagnostics and press "Install toolchain". It downloads the pinned', 'c-gold');
      term('builds into your user folder — no admin rights, nothing added to PATH.', 'c-gold');
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

  if (!r || !r.success) {
    const msg = String((r && r.error) || 'Unknown failure.');
    term('FAILED after ' + secs + 's', 'c-err');
    term(msg, 'c-err');
    term('');
    // The preflight above should have caught a missing toolchain, but it is allowed to
    // fail open -- a health-check bug must not block a machine that can actually build.
    // If the run then dies on a missing binary anyway, say what to do about it rather
    // than leaving "cross-compiler binary not found" as the last word.
    if (/cross-compiler binary not found|emulator binary not found/i.test(msg)) {
      term('That binary is part of the RISC-V toolchain, which TATVA does not bundle.', 'c-dim');
      term('Open Diagnostics and press "Install toolchain" — it downloads the pinned', 'c-gold');
      term('builds into your user folder, with no admin rights needed.', 'c-gold');
      term('');
      show('s5-toolchain', true);
    }
    term('Nothing was measured, so the report page stays empty.', 'c-dim');
    S.result = null;
    markDone('s5', false);
    $('btn-s5-next').disabled = true;
    show('report-body', false); show('report-empty', true);
    return;
  }

  S.result = r;
  term(`baseline    ${fmtMs(r.base_ms)}`, '');
  if (S.passes.fuse || S.passes.quantize) {
    const d = deltaText(r.speedup);
    term(`optimized   ${fmtMs(r.opt_ms)}`, '');
    term(`change      ${d.text.replace('−', '-')} latency`, d.cls ? (d.faster ? 'c-ok' : 'c-warn') : 'c-dim');
    if (r.speedup === 0) {
      term('            no measurable difference — the selected passes had', 'c-dim');
      term('            nothing to change in this graph', 'c-dim');
    }
    term(`parity MSE  ${fmtMse(r.mse)} vs host ONNX Runtime`, r.accuracy_ok ? 'c-ok' : 'c-err');
  } else {
    term('optimized   not built — no passes were selected', 'c-warn');
  }
  term(`status      ${r.status}`, r.accuracy_ok ? 'c-ok' : 'c-err');
  term('');
  term(`Completed in ${secs}s. Full numbers on the Benchmark Report page.`, 'c-gold');

  markDone('s5', true);
  $('btn-s5-next').disabled = false;
  renderReport();
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

  const hasOpt = S.passes.fuse || S.passes.quantize;

  text('r-base', fmtMs(r.base_ms));
  text('r-opt', hasOpt ? fmtMs(r.opt_ms) : '—');
  text('r-opt-n', hasOpt
    ? [S.passes.fuse ? 'fusion' : null, S.passes.quantize ? 'INT8' : null].filter(Boolean).join(' + ')
    : 'no passes selected');

  const d = deltaText(r.speedup);
  const delta = $('r-delta');
  delta.textContent = hasOpt ? d.text : '—';
  delta.className = 'v ' + (hasOpt ? d.cls : '');
  text('r-delta-n', !hasOpt
    ? 'no passes selected'
    : r.speedup === 0
      ? 'no measurable change'
      : d.faster ? 'faster than baseline' : 'slower than baseline');

  text('r-parity', r.accuracy_ok ? 'Within tolerance' : 'Out of tolerance');
  $('r-parity').className = 'v ' + (r.accuracy_ok ? 'ok' : 'err');
  text('r-mse', hasOpt ? `MSE ${fmtMse(r.mse)} vs host ONNX Runtime` : 'nothing to compare');

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
  row('Baseline (FP32, no passes)', fmtMs(r.base_ms), 'reference', '—', '<span class="badge available">Reference</span>');
  if (hasOpt) {
    row(
      'Optimized (' + [S.passes.fuse ? 'fusion' : null, S.passes.quantize ? 'INT8' : null].filter(Boolean).join(' + ') + ')',
      fmtMs(r.opt_ms),
      d.text,
      fmtMse(r.mse),
      r.accuracy_ok ? '<span class="badge available">Pass</span>' : '<span class="badge blocked">Fail</span>');
  }

  const dg = $('r-digest');
  dg.innerHTML = '';
  String(r.config_digest || '').split('\n').forEach(l => dg.appendChild(el('div', 'c-dim', esc(l) || '&nbsp;')));

  requestAnimationFrame(drawChart);
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
      'Stages 01–03 still work; 04 and 05 need it. Use Install toolchain below.');
  }

  if (!missingReq.length) setHealthPill('ok', 'Toolchain ready');
  else if (h.gcc || h.qemu) setHealthPill('warn', `${missingReq.join(' + ')} missing`);
  else setHealthPill('err', 'Toolchain not installed');

  loadToolchainPlan();
}

/* ---- in-app toolchain install --------------------------------------------
 * Everything below exists so that whoever receives the zip can reach stage 05.
 * Without it they get through analyze, map and optimize and then hit "RISC-V GCC
 * cross-compiler binary not found", with the documented fix being a terminal
 * command from a repository they do not have.
 */
async function loadToolchainPlan() {
  const tb = $('tc-plan');
  let p;
  try { p = await call('get_toolchain_plan'); }
  catch { return; }

  const card = $('diag-install-card');
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
  $('btn-s3-next').addEventListener('click', () => switchView('s4'));

  $('pass-fuse').addEventListener('click', () => togglePass('fuse'));
  $('pass-quant').addEventListener('click', () => togglePass('quantize'));
  $('btn-s4-next').addEventListener('click', () => { markDone('s4', true); switchView('s5'); });

  $('btn-build').addEventListener('click', runBuild);
  $('btn-s5-next').addEventListener('click', () => switchView('report'));

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

  let theme = 'dark';
  try { theme = localStorage.getItem('tatva-theme') || 'dark'; } catch (_) { /* private mode */ }
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
    // b.version is already display-ready ("Beta 2.0"). Do not decorate it: prefixing
    // a "v" here is how the badge once ended up reading "vv0.3.0".
    text('build-badge', b.version || b.label || '—');
    const badge = $('build-badge');
    if (badge && b.label) badge.title = b.label;   // full label, incl. the exact build
  } catch (_) { text('build-badge', '—'); }

  await loadTargets();
  await loadHealth();
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
