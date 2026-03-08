// ── PERFUSION PAGE JS ──

let perfusionFile    = null;
let perfusionRawFile = null;
let spectralChart    = null;
let rfChart          = null;
let svmChart         = null;
let _perfTimerRef    = null;
let _perfProgRef     = null;  // progress stepper interval

// ── Timer ──────────────────────────────────────────────
function _startPerfTimer() {
  const se = document.getElementById('perfusion-timer-secs');
  if (!se) return;
  let s = 0;
  se.textContent = '0s';
  _perfTimerRef = setInterval(() => {
    s++;
    const m = Math.floor(s / 60);
    se.textContent = m > 0 ? m + 'm ' + (s % 60) + 's' : s + 's';
  }, 1000);
}
function _stopPerfTimer() {
  clearInterval(_perfTimerRef);
  _perfTimerRef = null;
}

// ── Progress panel ──────────────────────────────────────
const PERF_STEPS = [
  'perf-step-0',
  'perf-step-1',
  'perf-step-2',
  'perf-step-3',
  'perf-step-4'
];
// Realistic estimated durations (ms) per step — b2nd can be very long
// These are display-only estimates; we advance on a schedule
const PERF_STEP_MS = [1200, 2000, 4000, 5000, 1500];

function _perfSetStep(idx, state) {
  const el = document.getElementById(PERF_STEPS[idx]);
  if (el) el.dataset.state = state;
}

function _perfSetProgress(pct) {
  const bar = document.getElementById('perfusion-progress-bar');
  if (bar) bar.style.width = Math.min(pct, 100) + '%';
}

function _startPerfProgress() {
  const panel = document.getElementById('perfusion-progress-panel');
  if (panel) panel.style.display = 'block';

  // Reset all
  PERF_STEPS.forEach((_, i) => _perfSetStep(i, 'idle'));
  _perfSetProgress(0);

  let currentStep = 0;
  const totalSteps = PERF_STEPS.length;
  const pctPerStep = 90 / totalSteps; // leave last 10% for actual response

  function advance() {
    if (currentStep >= totalSteps) return;
    if (currentStep > 0) _perfSetStep(currentStep - 1, 'done');
    _perfSetStep(currentStep, 'active');
    _perfSetProgress(currentStep * pctPerStep + pctPerStep * 0.4);

    const delay = PERF_STEP_MS[currentStep] || 2000;
    currentStep++;
    _perfProgRef = setTimeout(advance, delay);
  }
  advance();
}

function _stopPerfProgress(success) {
  clearTimeout(_perfProgRef);
  _perfProgRef = null;

  const panel = document.getElementById('perfusion-progress-panel');

  if (success) {
    // Mark all done, fill bar
    PERF_STEPS.forEach((_, i) => _perfSetStep(i, 'done'));
    _perfSetProgress(100);
    // Hide panel after brief moment so user sees 100%
    setTimeout(() => { if (panel) panel.style.display = 'none'; }, 900);
  } else {
    // Mark current active as error, hide panel after delay
    PERF_STEPS.forEach((_, i) => {
      const el = document.getElementById(PERF_STEPS[i]);
      if (el && el.dataset.state === 'active') el.dataset.state = 'error';
    });
    setTimeout(() => { if (panel) panel.style.display = 'none'; }, 2500);
  }
}

// ── File helpers ───────────────────────────────────────
function _isHdr(file) {
  return file && file.name.toLowerCase().endsWith('.hdr');
}

function _updateRunBtn() {
  const runBtn    = document.getElementById('perfusion-run-btn');
  const localPath = window._perfLocalPath;
  const fileReady = perfusionFile && (!_isHdr(perfusionFile) || perfusionRawFile);
  runBtn.disabled = !localPath && !fileReady;
}

// ── DOMContentLoaded ───────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadPerfusionEvaluation();

  const dropZone  = document.getElementById('perfusion-drop-zone');
  const fileInput = document.getElementById('perfusion-file-input');
  const browseBtn = document.getElementById('perfusion-browse-btn');
  const rawInput  = document.getElementById('perfusion-raw-input');
  const rawBrowse = document.getElementById('perfusion-raw-browse-btn');
  const rawRow    = document.getElementById('perfusion-raw-row');
  const runBtn    = document.getElementById('perfusion-run-btn');
  const demoBtn   = document.getElementById('perfusion-demo-btn');

  browseBtn.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('click',  () => fileInput.click());

  fileInput.addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    perfusionFile = f;
    window._perfFile = f;
    document.getElementById('perfusion-file-name').textContent = f.name;
    if (_isHdr(f)) {
      rawRow.style.display = 'block';
      perfusionRawFile = null;
      document.getElementById('perfusion-raw-name').textContent = 'No .raw file selected';
    } else {
      rawRow.style.display = 'none';
      perfusionRawFile = null;
    }
    _updateRunBtn();
  });

  if (rawBrowse) rawBrowse.addEventListener('click', () => rawInput.click());
  if (rawInput) {
    rawInput.addEventListener('change', e => {
      const f = e.target.files[0];
      if (!f) return;
      perfusionRawFile = f;
      document.getElementById('perfusion-raw-name').textContent = f.name;
      _updateRunBtn();
    });
  }

  dropZone.addEventListener('dragover',  e  => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (!f) return;
    perfusionFile = f;
    window._perfFile = f;
    document.getElementById('perfusion-file-name').textContent = f.name;
    if (_isHdr(f)) {
      rawRow.style.display = 'block';
      perfusionRawFile = null;
      document.getElementById('perfusion-raw-name').textContent = 'No .raw file selected';
    } else {
      rawRow.style.display = 'none';
      perfusionRawFile = null;
    }
    _updateRunBtn();
  });

  const pathInput = document.getElementById('perfusion-local-path');
  if (pathInput) {
    pathInput.addEventListener('input', () => {
      window._perfLocalPath = pathInput.value.trim() || null;
      _updateRunBtn();
    });
  }

  runBtn.addEventListener('click', async () => {
    const localPath = window._perfLocalPath;
    if (!localPath && !perfusionFile) return;

    if (perfusionFile && _isHdr(perfusionFile) && !perfusionRawFile) {
      showError('ENVI .hdr files require the companion .raw data file. Please select the .raw binary using the second file picker above.');
      return;
    }

    setLoading(true);
    _startPerfTimer();
    _startPerfProgress();

    try {
      let data;
      if (localPath) {
        data = await HSI_API.predictPerfusionLocal(localPath);
      } else if (_isHdr(perfusionFile)) {
        data = await HSI_API.predictPerfusion(perfusionFile, perfusionRawFile);
      } else {
        data = await HSI_API.predictPerfusion(perfusionFile);
      }
      _stopPerfTimer();
      _stopPerfProgress(true);
      renderResults(data);
    } catch (err) {
      _stopPerfTimer();
      _stopPerfProgress(false);
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });

  demoBtn.addEventListener('click', () => {
    setLoading(true);
    _startPerfTimer();
    _startPerfProgress();
    setTimeout(() => {
      _stopPerfTimer();
      _stopPerfProgress(true);
      setLoading(false);
      renderResults(generatePerfusionDemo());
    }, 1400);
  });
});

// ── UI helpers ─────────────────────────────────────────
function setLoading(on) {
  const btn     = document.getElementById('perfusion-run-btn');
  const spinner = document.getElementById('perfusion-spinner');
  const txt     = document.getElementById('perfusion-btn-text');
  spinner.classList.toggle('hidden', !on);
  txt.textContent = on ? 'Analyzing...' : 'Run Perfusion Analysis';
  btn.disabled = on;
}

function showError(msg) {
  document.getElementById('perfusion-error-text').textContent = msg;
  document.getElementById('perfusion-error-panel').classList.remove('hidden');
}

function resetPerfusion() {
  document.getElementById('perfusion-error-panel').classList.add('hidden');
}

// ── Result rendering ────────────────────────────────────
const COLOR_MAP = {
  'Normal Perfusion':   { dot: '#00ff88', badge: 'badge-normal'   },
  'Reduced Perfusion':  { dot: '#ff8c00', badge: 'badge-reduced'  },
  'Abnormal Perfusion': { dot: '#ff2244', badge: 'badge-abnormal' }
};

function renderResults(data) {
  document.getElementById('perfusion-error-panel').classList.add('hidden');

  const summaryPanel = document.getElementById('perfusion-summary-panel');
  summaryPanel.classList.remove('hidden');
  const cm = COLOR_MAP[data.prediction] || COLOR_MAP['Normal Perfusion'];
  const label = document.getElementById('perfusion-summary-label');
  label.textContent = data.prediction;
  label.style.color = cm.dot;
  document.getElementById('perfusion-summary-conf').textContent = ' (Confidence: ' + data.confidence + '%)';
  document.getElementById('perfusion-summary-text').textContent = data.ai_summary;

  document.getElementById('perfusion-viz-row').classList.remove('hidden');
  renderHeatmap('perfusion-heatmap', data.heatmap, 'perfusion');
  renderSpectralChart(data.spectral_signature);
  renderFusionSummary(data);

  document.getElementById('perfusion-classifiers-row').classList.remove('hidden');
  renderClassifierCharts(data);
  renderProbBars(data);

  summaryPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderHeatmap(containerId, heatmapData, type) {
  const colorscale = type === 'perfusion'
    ? [[0, '#00ff88'], [0.5, '#ffaa00'], [1, '#ff2244']]
    : [[0, '#0055ff'], [0.5, '#8800ff'], [1, '#ff2244']];
  Plotly.newPlot(containerId, [{
    z: heatmapData, type: 'heatmap', colorscale, showscale: false
  }], {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    xaxis: { visible: false }, yaxis: { visible: false }
  }, { displayModeBar: false, responsive: true });
}

function renderSpectralChart(sig) {
  if (spectralChart) spectralChart.destroy();
  const ctx = document.getElementById('perfusion-spectral-chart').getContext('2d');
  spectralChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sig.bands.map(b => 'B' + b),
      datasets: [
        { label: 'Tissue Sample',     data: sig.tissue,  borderColor: '#ff4466', backgroundColor: 'rgba(255,60,100,0.1)',  borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true },
        { label: 'Healthy Reference', data: sig.healthy, borderColor: '#00e5ff', backgroundColor: 'rgba(0,220,255,0.06)', borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#c8dcff', font: { family: 'Figtree', size: 11 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { color: '#6080aa', font: { size: 9 } }, grid: { color: 'rgba(0,100,255,0.07)' } },
        y: { ticks: { color: '#6080aa', font: { size: 9 } }, grid: { color: 'rgba(0,100,255,0.07)' } }
      }
    }
  });
}

function renderFusionSummary(data) {
  const container = document.getElementById('perfusion-fusion-summary');
  const fp = data.fusion_probabilities;
  container.innerHTML = '';
  Object.entries(fp).forEach(function(entry) {
    const label = entry[0], pct = entry[1];
    const cm = COLOR_MAP[label];
    const isWinner = label === data.prediction;
    const row = document.createElement('div');
    row.className = 'fusion-row' + (isWinner ? ' winner' : '');
    row.innerHTML =
      '<span class="fusion-row-label" style="color:' + (cm ? cm.dot : '#ccc') + '">' + label + '</span>' +
      '<span class="fusion-row-pct"  style="color:' + (cm ? cm.dot : '#ccc') + '">' + pct + '%</span>';
    container.appendChild(row);
  });
  const badge = document.getElementById('perfusion-final-badge');
  const cm = COLOR_MAP[data.prediction];
  badge.className = 'final-result-badge ' + (cm ? cm.badge : '');
  badge.textContent = 'FINAL: ' + data.prediction.toUpperCase();
}

function renderClassifierCharts(data) {
  const rf  = data.classifiers.random_forest;
  const svm = data.classifiers.svm;
  document.getElementById('perf-rf-label').textContent    = rf.prediction;
  document.getElementById('perf-rf-conf-val').textContent  = rf.confidence + '%';
  document.getElementById('perf-svm-label').textContent   = svm.prediction;
  document.getElementById('perf-svm-conf-val').textContent = svm.confidence + '%';
  if (rfChart)  rfChart.destroy();
  rfChart  = new Chart(document.getElementById('perf-rf-chart').getContext('2d'),  barChartConfig(rf.probabilities,  '#0096ff'));
  if (svmChart) svmChart.destroy();
  svmChart = new Chart(document.getElementById('perf-svm-chart').getContext('2d'), barChartConfig(svm.probabilities, '#00e5ff'));
}

function barChartConfig(probabilities, color) {
  return {
    type: 'bar',
    data: {
      labels: Object.keys(probabilities).map(function(k) { return k.replace(' Perfusion', '').replace(' Tissue', ''); }),
      datasets: [{ data: Object.values(probabilities), backgroundColor: color + '55', borderColor: color, borderWidth: 1.5, borderRadius: 4 }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#6080aa', font: { size: 8 } }, grid: { display: false } },
        y: { ticks: { color: '#6080aa', font: { size: 8 } }, grid: { color: 'rgba(0,100,255,0.07)' }, max: 100 }
      }
    }
  };
}

async function loadPerfusionEvaluation() {
  const loading = document.getElementById('perfusion-eval-loading');
  const metrics = document.getElementById('perfusion-eval-metrics');
  const none    = document.getElementById('perfusion-eval-none');
  try {
    const data = await HSI_API.getEvaluationPerfusion();
    loading.classList.add('hidden');
    if (data && data.fusion) {
      none.classList.add('hidden');
      metrics.classList.remove('hidden');
      document.getElementById('perf-eval-accuracy').textContent = (data.fusion.accuracy * 100).toFixed(2) + '%';
      document.getElementById('perf-eval-f1').textContent       = (data.fusion.f1_score  * 100).toFixed(2) + '%';
      const labels = data.fusion.confusion_matrix_labels || data.classes || [];
      const cm     = data.fusion.confusion_matrix || [];
      renderConfusionMatrix('perfusion-confusion-matrix', cm, labels);
    } else {
      metrics.classList.add('hidden'); none.classList.remove('hidden');
    }
  } catch (e) {
    loading.classList.add('hidden'); metrics.classList.add('hidden'); none.classList.remove('hidden');
  }
}

function renderConfusionMatrix(containerId, cm, labels) {
  if (!cm.length || !labels.length) return;
  Plotly.newPlot(containerId, [{
    z: cm, x: labels, y: labels, type: 'heatmap',
    colorscale: 'Blues', showscale: true, hoverongaps: false
  }], {
    margin: { t: 30, b: 60, l: 80, r: 40 },
    xaxis: { tickangle: -30 }, yaxis: { autorange: 'reversed' },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#c8dcff', size: 10 }
  }, { displayModeBar: true, responsive: true });
}

function renderProbBars(data) {
  const fp     = data.fusion_probabilities;
  const colors = { 'Normal Perfusion': '#00ff88', 'Reduced Perfusion': '#ff8c00', 'Abnormal Perfusion': '#ff2244' };
  const ids    = {
    'Normal Perfusion':   ['perf-prob-normal',   'perf-prob-normal-bar'],
    'Reduced Perfusion':  ['perf-prob-reduced',  'perf-prob-reduced-bar'],
    'Abnormal Perfusion': ['perf-prob-abnormal', 'perf-prob-abnormal-bar']
  };
  Object.keys(fp).forEach(function(key) {
    const pct = fp[key];
    const pair = ids[key];
    if (!pair) return;
    const pctEl = document.getElementById(pair[0]);
    const barEl = document.getElementById(pair[1]);
    if (pctEl) { pctEl.textContent = pct + '%'; pctEl.style.color = colors[key] || '#ccc'; }
    if (barEl) { barEl.style.background = colors[key] || '#ccc'; setTimeout(function() { barEl.style.width = pct + '%'; }, 100); }
  });
  const cm = COLOR_MAP[data.prediction];
  const final = document.getElementById('perf-final-row');
  if (final) { final.className = 'final-label-row ' + (cm ? cm.badge : ''); final.textContent = 'Final Result: ' + data.prediction; }
}
