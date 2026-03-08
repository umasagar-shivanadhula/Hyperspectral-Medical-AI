// ── TUMOR PAGE JS ──

let tumorFile          = null;
let tumorRawFile       = null;
let tumorSpectralChart = null;
let tumorRfChart       = null;
let tumorSvmChart      = null;
let _timerRef          = null;
let _tumorProgRef      = null;

// ── Timer ──────────────────────────────────────────────
function _startTimer(page) {
  const se = document.getElementById(page + '-timer-secs');
  if (!se) return;
  let s = 0;
  se.textContent = '0s';
  _timerRef = setInterval(function() {
    s++;
    const m = Math.floor(s / 60);
    se.textContent = m > 0 ? m + 'm ' + (s % 60) + 's' : s + 's';
  }, 1000);
}
function _stopTimer() {
  clearInterval(_timerRef);
  _timerRef = null;
}

// ── Progress panel ──────────────────────────────────────
const TUMOR_STEPS    = ['tumor-step-0','tumor-step-1','tumor-step-2','tumor-step-3','tumor-step-4'];
const TUMOR_STEP_MS  = [1000, 1800, 3500, 4000, 1200];

function _tumorSetStep(idx, state) {
  const el = document.getElementById(TUMOR_STEPS[idx]);
  if (el) el.dataset.state = state;
}
function _tumorSetProgress(pct) {
  const bar = document.getElementById('tumor-progress-bar');
  if (bar) bar.style.width = Math.min(pct, 100) + '%';
}
function _startTumorProgress() {
  const panel = document.getElementById('tumor-progress-panel');
  if (panel) panel.style.display = 'block';
  TUMOR_STEPS.forEach(function(_, i) { _tumorSetStep(i, 'idle'); });
  _tumorSetProgress(0);

  let currentStep = 0;
  const pctPerStep = 90 / TUMOR_STEPS.length;

  function advance() {
    if (currentStep >= TUMOR_STEPS.length) return;
    if (currentStep > 0) _tumorSetStep(currentStep - 1, 'done');
    _tumorSetStep(currentStep, 'active');
    _tumorSetProgress(currentStep * pctPerStep + pctPerStep * 0.4);
    const delay = TUMOR_STEP_MS[currentStep] || 2000;
    currentStep++;
    _tumorProgRef = setTimeout(advance, delay);
  }
  advance();
}
function _stopTumorProgress(success) {
  clearTimeout(_tumorProgRef);
  _tumorProgRef = null;
  const panel = document.getElementById('tumor-progress-panel');
  if (success) {
    TUMOR_STEPS.forEach(function(_, i) { _tumorSetStep(i, 'done'); });
    _tumorSetProgress(100);
    setTimeout(function() { if (panel) panel.style.display = 'none'; }, 900);
  } else {
    TUMOR_STEPS.forEach(function(_, i) {
      const el = document.getElementById(TUMOR_STEPS[i]);
      if (el && el.dataset.state === 'active') el.dataset.state = 'error';
    });
    setTimeout(function() { if (panel) panel.style.display = 'none'; }, 2500);
  }
}

// ── File helpers ───────────────────────────────────────
function _isHdr(file) {
  return file && file.name.toLowerCase().endsWith('.hdr');
}
function _updateRunBtn() {
  const runBtn    = document.getElementById('tumor-run-btn');
  const localPath = window._tumorLocalPath;
  const fileReady = tumorFile && (!_isHdr(tumorFile) || tumorRawFile);
  runBtn.disabled = !localPath && !fileReady;
}

// ── DOMContentLoaded ───────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  loadTumorEvaluation();

  const dropZone  = document.getElementById('tumor-drop-zone');
  const fileInput = document.getElementById('tumor-file-input');
  const browseBtn = document.getElementById('tumor-browse-btn');
  const rawInput  = document.getElementById('tumor-raw-input');
  const rawBrowse = document.getElementById('tumor-raw-browse-btn');
  const rawRow    = document.getElementById('tumor-raw-row');
  const runBtn    = document.getElementById('tumor-run-btn');
  const demoBtn   = document.getElementById('tumor-demo-btn');

  browseBtn.addEventListener('click', function() { fileInput.click(); });
  dropZone.addEventListener('click',  function() { fileInput.click(); });

  fileInput.addEventListener('change', function(e) {
    const f = e.target.files[0];
    if (!f) return;
    tumorFile = f;
    document.getElementById('tumor-file-name').textContent = f.name;
    if (_isHdr(f)) {
      rawRow.style.display = 'block';
      tumorRawFile = null;
      document.getElementById('tumor-raw-name').textContent = 'No .raw file selected';
    } else {
      rawRow.style.display = 'none';
      tumorRawFile = null;
    }
    _updateRunBtn();
  });

  if (rawBrowse) rawBrowse.addEventListener('click', function() { rawInput.click(); });
  if (rawInput) {
    rawInput.addEventListener('change', function(e) {
      const f = e.target.files[0];
      if (!f) return;
      tumorRawFile = f;
      document.getElementById('tumor-raw-name').textContent = f.name;
      _updateRunBtn();
    });
  }

  dropZone.addEventListener('dragover',  function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', function()  { dropZone.classList.remove('dragover'); });
  dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (!f) return;
    tumorFile = f;
    document.getElementById('tumor-file-name').textContent = f.name;
    if (_isHdr(f)) {
      rawRow.style.display = 'block';
      tumorRawFile = null;
      document.getElementById('tumor-raw-name').textContent = 'No .raw file selected';
    } else {
      rawRow.style.display = 'none';
      tumorRawFile = null;
    }
    _updateRunBtn();
  });

  const pathInput = document.getElementById('tumor-local-path');
  if (pathInput) {
    pathInput.addEventListener('input', function() {
      window._tumorLocalPath = pathInput.value.trim() || null;
      _updateRunBtn();
    });
  }

  runBtn.addEventListener('click', async function() {
    const localPath = window._tumorLocalPath;
    if (!localPath && !tumorFile) return;

    if (tumorFile && _isHdr(tumorFile) && !tumorRawFile) {
      showTumorError('ENVI .hdr files require the companion .raw data file. Please select the .raw binary using the second file picker above.');
      return;
    }

    setLoading(true);
    _startTimer('tumor');
    _startTumorProgress();

    try {
      let data;
      if (localPath) {
        data = await HSI_API.predictTumorLocal(localPath);
      } else if (_isHdr(tumorFile)) {
        data = await HSI_API.predictTumor(tumorFile, tumorRawFile);
      } else {
        data = await HSI_API.predictTumor(tumorFile);
      }
      _stopTimer();
      _stopTumorProgress(true);
      renderTumorResults(data);
    } catch (err) {
      _stopTimer();
      _stopTumorProgress(false);
      showTumorError(err.message);
    } finally {
      setLoading(false);
    }
  });

  demoBtn.addEventListener('click', function() {
    setLoading(true);
    _startTimer('tumor');
    _startTumorProgress();
    setTimeout(function() {
      _stopTimer();
      _stopTumorProgress(true);
      setLoading(false);
      renderTumorResults(generateTumorDemo());
    }, 1400);
  });
});

// ── UI helpers ─────────────────────────────────────────
function setLoading(on) {
  const btn     = document.getElementById('tumor-run-btn');
  const spinner = document.getElementById('tumor-spinner');
  const txt     = document.getElementById('tumor-btn-text');
  spinner.classList.toggle('hidden', !on);
  txt.textContent = on ? 'Detecting...' : 'Run Tumor Detection';
  btn.disabled = on;
}
function showTumorError(msg) {
  document.getElementById('tumor-error-text').textContent = msg;
  document.getElementById('tumor-error-panel').classList.remove('hidden');
}
function resetTumor() {
  document.getElementById('tumor-error-panel').classList.add('hidden');
}

// ── Color map ──────────────────────────────────────────
const TUMOR_COLOR_MAP = {
  'Non-Tumor Tissue':   { dot: '#0096ff', badge: 'badge-nontumor' },
  'Glioblastoma Tumor': { dot: '#ff2244', badge: 'badge-tumor'    }
};

// ── Result rendering ────────────────────────────────────
function renderTumorResults(data) {
  document.getElementById('tumor-error-panel').classList.add('hidden');
  const summaryPanel = document.getElementById('tumor-summary-panel');
  summaryPanel.classList.remove('hidden');
  const cm = TUMOR_COLOR_MAP[data.prediction] || TUMOR_COLOR_MAP['Non-Tumor Tissue'];
  const label = document.getElementById('tumor-summary-label');
  label.textContent = data.prediction;
  label.style.color = cm.dot;
  document.getElementById('tumor-summary-conf').textContent = ' (Confidence: ' + data.confidence + '%)';
  document.getElementById('tumor-summary-text').textContent = data.ai_summary;

  document.getElementById('tumor-viz-row').classList.remove('hidden');
  renderHeatmapTumor('tumor-heatmap', data.heatmap);
  renderTumorSpectralChart(data.spectral_signature);
  renderTumorFusionSummary(data);

  document.getElementById('tumor-classifiers-row').classList.remove('hidden');
  renderTumorClassifierCharts(data);
  renderTumorProbBars(data);

  summaryPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderHeatmapTumor(containerId, heatmapData) {
  Plotly.newPlot(containerId, [{
    z: heatmapData, type: 'heatmap',
    colorscale: [[0, '#0055ff'], [0.5, '#8800ff'], [1, '#ff2244']], showscale: false
  }], {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    xaxis: { visible: false }, yaxis: { visible: false }
  }, { displayModeBar: false, responsive: true });
}

function renderTumorSpectralChart(sig) {
  if (tumorSpectralChart) tumorSpectralChart.destroy();
  const ctx = document.getElementById('tumor-spectral-chart').getContext('2d');
  tumorSpectralChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sig.bands.map(function(b) { return 'B' + b; }),
      datasets: [
        { label: 'Tissue Sample',     data: sig.tissue,  borderColor: '#ff4466', backgroundColor: 'rgba(255,60,100,0.1)',  borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true },
        { label: 'Healthy Reference', data: sig.healthy, borderColor: '#0096ff', backgroundColor: 'rgba(0,150,255,0.06)', borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true }
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

function renderTumorFusionSummary(data) {
  const container = document.getElementById('tumor-fusion-summary');
  const fp = data.fusion_probabilities;
  container.innerHTML = '';
  Object.entries(fp).forEach(function(entry) {
    const label = entry[0], pct = entry[1];
    const cm = TUMOR_COLOR_MAP[label];
    const isWinner = label === data.prediction;
    const row = document.createElement('div');
    row.className = 'fusion-row' + (isWinner ? ' winner' : '');
    row.innerHTML =
      '<span class="fusion-row-label" style="color:' + (cm ? cm.dot : '#ccc') + '">' + label + '</span>' +
      '<span class="fusion-row-pct"  style="color:' + (cm ? cm.dot : '#ccc') + '">' + pct + '%</span>';
    container.appendChild(row);
  });
  const badge = document.getElementById('tumor-final-badge');
  const cm = TUMOR_COLOR_MAP[data.prediction];
  badge.className = 'final-result-badge ' + (cm ? cm.badge : '');
  badge.textContent = 'FINAL: ' + data.prediction.toUpperCase();
}

function renderTumorClassifierCharts(data) {
  const rf  = data.classifiers.random_forest;
  const svm = data.classifiers.svm;
  document.getElementById('tumor-rf-label').textContent    = rf.prediction;
  document.getElementById('tumor-rf-conf-val').textContent  = rf.confidence + '%';
  document.getElementById('tumor-svm-label').textContent   = svm.prediction;
  document.getElementById('tumor-svm-conf-val').textContent = svm.confidence + '%';
  if (tumorRfChart)  tumorRfChart.destroy();
  tumorRfChart  = new Chart(document.getElementById('tumor-rf-chart').getContext('2d'),  tumorBarChart(rf.probabilities,  '#0096ff'));
  if (tumorSvmChart) tumorSvmChart.destroy();
  tumorSvmChart = new Chart(document.getElementById('tumor-svm-chart').getContext('2d'), tumorBarChart(svm.probabilities, '#ff4466'));
}

function tumorBarChart(probabilities, color) {
  return {
    type: 'bar',
    data: {
      labels: Object.keys(probabilities).map(function(k) { return k.replace(' Tissue', '').replace('Glioblastoma ', 'GBM '); }),
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

async function loadTumorEvaluation() {
  const loading = document.getElementById('tumor-eval-loading');
  const metrics = document.getElementById('tumor-eval-metrics');
  const none    = document.getElementById('tumor-eval-none');
  try {
    const data = await HSI_API.getEvaluationTumor();
    loading.classList.add('hidden');
    if (data && data.fusion) {
      none.classList.add('hidden');
      metrics.classList.remove('hidden');
      document.getElementById('tumor-eval-accuracy').textContent = (data.fusion.accuracy * 100).toFixed(2) + '%';
      document.getElementById('tumor-eval-f1').textContent       = (data.fusion.f1_score  * 100).toFixed(2) + '%';
      const labels = data.fusion.confusion_matrix_labels || data.classes || [];
      const cm     = data.fusion.confusion_matrix || [];
      renderTumorConfusionMatrix('tumor-confusion-matrix', cm, labels);
    } else {
      metrics.classList.add('hidden'); none.classList.remove('hidden');
    }
  } catch (e) {
    loading.classList.add('hidden'); metrics.classList.add('hidden'); none.classList.remove('hidden');
  }
}

function renderTumorConfusionMatrix(containerId, cm, labels) {
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

function renderTumorProbBars(data) {
  const fp     = data.fusion_probabilities;
  const colors = { 'Non-Tumor Tissue': '#0096ff', 'Glioblastoma Tumor': '#ff2244' };
  const ids    = {
    'Non-Tumor Tissue':   ['tumor-prob-normal', 'tumor-prob-normal-bar'],
    'Glioblastoma Tumor': ['tumor-prob-tumor',  'tumor-prob-tumor-bar']
  };
  Object.keys(fp).forEach(function(key) {
    const pct  = fp[key];
    const pair = ids[key];
    if (!pair) return;
    const pctEl = document.getElementById(pair[0]);
    const barEl = document.getElementById(pair[1]);
    if (pctEl) { pctEl.textContent = pct + '%'; pctEl.style.color = colors[key]; }
    if (barEl) { barEl.style.background = colors[key]; setTimeout(function() { barEl.style.width = pct + '%'; }, 100); }
  });
  const cm = TUMOR_COLOR_MAP[data.prediction];
  const final = document.getElementById('tumor-final-row');
  if (final) { final.className = 'final-label-row ' + (cm ? cm.badge : ''); final.textContent = 'Final Result: ' + data.prediction; }
}
