// ── PERFUSION PAGE JS ──

let perfusionFile = null;
let spectralChart = null;
let rfChart = null;
let svmChart = null;

document.addEventListener('DOMContentLoaded', () => {
  loadPerfusionEvaluation();

  const dropZone = document.getElementById('perfusion-drop-zone');
  const fileInput = document.getElementById('perfusion-file-input');
  const browseBtn = document.getElementById('perfusion-browse-btn');
  const runBtn = document.getElementById('perfusion-run-btn');
  const demoBtn = document.getElementById('perfusion-demo-btn');

  browseBtn.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) {
      perfusionFile = e.target.files[0];
      document.getElementById('perfusion-file-name').textContent = perfusionFile.name;
      runBtn.disabled = false;
    }
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) {
      perfusionFile = f;
      document.getElementById('perfusion-file-name').textContent = f.name;
      runBtn.disabled = false;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!perfusionFile) return;
    setLoading(true);
    try {
      const data = await HSI_API.predictPerfusion(perfusionFile);
      renderResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });

  demoBtn.addEventListener('click', () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      renderResults(generatePerfusionDemo());
    }, 1200);
  });
});

function setLoading(on) {
  const btn = document.getElementById('perfusion-run-btn');
  const spinner = document.getElementById('perfusion-spinner');
  const txt = document.getElementById('perfusion-btn-text');
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

const COLOR_MAP = {
  'Normal Perfusion': { dot: '#00ff88', bar: 'rgba(0,255,136,0.6)', badge: 'badge-normal' },
  'Reduced Perfusion': { dot: '#ff8c00', bar: 'rgba(255,140,0,0.6)', badge: 'badge-reduced' },
  'Abnormal Perfusion': { dot: '#ff2244', bar: 'rgba(255,34,68,0.6)', badge: 'badge-abnormal' }
};

function renderResults(data) {
  // Hide error
  document.getElementById('perfusion-error-panel').classList.add('hidden');

  // Summary
  const summaryPanel = document.getElementById('perfusion-summary-panel');
  summaryPanel.classList.remove('hidden');
  const cm = COLOR_MAP[data.prediction] || COLOR_MAP['Normal Perfusion'];
  const label = document.getElementById('perfusion-summary-label');
  label.textContent = data.prediction;
  label.style.color = cm.dot;
  document.getElementById('perfusion-summary-conf').textContent = ` (Confidence: ${data.confidence}%)`;
  document.getElementById('perfusion-summary-text').textContent = data.ai_summary;

  // Viz row
  document.getElementById('perfusion-viz-row').classList.remove('hidden');
  renderHeatmap('perfusion-heatmap', data.heatmap, 'perfusion');
  renderSpectralChart(data.spectral_signature);
  renderFusionSummary(data);

  // Classifiers row
  document.getElementById('perfusion-classifiers-row').classList.remove('hidden');
  renderClassifierCharts(data);
  renderProbBars(data);

  // Scroll into view
  summaryPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderHeatmap(containerId, heatmapData, type) {
  const colorscale = type === 'perfusion'
    ? [[0, '#00ff88'], [0.5, '#ffaa00'], [1, '#ff2244']]
    : [[0, '#0055ff'], [0.5, '#8800ff'], [1, '#ff2244']];

  Plotly.newPlot(containerId, [{
    z: heatmapData,
    type: 'heatmap',
    colorscale,
    showscale: false
  }], {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { visible: false },
    yaxis: { visible: false }
  }, { displayModeBar: false, responsive: true });
}

function renderSpectralChart(sig) {
  if (spectralChart) spectralChart.destroy();
  const ctx = document.getElementById('perfusion-spectral-chart').getContext('2d');
  spectralChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sig.bands.map(b => `B${b}`),
      datasets: [
        {
          label: 'Tissue Sample',
          data: sig.tissue,
          borderColor: '#ff4466',
          backgroundColor: 'rgba(255,60,100,0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.4,
          fill: true
        },
        {
          label: 'Healthy Reference',
          data: sig.healthy,
          borderColor: '#00e5ff',
          backgroundColor: 'rgba(0,220,255,0.06)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#c8dcff', font: { family: 'Rajdhani', size: 11 }, boxWidth: 12 } }
      },
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
  Object.entries(fp).forEach(([label, pct]) => {
    const cm = COLOR_MAP[label];
    const isWinner = label === data.prediction;
    const row = document.createElement('div');
    row.className = `fusion-row ${isWinner ? 'winner' : ''}`;
    row.innerHTML = `
      <span class="fusion-row-label" style="color:${cm ? cm.dot : '#ccc'}">${label}</span>
      <span class="fusion-row-pct" style="color:${cm ? cm.dot : '#ccc'}">${pct}%</span>`;
    container.appendChild(row);
  });

  const badge = document.getElementById('perfusion-final-badge');
  const cm = COLOR_MAP[data.prediction];
  badge.className = `final-result-badge ${cm ? cm.badge : ''}`;
  badge.textContent = `FINAL: ${data.prediction.toUpperCase()}`;
}

function renderClassifierCharts(data) {
  const rf = data.classifiers.random_forest;
  const svm = data.classifiers.svm;

  document.getElementById('perf-rf-label').textContent = rf.prediction;
  document.getElementById('perf-rf-conf-val').textContent = `${rf.confidence}%`;
  document.getElementById('perf-svm-label').textContent = svm.prediction;
  document.getElementById('perf-svm-conf-val').textContent = `${svm.confidence}%`;

  if (rfChart) rfChart.destroy();
  const rfCtx = document.getElementById('perf-rf-chart').getContext('2d');
  rfChart = new Chart(rfCtx, barChartConfig(rf.probabilities, '#0096ff'));

  if (svmChart) svmChart.destroy();
  const svmCtx = document.getElementById('perf-svm-chart').getContext('2d');
  svmChart = new Chart(svmCtx, barChartConfig(svm.probabilities, '#00e5ff'));
}

function barChartConfig(probabilities, color) {
  return {
    type: 'bar',
    data: {
      labels: Object.keys(probabilities).map(k => k.replace(' Perfusion', '').replace(' Tissue', '')),
      datasets: [{
        data: Object.values(probabilities),
        backgroundColor: color + '55',
        borderColor: color,
        borderWidth: 1.5,
        borderRadius: 4
      }]
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
  const none = document.getElementById('perfusion-eval-none');
  try {
    const data = await HSI_API.getEvaluationPerfusion();
    loading.classList.add('hidden');
    if (data && data.fusion) {
      none.classList.add('hidden');
      metrics.classList.remove('hidden');
      document.getElementById('perf-eval-accuracy').textContent = (data.fusion.accuracy * 100).toFixed(2) + '%';
      document.getElementById('perf-eval-f1').textContent = (data.fusion.f1_score * 100).toFixed(2) + '%';
      const labels = data.fusion.confusion_matrix_labels || data.classes || [];
      const cm = data.fusion.confusion_matrix || [];
      renderConfusionMatrix('perfusion-confusion-matrix', cm, labels);
    } else {
      metrics.classList.add('hidden');
      none.classList.remove('hidden');
    }
  } catch (e) {
    loading.classList.add('hidden');
    metrics.classList.add('hidden');
    none.classList.remove('hidden');
  }
}

function renderConfusionMatrix(containerId, cm, labels) {
  if (!cm.length || !labels.length) return;
  const trace = {
    z: cm,
    x: labels,
    y: labels,
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: true,
    hoverongaps: false
  };
  Plotly.newPlot(containerId, [trace], {
    margin: { t: 30, b: 60, l: 80, r: 40 },
    xaxis: { tickangle: -30 },
    yaxis: { autorange: 'reversed' },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#c8dcff', size: 10 }
  }, { displayModeBar: true, responsive: true });
}

function renderProbBars(data) {
  const fp = data.fusion_probabilities;
  const keys = Object.keys(fp);
  const colors = { 'Normal Perfusion': '#00ff88', 'Reduced Perfusion': '#ff8c00', 'Abnormal Perfusion': '#ff2244' };

  const ids = {
    'Normal Perfusion': ['perf-prob-normal', 'perf-prob-normal-bar'],
    'Reduced Perfusion': ['perf-prob-reduced', 'perf-prob-reduced-bar'],
    'Abnormal Perfusion': ['perf-prob-abnormal', 'perf-prob-abnormal-bar']
  };

  keys.forEach(key => {
    const pct = fp[key];
    const [pctId, barId] = ids[key] || [];
    if (!pctId) return;
    const pctEl = document.getElementById(pctId);
    const barEl = document.getElementById(barId);
    if (pctEl) pctEl.textContent = `${pct}%`;
    if (pctEl) pctEl.style.color = colors[key] || '#ccc';
    if (barEl) {
      barEl.style.background = colors[key] || '#ccc';
      setTimeout(() => { barEl.style.width = `${pct}%`; }, 100);
    }
  });

  const cm = COLOR_MAP[data.prediction];
  const final = document.getElementById('perf-final-row');
  if (final) {
    final.className = `final-label-row ${cm ? cm.badge : ''}`;
    final.textContent = `Final Result: ${data.prediction}`;
  }
}
