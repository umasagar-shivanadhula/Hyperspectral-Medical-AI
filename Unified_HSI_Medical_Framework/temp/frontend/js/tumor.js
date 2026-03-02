// ── TUMOR PAGE JS ──

let tumorFile = null;
let tumorSpectralChart = null;
let tumorRfChart = null;
let tumorSvmChart = null;

document.addEventListener('DOMContentLoaded', () => {
  loadTumorEvaluation();

  const dropZone = document.getElementById('tumor-drop-zone');
  const fileInput = document.getElementById('tumor-file-input');
  const browseBtn = document.getElementById('tumor-browse-btn');
  const runBtn = document.getElementById('tumor-run-btn');
  const demoBtn = document.getElementById('tumor-demo-btn');

  browseBtn.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) {
      tumorFile = e.target.files[0];
      document.getElementById('tumor-file-name').textContent = tumorFile.name;
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
      tumorFile = f;
      document.getElementById('tumor-file-name').textContent = f.name;
      runBtn.disabled = false;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!tumorFile) return;
    setLoading(true);
    try {
      const data = await HSI_API.predictTumor(tumorFile);
      renderTumorResults(data);
    } catch (err) {
      showTumorError(err.message);
    } finally {
      setLoading(false);
    }
  });

  demoBtn.addEventListener('click', () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      renderTumorResults(generateTumorDemo());
    }, 1200);
  });
});

function setLoading(on) {
  const btn = document.getElementById('tumor-run-btn');
  const spinner = document.getElementById('tumor-spinner');
  const txt = document.getElementById('tumor-btn-text');
  spinner.classList.toggle('hidden', !on);
  txt.textContent = on ? 'Detecting...' : 'Detect Tumor';
  btn.disabled = on;
}

function showTumorError(msg) {
  document.getElementById('tumor-error-text').textContent = msg;
  document.getElementById('tumor-error-panel').classList.remove('hidden');
}

function resetTumor() {
  document.getElementById('tumor-error-panel').classList.add('hidden');
}

const TUMOR_COLOR_MAP = {
  'Non-Tumor Tissue': { dot: '#0096ff', bar: 'rgba(0,150,255,0.6)', badge: 'badge-nontumor' },
  'Glioblastoma Tumor': { dot: '#ff2244', bar: 'rgba(255,34,68,0.6)', badge: 'badge-tumor' }
};

function renderTumorResults(data) {
  document.getElementById('tumor-error-panel').classList.add('hidden');

  const summaryPanel = document.getElementById('tumor-summary-panel');
  summaryPanel.classList.remove('hidden');
  const cm = TUMOR_COLOR_MAP[data.prediction] || TUMOR_COLOR_MAP['Non-Tumor Tissue'];
  const label = document.getElementById('tumor-summary-label');
  label.textContent = data.prediction;
  label.style.color = cm.dot;
  document.getElementById('tumor-summary-conf').textContent = ` (Confidence: ${data.confidence}%)`;
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
    z: heatmapData,
    type: 'heatmap',
    colorscale: [[0, '#0055ff'], [0.5, '#8800ff'], [1, '#ff2244']],
    showscale: false
  }], {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { visible: false },
    yaxis: { visible: false }
  }, { displayModeBar: false, responsive: true });
}

function renderTumorSpectralChart(sig) {
  if (tumorSpectralChart) tumorSpectralChart.destroy();
  const ctx = document.getElementById('tumor-spectral-chart').getContext('2d');
  tumorSpectralChart = new Chart(ctx, {
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
          borderColor: '#0096ff',
          backgroundColor: 'rgba(0,150,255,0.06)',
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

function renderTumorFusionSummary(data) {
  const container = document.getElementById('tumor-fusion-summary');
  const fp = data.fusion_probabilities;
  container.innerHTML = '';
  Object.entries(fp).forEach(([label, pct]) => {
    const cm = TUMOR_COLOR_MAP[label];
    const isWinner = label === data.prediction;
    const row = document.createElement('div');
    row.className = `fusion-row ${isWinner ? 'winner' : ''}`;
    row.innerHTML = `
      <span class="fusion-row-label" style="color:${cm ? cm.dot : '#ccc'}">${label}</span>
      <span class="fusion-row-pct" style="color:${cm ? cm.dot : '#ccc'}">${pct}%</span>`;
    container.appendChild(row);
  });

  const badge = document.getElementById('tumor-final-badge');
  const cm = TUMOR_COLOR_MAP[data.prediction];
  badge.className = `final-result-badge ${cm ? cm.badge : ''}`;
  badge.textContent = `FINAL: ${data.prediction.toUpperCase()}`;
}

function renderTumorClassifierCharts(data) {
  const rf = data.classifiers.random_forest;
  const svm = data.classifiers.svm;

  document.getElementById('tumor-rf-label').textContent = rf.prediction;
  document.getElementById('tumor-rf-conf-val').textContent = `${rf.confidence}%`;
  document.getElementById('tumor-svm-label').textContent = svm.prediction;
  document.getElementById('tumor-svm-conf-val').textContent = `${svm.confidence}%`;

  if (tumorRfChart) tumorRfChart.destroy();
  const rfCtx = document.getElementById('tumor-rf-chart').getContext('2d');
  tumorRfChart = new Chart(rfCtx, tumorBarChart(rf.probabilities, '#0096ff'));

  if (tumorSvmChart) tumorSvmChart.destroy();
  const svmCtx = document.getElementById('tumor-svm-chart').getContext('2d');
  tumorSvmChart = new Chart(svmCtx, tumorBarChart(svm.probabilities, '#ff4466'));
}

function tumorBarChart(probabilities, color) {
  return {
    type: 'bar',
    data: {
      labels: Object.keys(probabilities).map(k => k.replace(' Tissue', '').replace('Glioblastoma ', 'GBM ')),
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

async function loadTumorEvaluation() {
  const loading = document.getElementById('tumor-eval-loading');
  const metrics = document.getElementById('tumor-eval-metrics');
  const none = document.getElementById('tumor-eval-none');
  try {
    const data = await HSI_API.getEvaluationTumor();
    loading.classList.add('hidden');
    if (data && data.fusion) {
      none.classList.add('hidden');
      metrics.classList.remove('hidden');
      document.getElementById('tumor-eval-accuracy').textContent = (data.fusion.accuracy * 100).toFixed(2) + '%';
      document.getElementById('tumor-eval-f1').textContent = (data.fusion.f1_score * 100).toFixed(2) + '%';
      const labels = data.fusion.confusion_matrix_labels || data.classes || [];
      const cm = data.fusion.confusion_matrix || [];
      renderTumorConfusionMatrix('tumor-confusion-matrix', cm, labels);
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

function renderTumorConfusionMatrix(containerId, cm, labels) {
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

function renderTumorProbBars(data) {
  const fp = data.fusion_probabilities;
  const colors = { 'Non-Tumor Tissue': '#0096ff', 'Glioblastoma Tumor': '#ff2244' };
  const ids = {
    'Non-Tumor Tissue': ['tumor-prob-normal', 'tumor-prob-normal-bar'],
    'Glioblastoma Tumor': ['tumor-prob-tumor', 'tumor-prob-tumor-bar']
  };

  Object.keys(fp).forEach(key => {
    const pct = fp[key];
    const [pctId, barId] = ids[key] || [];
    if (!pctId) return;
    const pctEl = document.getElementById(pctId);
    const barEl = document.getElementById(barId);
    if (pctEl) { pctEl.textContent = `${pct}%`; pctEl.style.color = colors[key]; }
    if (barEl) {
      barEl.style.background = colors[key];
      setTimeout(() => { barEl.style.width = `${pct}%`; }, 100);
    }
  });

  const cm = TUMOR_COLOR_MAP[data.prediction];
  const final = document.getElementById('tumor-final-row');
  if (final) {
    final.className = `final-label-row ${cm ? cm.badge : ''}`;
    final.textContent = `Final Result: ${data.prediction}`;
  }
}
