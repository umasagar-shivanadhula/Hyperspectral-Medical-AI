// ── API MODULE ──
const API_BASE = 'http://localhost:8000';

const HSI_API = {
  async predictPerfusion(file) {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${API_BASE}/predict/perfusion`, {
      method: 'POST', body: formData
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  },

  async predictTumor(file) {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${API_BASE}/predict/tumor`, {
      method: 'POST', body: formData
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  },

  async getEvaluationPerfusion() {
    const resp = await fetch(`${API_BASE}/predict/evaluation/perfusion`);
    if (!resp.ok) return null;
    return resp.json();
  },

  async getEvaluationTumor() {
    const resp = await fetch(`${API_BASE}/predict/evaluation/tumor`);
    if (!resp.ok) return null;
    return resp.json();
  }
};

// ── SYNTHETIC DEMO DATA GENERATORS ──

function generatePerfusionDemo() {
  const classes = ['Normal Perfusion', 'Reduced Perfusion', 'Abnormal Perfusion'];
  const probs = softmaxRand(3);
  const winner = probs.indexOf(Math.max(...probs));
  const rfProbs = softmaxRand(3);
  const svmProbs = softmaxRand(3);

  return {
    task: 'perfusion',
    prediction: classes[winner],
    confidence: +(probs[winner] * 100).toFixed(1),
    fusion_probabilities: {
      'Normal Perfusion': +(probs[0] * 100).toFixed(1),
      'Reduced Perfusion': +(probs[1] * 100).toFixed(1),
      'Abnormal Perfusion': +(probs[2] * 100).toFixed(1)
    },
    classifiers: {
      random_forest: {
        prediction: classes[rfProbs.indexOf(Math.max(...rfProbs))],
        confidence: +(Math.max(...rfProbs) * 100).toFixed(1),
        probabilities: {
          'Normal Perfusion': +(rfProbs[0] * 100).toFixed(1),
          'Reduced Perfusion': +(rfProbs[1] * 100).toFixed(1),
          'Abnormal Perfusion': +(rfProbs[2] * 100).toFixed(1)
        }
      },
      svm: {
        prediction: classes[svmProbs.indexOf(Math.max(...svmProbs))],
        confidence: +(Math.max(...svmProbs) * 100).toFixed(1),
        probabilities: {
          'Normal Perfusion': +(svmProbs[0] * 100).toFixed(1),
          'Reduced Perfusion': +(svmProbs[1] * 100).toFixed(1),
          'Abnormal Perfusion': +(svmProbs[2] * 100).toFixed(1)
        }
      }
    },
    spectral_signature: generateSpectralSignature(16),
    heatmap: generateHeatmapData(32, 32, winner),
    ai_summary: generatePerfusionSummary(classes[winner], probs[winner])
  };
}

function generateTumorDemo() {
  const classes = ['Non-Tumor Tissue', 'Glioblastoma Tumor'];
  const probs = softmaxRand(2);
  const winner = probs.indexOf(Math.max(...probs));
  const rfProbs = softmaxRand(2);
  const svmProbs = softmaxRand(2);

  return {
    task: 'tumor',
    prediction: classes[winner],
    confidence: +(probs[winner] * 100).toFixed(1),
    fusion_probabilities: {
      'Non-Tumor Tissue': +(probs[0] * 100).toFixed(1),
      'Glioblastoma Tumor': +(probs[1] * 100).toFixed(1)
    },
    classifiers: {
      random_forest: {
        prediction: classes[rfProbs.indexOf(Math.max(...rfProbs))],
        confidence: +(Math.max(...rfProbs) * 100).toFixed(1),
        probabilities: {
          'Non-Tumor Tissue': +(rfProbs[0] * 100).toFixed(1),
          'Glioblastoma Tumor': +(rfProbs[1] * 100).toFixed(1)
        }
      },
      svm: {
        prediction: classes[svmProbs.indexOf(Math.max(...svmProbs))],
        confidence: +(Math.max(...svmProbs) * 100).toFixed(1),
        probabilities: {
          'Non-Tumor Tissue': +(svmProbs[0] * 100).toFixed(1),
          'Glioblastoma Tumor': +(svmProbs[1] * 100).toFixed(1)
        }
      }
    },
    spectral_signature: generateSpectralSignature(16),
    heatmap: generateHeatmapData(32, 32, winner),
    ai_summary: generateTumorSummary(classes[winner], probs[winner])
  };
}

function softmaxRand(n) {
  const raw = Array.from({ length: n }, () => Math.random() * 3);
  const exp = raw.map(x => Math.exp(x));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

function generateSpectralSignature(bands) {
  const tissue = [], healthy = [];
  for (let i = 0; i < bands; i++) {
    const base = 40 + Math.sin(i * 0.5) * 20 + Math.random() * 15;
    tissue.push(+(base).toFixed(2));
    healthy.push(+(base + 10 + Math.random() * 10).toFixed(2));
  }
  return { bands: Array.from({ length: bands }, (_, i) => i + 1), tissue, healthy };
}

function generateHeatmapData(rows, cols, classIdx) {
  const data = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const cx = cols / 2, cy = rows / 2;
      const dist = Math.sqrt((r - cy) ** 2 + (c - cx) ** 2);
      const maxDist = Math.sqrt(cy * cy + cx * cx);
      let val = 1 - dist / maxDist + (Math.random() - 0.5) * 0.3;
      val = Math.max(0, Math.min(1, val));
      row.push(classIdx > 0 ? val : 1 - val);
    }
    data.push(row);
  }
  return data;
}

function generatePerfusionSummary(label, conf) {
  const msgs = {
    'Normal Perfusion': 'Normal blood flow detected throughout the tissue region. Oxygenation levels are within expected physiological range.',
    'Reduced Perfusion': 'Decreased blood flow detected in the tissue region. Spectral analysis indicates reduced oxygenation, consistent with partial vascular occlusion.',
    'Abnormal Perfusion': 'Significant perfusion abnormality detected. Severely reduced blood flow with oxygenation indicators below clinical threshold.'
  };
  return msgs[label] || 'Analysis complete.';
}

function generateTumorSummary(label, conf) {
  const msgs = {
    'Non-Tumor Tissue': 'No tumor signatures detected in the tissue sample. Spectral profile is consistent with healthy neural tissue histology.',
    'Glioblastoma Tumor': 'Glioblastoma tumor tissue identified in the histological sample. Spectral-spatial features match ROI_T classification patterns from training data.'
  };
  return msgs[label] || 'Analysis complete.';
}
