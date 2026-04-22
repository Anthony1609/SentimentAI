/**
 * SentimentAI — High-End Frontend Architecture
 */

// ─── DOM References ────────────────────────────────────────────
const textInput = document.getElementById('text-input');
const charCounter = document.getElementById('char-counter');
const btnAnalyze = document.getElementById('btn-analyze');
const btnUpload = document.getElementById('btn-upload');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const removeFile = document.getElementById('remove-file');
const loadingOverlay = document.getElementById('loading-overlay');
const singleResult = document.getElementById('single-result');
const batchResult = document.getElementById('batch-result');

let emotionChart = null;
let batchChart = null;
let historyChart = null;
let selectedFile = null;

// Premium Color Palette
const COLORS = {
    pos: '#10b981', // Emerald
    neg: '#f43f5e', // Rose
    neu: '#f59e0b', // Amber
    posBg: 'rgba(16, 185, 129, 0.7)',
    negBg: 'rgba(244, 63, 94, 0.7)',
    neuBg: 'rgba(245, 158, 11, 0.7)',
    accent: '#6366f1',
    accentBg: 'rgba(99, 102, 241, 0.15)',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#94a3b8'
};

// ─── Initialize ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Re-init lucide icons incase dynamic elements need it
    lucide.createIcons();

    if (textInput) {
        initTextInput();
        initFileUpload();
        initAnalyzeButton();
        initUploadButton();
    }
    
    if (document.getElementById('history-table-body') || document.getElementById('dashboard-chart')) {
        loadHistory();
        document.getElementById('btn-refresh')?.addEventListener('click', loadHistory);
    }
});

const SAMPLES = {
    '1': "Oh, I just absolutely love it when the software crashes in the middle of saving my four-hour project. It's truly a masterpiece of engineering.",
    '2': "When I first opened the package, I was genuinely horrified by the cracked screen. But after peeling off the protective film, I realized the device itself is stunningly beautiful and runs incredibly fast.",
    '3': "The quarterly earnings report indicates a 2.4% variance in expected revenue over the fiscal year. The board meeting is scheduled for Tuesday."
};

// ─── Text Input ────────────────────────────────────────────────
function initTextInput() {
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCounter.textContent = `${len} / 5000`;
        
        // Color coding
        if (len > 4500) charCounter.style.color = COLORS.neg;
        else if (len > 4000) charCounter.style.color = COLORS.neu;
        else charCounter.style.color = 'var(--text-muted)';

        btnAnalyze.disabled = len === 0;
    });

    textInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter' && textInput.value.trim()) {
            e.preventDefault();
            btnAnalyze.click();
        }
    });

    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            textInput.value = SAMPLES[e.target.dataset.sample];
            textInput.dispatchEvent(new Event('input'));
        });
    });
}

// ─── File Upload ───────────────────────────────────────────────
function initFileUpload() {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    removeFile.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.remove('active');
        btnUpload.disabled = true;
    });
}

function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['txt', 'csv'].includes(ext)) {
        showToast('Only .txt and .csv files are supported', 'error');
        return;
    }

    selectedFile = file;
    fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    fileInfo.classList.add('active');
    btnUpload.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ─── Analyze Button ────────────────────────────────────────────
function initAnalyzeButton() {
    btnAnalyze.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;

        const model = document.querySelector('input[name="model"]:checked').value;

        showLoading();

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model }),
            });

            const data = await res.json();
            hideLoading();

            if (!res.ok) {
                showToast(data.error || 'Analysis failed', 'error');
                return;
            }

            batchResult.classList.remove('active');
            renderSingleResult(data);
        } catch (e) {
            hideLoading();
            showToast('Network error. Please try again.', 'error');
        }
    });
}

// ─── Upload Button ─────────────────────────────────────────────
function initUploadButton() {
    btnUpload.addEventListener('click', async () => {
        if (!selectedFile) return;

        const model = document.querySelector('input[name="model-file"]:checked').value;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', model);

        showLoading();

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();
            hideLoading();

            if (!res.ok) {
                showToast(data.error || 'Upload failed', 'error');
                return;
            }

            singleResult.classList.remove('active');
            renderBatchResult(data);
        } catch (e) {
            hideLoading();
            showToast('Network error. Please try again.', 'error');
        }
    });
}

// ─── Render Single Result ──────────────────────────────────────
function renderSingleResult(data) {
    singleResult.classList.add('active'); // Make visible before rendering charts
    
    const s = data.sentiment;
    const v = data.vader;
    const e = data.emotions;

    // Badge
    const badge = document.getElementById('result-badge');
    badge.innerHTML = `<i data-lucide="${getIconForSentiment(s.label)}"></i> ${s.label}`;
    badge.className = `status-badge ${s.label.toLowerCase()}`;

    document.getElementById('result-model').textContent = `Model: ${s.model_used}`;
    document.getElementById('result-confidence').textContent = `${s.confidence.toFixed(2)}%`;

    // Probabilities
    ['Positive', 'Negative', 'Neutral'].forEach(cls => {
        const key = cls.toLowerCase();
        const val = s.probabilities[cls] || 0;
        const barEl = document.getElementById(`prob-${key}`);
        const valEl = document.getElementById(`prob-${key}-val`);
        if (barEl) {
            setTimeout(() => { barEl.style.width = `${val}%`; }, 50);
        }
        if (valEl) valEl.textContent = `${val.toFixed(1)}%`;
    });

    // Gauge
    animateGauge(v.intensity_pct, v.compound);

    // VADER scores
    document.getElementById('vader-pos').textContent = v.pos.toFixed(3);
    document.getElementById('vader-neg').textContent = v.neg.toFixed(3);
    document.getElementById('vader-neu').textContent = v.neu.toFixed(3);

    // Emotion chart
    renderEmotionChart(e.emotions);

    // Top emotion
    const topEl = document.getElementById('top-emotion');
    if (e.top_emotion && e.top_emotion !== 'none') {
        topEl.style.display = 'inline-flex';
        document.getElementById('top-emotion-name').textContent = e.top_emotion;
    } else {
        topEl.style.display = 'none';
    }

    document.getElementById('result-text').textContent = data.text;
    lucide.createIcons();
}

// ─── Render Batch Result ───────────────────────────────────────
function renderBatchResult(data) {
    batchResult.classList.add('active'); // Make visible before rendering charts
    
    document.getElementById('batch-total').textContent = data.total;
    document.getElementById('batch-pos').textContent = data.summary.Positive || 0;
    document.getElementById('batch-neg').textContent = data.summary.Negative || 0;
    document.getElementById('batch-neu').textContent = data.summary.Neutral || 0;

    renderDoughnutChart('batch-chart', batchChart, data.summary);

    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = data.results.map((r, i) => {
        const cls = r.label.toLowerCase();
        return `
            <tr>
                <td>${i + 1}</td>
                <td class="cell-text" title="${escapeHtml(r.text)}">${escapeHtml(r.text)}</td>
                <td><span class="status-badge ${cls}" style="font-size: 0.65rem; padding: 2px 8px;">${r.label}</span></td>
                <td style="font-family: var(--font-mono)">${r.confidence.toFixed(1)}%</td>
            </tr>
        `;
    }).join('');
}

// ─── Gauge Animation ───────────────────────────────────────────
function animateGauge(pct, compound) {
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');

    const arcLength = 283; // Approx for r=80 half circle 
    const offset = arcLength - (arcLength * (pct / 100));

    let color = COLORS.neu;
    if (compound >= 0.3) color = COLORS.pos;
    else if (compound <= -0.3) color = COLORS.neg;

    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
        gaugeFill.style.stroke = color;
    }, 50);

    gaugeValue.textContent = compound.toFixed(3);
    gaugeValue.style.color = color;
}

// ─── Emotion Chart ─────────────────────────────────────────────
function renderEmotionChart(emotions) {
    const ctx = document.getElementById('emotion-chart');
    if (!ctx) return;

    if (emotionChart) emotionChart.destroy();

    const labels = ['joy', 'trust', 'anticipation', 'surprise', 'fear', 'sadness', 'anger', 'disgust'];
    const values = labels.map(l => (emotions[l] || 0) * 100);

    // Dark SaaS theme configuration
    Chart.defaults.color = COLORS.text;
    Chart.defaults.font.family = "'Inter', sans-serif";

    emotionChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                label: 'Intensity',
                data: values,
                backgroundColor: COLORS.accentBg,
                borderColor: COLORS.accent,
                borderWidth: 1.5,
                pointBackgroundColor: COLORS.accent,
                pointBorderColor: '#fff',
                pointBorderWidth: 1,
                pointRadius: 3,
                pointHoverRadius: 5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: Math.max(30, ...values) + 5,
                    grid: { color: COLORS.grid },
                    angleLines: { color: COLORS.grid },
                    pointLabels: {
                        color: COLORS.text,
                        font: { size: 10 }
                    },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: getTooltipConfig()
            }
        }
    });
}

// ─── Doughnut Chart ────────────────────────────────────────────
function renderDoughnutChart(canvasId, chartInstance, summary) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    if (chartInstance) chartInstance.destroy();

    const newChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [summary.Positive || 0, summary.Negative || 0, summary.Neutral || 0],
                backgroundColor: [COLORS.posBg, COLORS.negBg, COLORS.neuBg],
                borderColor: [COLORS.pos, COLORS.neg, COLORS.neu],
                borderWidth: 1.5,
                hoverOffset: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { display: false },
                tooltip: getTooltipConfig()
            }
        }
    });

    if (canvasId === 'batch-chart') batchChart = newChart;
    if (canvasId === 'history-chart' || canvasId === 'dashboard-chart') historyChart = newChart;
}

// ─── History Data Loading ──────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/api/history?limit=100');
        const data = await res.json();
        
        if (!res.ok) return;

        // Render Summary for History Page
        const dist = data.distribution || {};
        if (document.getElementById('hist-total')) {
            document.getElementById('hist-total').textContent = data.total || 0;
            document.getElementById('hist-pos').textContent = dist.Positive || 0;
            document.getElementById('hist-neg').textContent = dist.Negative || 0;
            document.getElementById('hist-neu').textContent = dist.Neutral || 0;
            renderDoughnutChart('history-chart', historyChart, dist);
        }

        // Render Summary for Dashboard Page
        if (document.getElementById('dashboard-chart')) {
            document.getElementById('dashboard-total').textContent = data.total || 0;
            document.getElementById('dashboard-pos').textContent = dist.Positive || 0;
            document.getElementById('dashboard-neg').textContent = dist.Negative || 0;
            document.getElementById('dashboard-neu').textContent = dist.Neutral || 0;
            renderDoughnutChart('dashboard-chart', historyChart, dist);

            // Calculate majority sentiment
            let maxKey = 'None';
            let maxVal = -1;
            for (const [k, v] of Object.entries(dist)) {
                if (v > maxVal) {
                    maxVal = v;
                    maxKey = k;
                }
            }
            const topEl = document.getElementById('dashboard-top-sentiment');
            topEl.textContent = maxVal > 0 ? maxKey : '--';
            
            // Set color based on sentiment
            if (maxKey === 'Positive') topEl.style.color = COLORS.pos;
            else if (maxKey === 'Negative') topEl.style.color = COLORS.neg;
            else if (maxKey === 'Neutral') topEl.style.color = COLORS.neu;
        }

        // Render Table for History Page
        const tbody = document.getElementById('history-table-body');
        const records = data.history || [];
        if (tbody) {
            tbody.innerHTML = records.map(r => {
                const cls = r.sentiment.toLowerCase();
                const time = new Date(r.created_at).toLocaleString();
                return `
                    <tr>
                        <td style="font-size: 0.8rem">${time}</td>
                        <td class="cell-text" title="${escapeHtml(r.text)}">${escapeHtml(r.text)}</td>
                        <td><span class="status-badge ${cls}" style="font-size: 0.65rem; padding: 2px 8px;">${r.sentiment}</span></td>
                        <td style="font-family: var(--font-mono)">${r.confidence.toFixed(1)}%</td>
                        <td style="font-size: 0.8rem">${r.model_used.replace(' Regression', '')}</td>
                    </tr>
                `;
            }).join('');
        }

    } catch (e) {
        console.error('Failed to load history', e);
    }
}

// ─── Helpers ───────────────────────────────────────────────────
function showLoading() {
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = type === 'error' 
        ? `<i data-lucide="alert-circle" style="vertical-align: text-bottom; margin-right: 6px;"></i> ${message}`
        : `<i data-lucide="info" style="vertical-align: text-bottom; margin-right: 6px;"></i> ${message}`;
    container.appendChild(toast);
    lucide.createIcons({ root: toast });
    setTimeout(() => toast.remove(), 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getIconForSentiment(label) {
    if (label === 'Positive') return 'trending-up';
    if (label === 'Negative') return 'trending-down';
    return 'minus';
}

function getTooltipConfig() {
    return {
        backgroundColor: 'rgba(3, 7, 18, 0.95)',
        titleColor: '#f8fafc',
        bodyColor: '#94a3b8',
        bodyFont: { family: "'JetBrains Mono', monospace" },
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        cornerRadius: 6,
        padding: 12,
        boxPadding: 6,
    };
}
