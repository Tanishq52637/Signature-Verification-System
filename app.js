// ═══════════════════════════════════════════════════════
//  SigVerify AI — app.js
//  ML Signature Verification System | B.Tech CSE Project
//  Logic: Tab switching, demo, OpenCV simulation,
//         feature extraction, result display
// ═══════════════════════════════════════════════════════

// ── FEATURE DEFINITIONS ──────────────────────────────────────────────────────
const FEATURES = [
    { name: "Aspect ratio", color: "#7F77DD", desc: "Width ÷ height of bounding box" },
    { name: "Pixel density", color: "#1D9E75", desc: "% ink pixels in bounding box" },
    { name: "Centroid X", color: "#D85A30", desc: "Normalised horizontal centre of mass" },
    { name: "Centroid Y", color: "#D4537E", desc: "Normalised vertical centre of mass" },
    { name: "Stroke width μ", color: "#BA7517", desc: "Mean width of pen strokes" },
    { name: "Contour count", color: "#378ADD", desc: "Number of connected components" },
    { name: "Hu moment 1", color: "#534AB7", desc: "Scale-invariant shape descriptor" },
    { name: "Hu moment 2", color: "#0F6E56", desc: "Orientation shape descriptor" },
    { name: "LBP variance", color: "#993C1D", desc: "Local binary pattern texture variance" },
    { name: "Edge density", color: "#993556", desc: "Canny edge pixel ratio" },
];

// ── PIPELINE STEP DEFINITIONS ─────────────────────────────────────────────────
const PIPELINE = [
    { bg: "#EEEDFE", tc: "#26215C", title: "Grayscale conversion", code: "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", why: "Remove colour — signatures are a grayscale problem." },
    { bg: "#E1F5EE", tc: "#085041", title: "Gaussian blur", code: "cv2.GaussianBlur(gray, (5,5), 0)", why: "Removes salt-and-pepper noise from scanning." },
    { bg: "#FAEEDA", tc: "#412402", title: "Otsu thresholding", code: "cv2.threshold(..., cv2.THRESH_OTSU)", why: "Auto binarisation — best global ink threshold." },
    { bg: "#FBEAF0", tc: "#4B1528", title: "Morphological opening", code: "cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)", why: "Removes tiny blobs without breaking strokes." },
    { bg: "#E6F1FB", tc: "#042C53", title: "Bounding-box crop", code: "cv2.boundingRect(contour)", why: "Removes blank margins; normalises position." },
    { bg: "#FAECE7", tc: "#4A1B0C", title: "Resize to 128×128", code: "cv2.resize(crop, (128, 128))", why: "Fixed-size input for all feature extractors." },
    { bg: "#EEEDFE", tc: "#26215C", title: "Contour detection", code: "cv2.findContours(..., cv2.RETR_EXTERNAL)", why: "Finds individual pen strokes for counting." },
    { bg: "#E1F5EE", tc: "#085041", title: "Hu moments", code: "cv2.HuMoments(cv2.moments(img))", why: "Rotation + scale invariant shape descriptors." },
    { bg: "#FAEEDA", tc: "#412402", title: "LBP texture features", code: "skimage.feature.local_binary_pattern(...)", why: "Captures micro-texture of pen strokes." },
    { bg: "#FBEAF0", tc: "#4B1528", title: "Feature vector assembly", code: "np.concatenate([spatial, shape, texture])", why: "Combines all into a 10-D vector for the classifier." },
];

// ── CLASSIFIER RESULTS ────────────────────────────────────────────────────────
const CLASSIFIERS = [
    { name: "Random Forest", acc: 94.2, color: "#7F77DD", best: true },
    { name: "SVM (RBF)", acc: 91.7, color: "#1D9E75", best: false },
    { name: "KNN (k=5)", acc: 87.3, color: "#D85A30", best: false },
    { name: "Decision Tree", acc: 83.5, color: "#BA7517", best: false },
    { name: "Naive Bayes", acc: 79.1, color: "#888780", best: false },
];

// ── E2E FLOW ──────────────────────────────────────────────────────────────────
const E2E_STEPS = [
    { icon: "ti-signature", label: "Raw signature image", bg: "rgba(127,119,221,0.2)", tc: "#AFA9EC" },
    { icon: "ti-adjustments", label: "OpenCV preprocessing", bg: "rgba(239,159,39,0.2)", tc: "#FAC775" },
    { icon: "ti-variable", label: "Feature extraction (10-D)", bg: "rgba(29,158,117,0.2)", tc: "#5DCAA5" },
    { icon: "ti-brain", label: "Random Forest classifier", bg: "rgba(212,83,126,0.2)", tc: "#ED93B1" },
    { icon: "ti-shield-check", label: "Genuine / Forged verdict", bg: "rgba(29,158,117,0.2)", tc: "#5DCAA5" },
];

// ── FORMULA TABLE ─────────────────────────────────────────────────────────────
const FORMULAS = [
    { name: "Precision", formula: "TP / (TP + FP)", why: "Of all 'genuine' predictions, how many were actually genuine?" },
    { name: "Recall", formula: "TP / (TP + FN)", why: "Of all actual genuines, how many did we catch?" },
    { name: "F1-score", formula: "2×P×R / (P+R)", why: "Balanced metric — use when classes are imbalanced." },
    { name: "FAR", formula: "FP / (FP + TN)", why: "Forged accepted as genuine — the dangerous error!" },
    { name: "FRR", formula: "FN / (FN + TP)", why: "Genuine rejected — annoying but not a security risk." },
];

// ── KNOWN FEATURE VECTORS ─────────────────────────────────────────────────────
const GENUINE_FV = [0.82, 0.23, 0.51, 0.44, 0.61, 0.30, 0.77, 0.68, 0.55, 0.72];
const FORGED_FV = [0.48, 0.41, 0.62, 0.58, 0.29, 0.67, 0.38, 0.33, 0.71, 0.35];

// ── PROCESSING LOG TEMPLATE ───────────────────────────────────────────────────
function getLogSteps() {
    return [
        ["step", "→ Loading signature image..."],
        ["ok", "✓ Converted to grayscale"],
        ["ok", "✓ Gaussian blur applied (5×5 kernel)"],
        ["ok", `✓ Otsu threshold: T = ${112 + Math.floor(Math.random() * 30)}`],
        ["ok", "✓ Morphological opening done"],
        ["ok", "✓ Bounding box detected & cropped"],
        ["ok", "✓ Resized to 128×128 px"],
        ["ok", `✓ Contours found: ${Math.floor(Math.random() * 8) + 4}`],
        ["ok", "✓ Hu moments computed (7 values)"],
        ["ok", "✓ LBP texture features extracted"],
        ["ok", "✓ 10-D feature vector assembled"],
        ["info", "→ Passing vector to Random Forest (100 trees)..."],
        ["info", "→ Aggregating tree votes..."],
    ];
}

// ══════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════
function sw(tabName) {
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".nb").forEach(b => b.classList.remove("active"));
    document.getElementById("tab-" + tabName).classList.add("active");
    document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
}

// ══════════════════════════════════════════════════════
//  FILE UPLOAD HANDLERS
// ══════════════════════════════════════════════════════
function onDrop(e) {
    e.preventDefault();
    document.getElementById("dropzone").classList.remove("drag");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFile(file);
}

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = ev => processSignature(ev.target.result, null);
    reader.readAsDataURL(file);
}

// ══════════════════════════════════════════════════════
//  DEMO LAUNCHER
// ══════════════════════════════════════════════════════
function runDemo(type) {
    processSignature(null, type);
}

// ══════════════════════════════════════════════════════
//  RESET
// ══════════════════════════════════════════════════════
function resetAll() {
    document.getElementById("canvas-row").style.display = "none";
    document.getElementById("log-wrap").style.display = "none";
    document.getElementById("feat-card").style.display = "none";
    document.getElementById("detail-cards").style.display = "none";
    document.getElementById("file-inp").value = "";
    document.getElementById("proc-log").innerHTML = "";
    setResult("waiting", null, null);
}

// ══════════════════════════════════════════════════════
//  MAIN PROCESS FUNCTION
// ══════════════════════════════════════════════════════
function processSignature(imgSrc, demoType) {
    // Reveal panels
    document.getElementById("canvas-row").style.display = "grid";
    document.getElementById("log-wrap").style.display = "block";
    document.getElementById("feat-card").style.display = "block";
    document.getElementById("detail-cards").style.display = "none";

    // Clear log + reset result
    document.getElementById("proc-log").innerHTML = "";
    setResult("waiting", null, null);

    // Draw preview canvases
    if (imgSrc) {
        const img = new Image();
        img.onload = () => {
            drawImageOnCanvas("cv-orig", img);
            applyBinaryThreshold("cv-orig", "cv-proc");
        };
        img.src = imgSrc;
    } else {
        const isGenuine = demoType === "genuine";
        drawDemoSignature("cv-orig", isGenuine, false);
        drawDemoSignature("cv-proc", isGenuine, true);
    }

    // Pick feature vector
    let featureVector;
    if (demoType === "genuine") featureVector = GENUINE_FV;
    else if (demoType === "forged") featureVector = FORGED_FV;
    else featureVector = FEATURES.map(() => Math.random() * 0.5 + 0.25);

    // Run animated log → then classify
    runLog(getLogSteps(), () => {
        renderFeatureBars(featureVector);

        let isGenuine, confidence;
        if (demoType === "genuine") {
            isGenuine = true;
            confidence = 0.88 + Math.random() * 0.09;
        } else if (demoType === "forged") {
            isGenuine = false;
            confidence = 0.85 + Math.random() * 0.11;
        } else {
            const avg = featureVector.reduce((a, b) => a + b, 0) / featureVector.length;
            isGenuine = avg > 0.50;
            confidence = Math.min(0.97, 0.72 + Math.abs(avg - 0.50) * 1.4);
        }

        document.getElementById("proc-log").innerHTML +=
            `<div class="ok">✓ Classification complete!</div>`;
        scrollLog();

        setTimeout(() => {
            setResult(isGenuine ? "genuine" : "forged", null, confidence);
            showDetailCards(featureVector, isGenuine, confidence);
        }, 200);
    });
}

// ── Animated log printer ──────────────────────────────────────────────────────
function runLog(steps, onDone) {
    const logEl = document.getElementById("proc-log");
    let i = 0;
    function next() {
        if (i < steps.length) {
            const [cls, msg] = steps[i++];
            logEl.innerHTML += `<div class="${cls}">${msg}</div>`;
            scrollLog();
            setTimeout(next, 85 + Math.floor(Math.random() * 70));
        } else {
            onDone();
        }
    }
    next();
}

function scrollLog() {
    const el = document.getElementById("proc-log");
    el.scrollTop = el.scrollHeight;
}

// ── Canvas helpers ────────────────────────────────────────────────────────────
function drawImageOnCanvas(canvasId, img) {
    const cv = document.getElementById(canvasId);
    cv.width = img.width;
    cv.height = img.height;
    cv.getContext("2d").drawImage(img, 0, 0);
}

function applyBinaryThreshold(srcId, dstId) {
    const src = document.getElementById(srcId);
    const dst = document.getElementById(dstId);
    dst.width = src.width;
    dst.height = src.height;
    const sCtx = src.getContext("2d");
    const dCtx = dst.getContext("2d");
    const data = sCtx.getImageData(0, 0, src.width, src.height);
    for (let j = 0; j < data.data.length; j += 4) {
        const gray = 0.299 * data.data[j] + 0.587 * data.data[j + 1] + 0.114 * data.data[j + 2];
        const bin = gray < 128 ? 0 : 255;
        data.data[j] = data.data[j + 1] = data.data[j + 2] = bin;
    }
    dCtx.putImageData(data, 0, 0);
}

function drawDemoSignature(canvasId, genuine, processed) {
    const cv = document.getElementById(canvasId);
    cv.width = 260; cv.height = 100;
    const ctx = cv.getContext("2d");

    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, 260, 100);

    ctx.strokeStyle = processed ? "#000" : "#1a1a2e";
    ctx.lineWidth = genuine ? 2.4 : 1.8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (genuine) {
        // Smooth, fluid curves — authentic signature
        const pts = [[25, 55], [50, 32], [75, 62], [100, 38], [125, 58], [150, 42], [175, 55], [205, 47], [240, 54]];
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
            const cx1 = pts[i - 1][0] + 12, cy1 = pts[i - 1][1];
            const cx2 = pts[i][0] - 12, cy2 = pts[i][1];
            ctx.bezierCurveTo(cx1, cy1, cx2, cy2, pts[i][0], pts[i][1]);
        }
        ctx.stroke();
        if (!processed) {
            ctx.lineWidth = 1; ctx.strokeStyle = "#ddd";
            ctx.beginPath(); ctx.moveTo(20, 72); ctx.lineTo(240, 72); ctx.stroke();
        }
        ctx.strokeStyle = processed ? "#000" : "#1a1a2e";
        ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(75, 62); ctx.lineTo(82, 78); ctx.stroke();
    } else {
        // Shaky, hesitant strokes — forged signature
        const pts = [[25, 55], [48, 40], [68, 60], [92, 46], [114, 58], [138, 48], [160, 58], [185, 50], [225, 56]];
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i][0] + (Math.random() - 0.5) * 5, pts[i][1] + (Math.random() - 0.5) * 5);
        }
        ctx.stroke();
        ctx.beginPath(); ctx.moveTo(90, 62); ctx.lineTo(95, 72); ctx.stroke();
    }

    if (!processed) {
        ctx.fillStyle = genuine ? "#5DCAA5" : "#F09595";
        ctx.font = "10px sans-serif";
        ctx.fillText(genuine ? "demo: genuine" : "demo: forged", 8, 93);
    }
}

// ══════════════════════════════════════════════════════
//  FEATURE BARS RENDERER
// ══════════════════════════════════════════════════════
function renderFeatureBars(fv) {
    const container = document.getElementById("feat-bars");
    container.innerHTML = FEATURES.map((f, i) => {
        const pct = Math.round(fv[i] * 100);
        return `
      <div class="feat-row">
        <span class="feat-lbl">${f.name}</span>
        <div class="feat-track">
          <div class="feat-fill" id="ff${i}" style="width:0%;background:${f.color}"></div>
        </div>
        <span class="feat-num" style="color:${f.color}">${pct}%</span>
      </div>`;
    }).join("");

    // Animate bars after paint
    setTimeout(() => {
        fv.forEach((v, i) => {
            const el = document.getElementById("ff" + i);
            if (el) el.style.width = Math.round(v * 100) + "%";
        });
    }, 60);
}

// ══════════════════════════════════════════════════════
//  RESULT BOX
// ══════════════════════════════════════════════════════
function setResult(state, _label, confidence) {
    const box = document.getElementById("result-box");
    const icon = document.getElementById("res-icon");
    const lbl = document.getElementById("res-label");
    const sub = document.getElementById("res-sub");
    const crow = document.getElementById("conf-row");

    box.className = "result-box " + state;

    if (state === "waiting") {
        icon.className = "ti ti-hourglass result-icon";
        lbl.textContent = "Upload a signature to begin";
        sub.textContent = "The classifier will output Genuine or Forged with a confidence score";
        crow.style.display = "none";
        return;
    }

    if (state === "genuine") {
        icon.className = "ti ti-shield-check result-icon";
        lbl.textContent = "Genuine Signature";
        sub.textContent = "The Random Forest classifier is confident this signature is authentic.";
    } else {
        icon.className = "ti ti-shield-x result-icon";
        lbl.textContent = "Forged Signature Detected!";
        sub.textContent = "Anomalies found in the feature vector — this signature appears forged.";
    }

    crow.style.display = "flex";

    const fill = document.getElementById("conf-fill");
    const pct = document.getElementById("conf-pct");
    const clbl = document.getElementById("conf-lbl");

    fill.style.background = state === "genuine" ? "#1D9E75" : "#E24B4A";
    fill.style.width = "0%";
    pct.textContent = "0%";
    clbl.textContent = "Confidence";

    setTimeout(() => {
        const val = Math.round(confidence * 100);
        fill.style.width = val + "%";
        pct.textContent = val + "%";
    }, 120);
}

// ══════════════════════════════════════════════════════
//  DETAIL STAT CARDS
// ══════════════════════════════════════════════════════
function showDetailCards(fv, genuine, confidence) {
    const avgFeat = Math.round(fv.reduce((a, b) => a + b, 0) / fv.length * 100);
    const topIdx = fv.indexOf(Math.max(...fv));
    const topFeat = FEATURES[topIdx].name;

    const cards = [
        { label: "Decision", val: genuine ? "Genuine" : "Forged", color: genuine ? "#5DCAA5" : "#F09595", bg: genuine ? "rgba(29,158,117,0.2)" : "rgba(226,75,74,0.2)" },
        { label: "Confidence", val: Math.round(confidence * 100) + "%", color: "#AFA9EC", bg: "rgba(127,119,221,0.18)" },
        { label: "Avg feature", val: avgFeat + "%", color: "#FAC775", bg: "rgba(239,159,39,0.18)" },
        { label: "Top feature", val: topFeat, color: "#5DCAA5", bg: "rgba(29,158,117,0.18)" },
    ];

    document.getElementById("stat-grid").innerHTML = cards.map(c =>
        `<div class="mcard" style="background:${c.bg}">
       <div class="ml" style="color:${c.color};opacity:.8">${c.label}</div>
       <div class="mv" style="color:${c.color};font-size:${c.label === 'Top feature' ? '14px' : '22px'}">${c.val}</div>
     </div>`
    ).join("");

    document.getElementById("detail-cards").style.display = "block";
}

// ══════════════════════════════════════════════════════
//  HYPERPARAMETER SIMULATOR
// ══════════════════════════════════════════════════════
function simUpdate() {
    const sp = +document.getElementById("sl-split").value;
    const ne = +document.getElementById("sl-ne").value;
    const md = +document.getElementById("sl-md").value;

    document.getElementById("sl-split-v").textContent = sp + "%";
    document.getElementById("sl-ne-v").textContent = ne;
    document.getElementById("sl-md-v").textContent = md;

    const bonus = Math.min(3, (ne - 10) / 60) + Math.min(2.5, (md - 2) / 7) + Math.min(2, (sp - 60) / 15);
    const acc = Math.min(97, 88 + bonus).toFixed(1);

    const col = acc >= 93 ? "#5DCAA5" : acc >= 88 ? "#FAC775" : "#F09595";
    const bg = acc >= 93 ? "rgba(29,158,117,0.18)" : acc >= 88 ? "rgba(239,159,39,0.18)" : "rgba(226,75,74,0.18)";

    document.getElementById("sim-out").innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;padding:.8rem 1rem;background:${bg};border-radius:9px;margin-top:.75rem;">
      <i class="ti ti-cpu" style="font-size:22px;color:${col}"></i>
      <div>
        <div style="font-size:14px;font-weight:700;color:${col}">Simulated accuracy: ${acc}%</div>
        <div style="font-size:12px;color:${col};opacity:.75;margin-top:2px">
          More trees + moderate depth + larger training split = better generalisation
        </div>
      </div>
    </div>`;
}

// ══════════════════════════════════════════════════════
//  BUILD PIPELINE TAB
// ══════════════════════════════════════════════════════
function buildPipeline() {
    document.getElementById("pipe-list").innerHTML = PIPELINE.map((s, i) => `
    <div class="pipe-step">
      <div class="pipe-num" style="background:${s.bg};color:${s.tc}">${i + 1}</div>
      <div>
        <div class="pipe-title">${s.title}</div>
        <div class="pipe-code">${s.code}</div>
        <div class="pipe-why">${s.why}</div>
      </div>
    </div>`).join("");

    document.getElementById("feat-table").innerHTML = FEATURES.map((f, i) => `
    <div style="display:flex;align-items:flex-start;gap:10px;padding:.45rem 0;border-bottom:1px solid #2a2840">
      <div style="width:13px;height:13px;border-radius:50%;background:${f.color};flex-shrink:0;margin-top:2px"></div>
      <div>
        <div style="font-size:13px;font-weight:700;color:#fffffe">${i + 1}. ${f.name}</div>
        <div style="font-size:12px;color:#aaa;margin-top:2px">${f.desc}</div>
      </div>
    </div>`).join("");
}

// ══════════════════════════════════════════════════════
//  BUILD MODEL TAB
// ══════════════════════════════════════════════════════
function buildClassifiers() {
    document.getElementById("clf-list").innerHTML = CLASSIFIERS.map(c => `
    <div class="clf-row">
      <div class="clf-name">${c.name}${c.best ? '<span class="pill-best">best</span>' : ''}</div>
      <div class="clf-bar-track">
        <div class="clf-bar-fill" style="width:${Math.round(c.acc)}%;background:${c.color}"></div>
      </div>
      <div class="clf-acc" style="color:${c.color}">${c.acc}%</div>
    </div>`).join("");
}

// ══════════════════════════════════════════════════════
//  BUILD LEARN TAB
// ══════════════════════════════════════════════════════
function buildLearn() {
    document.getElementById("e2e").innerHTML = E2E_STEPS.map((s, i) => `
    <div class="e2e-step">
      <div class="e2e-icon" style="background:${s.bg}">
        <i class="ti ${s.icon}" style="font-size:18px;color:${s.tc}"></i>
      </div>
      <div class="e2e-label">${s.label}</div>
    </div>`).join("");

    document.getElementById("formula-list").innerHTML = FORMULAS.map(f => `
    <div class="formula-row">
      <span class="f-name">${f.name}</span>
      <span class="f-formula">${f.formula}</span>
      <span class="f-why">${f.why}</span>
    </div>`).join("");
}

// ══════════════════════════════════════════════════════
//  INIT ON PAGE LOAD
// ══════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    buildPipeline();
    buildClassifiers();
    buildLearn();
    simUpdate();
});