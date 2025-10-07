(() => {
  /**
   * Manifest format (example):
   * [
   *   {
   *     caseId: "Brooke_1/SS7324_1_32_be45...",
   *     roiIndex: 1,
   *     thumb: "data/Brooke_1/SS7324_.../thumbnail_with_box_1.jpeg",
   *     box: "data/Brooke_1/SS7324_.../box_1.jpeg",
   *     cyto: "data/Brooke_1/SS7324_.../cyto_box_1.jpeg",
   *     draft: "AI draft reasoning text for this ROI"
   *   },
   *   ...
   * ]
   */

  const dom = {
    btnStart: document.getElementById('btnStart'),
    btnNext: document.getElementById('btnNext'),
    btnAccept: document.getElementById('btnAccept'),
    btnReject: document.getElementById('btnReject'),
    btnEdit: document.getElementById('btnEdit'),
    btnDownloadJSON: document.getElementById('btnDownloadJSON'),
    btnDownloadCSV: document.getElementById('btnDownloadCSV'),
    statusCase: document.getElementById('statusCase'),
    statusRoi: document.getElementById('statusRoi'),
    statusIndex: document.getElementById('statusIndex'),
    statusTimer: document.getElementById('statusTimer'),
    statusDecision: document.getElementById('statusDecision'),
    imgThumb: document.getElementById('imgThumb'),
    imgBox: document.getElementById('imgBox'),
    imgCyto: document.getElementById('imgCyto'),
    editThumb: document.getElementById('editThumb'),
    editWhy: document.getElementById('editWhy'),
    editBox: document.getElementById('editBox'),
    editCyto: document.getElementById('editCyto'),
    footerMsg: document.getElementById('footerMsg'),
  };

  /** @type {Array<any>} */
  let manifestItems = [];
  let currentIndex = -1;
  let perItemStartMs = 0;
  let timerHandle = null;
  let decisionState = null; // 'accept' | 'reject' | 'edit' | null
  const logs = [];
  const sessionId = generateSessionId();

  const counters = {
    acceptClicks: 0,
    rejectClicks: 0,
    editClicks: 0,
    keystrokesThumb: 0,
    keystrokesWhy: 0,
    keystrokesBox: 0,
    keystrokesCyto: 0,
  };

  // Simplified decisions: single reject-all or accept via Next

  function generateSessionId() {
    return 'sess_' + Math.random().toString(36).slice(2) + '_' + Date.now();
  }

  async function loadManifest() {
    // Prefer fetching manifest.json when served over http://, fallback to inline window.MANIFEST
    try {
      const res = await fetch('manifest.json', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (_) {}
    if (Array.isArray(window.MANIFEST) && window.MANIFEST.length > 0) return window.MANIFEST;
    // Final fallback: try example
    try {
      const res2 = await fetch('manifest.example.json', { cache: 'no-store' });
      if (res2.ok) {
        const data2 = await res2.json();
        if (Array.isArray(data2) && data2.length > 0) return data2;
      }
    } catch (_) {}
    throw new Error('No manifest found. Ensure manifest.json is next to index.html');
  }

  function setFooter(msg) {
    dom.footerMsg.textContent = msg;
  }

  function formatElapsed(ms) {
    const totalSec = Math.floor(ms / 1000);
    const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
    const ss = String(totalSec % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  }

  function startPerItemTimer() {
    perItemStartMs = performance.now();
    stopPerItemTimer();
    timerHandle = setInterval(() => {
      const ms = performance.now() - perItemStartMs;
      dom.statusTimer.textContent = 'Elapsed: ' + formatElapsed(ms);
    }, 250);
  }

  function stopPerItemTimer() {
    if (timerHandle) {
      clearInterval(timerHandle);
      timerHandle = null;
    }
  }

  function updateStatusBar(itemIndex) {
    const item = manifestItems[itemIndex];
    dom.statusCase.textContent = 'Case: ' + (item?.caseId ?? '-');
    dom.statusRoi.textContent = 'ROI: ' + (item?.roiIndex ?? '-');
    dom.statusIndex.textContent = `Progress: ${itemIndex + 1}/${manifestItems.length}`;
  }

  function renderItem(itemIndex) {
    const item = manifestItems[itemIndex];
    if (!item) return;
    // Reset UI state
    dom.editThumb.value = item.thumbDraft || '';
    dom.editWhy.value = item.whyDraft || '';
    dom.editBox.value = item.boxDraft || '';
    dom.editCyto.value = item.cytoDraft || '';
    dom.imgThumb.src = item.thumb || '';
    dom.imgBox.src = item.box || '';
    dom.imgCyto.src = item.cyto || '';
    decisionState = null;
    dom.statusDecision.textContent = 'Decision: -';
    dom.btnNext.disabled = false;
    counters.acceptClicks = 0;
    counters.rejectClicks = 0;
    counters.editClicks = 0;
    counters.keystrokesThumb = 0;
    counters.keystrokesWhy = 0;
    counters.keystrokesBox = 0;
    counters.keystrokesCyto = 0;
    updateStatusBar(itemIndex);
    startPerItemTimer();
  }

  function nextItem() {
    currentIndex += 1;
    if (currentIndex >= manifestItems.length) {
      stopPerItemTimer();
      setFooter('All items completed. Use Download to export logs.');
      dom.btnNext.disabled = true;
      return;
    }
    renderItem(currentIndex);
  }

  function commitDecision(decision) {
    decisionState = decision; // 'accept' | 'reject' | 'edit'
    dom.statusDecision.textContent = 'Decision: ' + decision.toUpperCase();
    dom.btnNext.disabled = false;
  }

  function handleAccept() {
    counters.acceptClicks += 1;
    commitDecision('accept');
  }

  function handleReject() {
    counters.rejectClicks += 1;
    commitDecision('reject');
  }

  function handleEdit() {
    counters.editClicks += 1;
    commitDecision('edit');
  }

  function handleKeystrokeThumb() { counters.keystrokesThumb += 1; }
  function handleKeystrokeWhy() { counters.keystrokesWhy += 1; }
  function handleKeystrokeBox() { counters.keystrokesBox += 1; }
  function handleKeystrokeCyto() { counters.keystrokesCyto += 1; }

  function computeLevenshtein(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
          dp[i - 1][j - 1] + cost
        );
      }
    }
    return dp[m][n];
  }

  function recordAndAdvance() {
    const item = manifestItems[currentIndex];
    const endMs = performance.now();
    const elapsedMs = Math.round(endMs - perItemStartMs);
    stopPerItemTimer();

    const aiThumbText = manifestItems[currentIndex]?.thumbDraft || '';
    const aiWhyText = manifestItems[currentIndex]?.whyDraft || '';
    const aiBoxText = manifestItems[currentIndex]?.boxDraft || '';
    const aiCytoText = manifestItems[currentIndex]?.cytoDraft || '';
    const userThumb = dom.editThumb.value || '';
    const userWhy = dom.editWhy.value || '';
    const userBox = dom.editBox.value || '';
    const userCyto = dom.editCyto.value || '';
    const editDistanceThumb = computeLevenshtein(aiThumbText, userThumb || aiThumbText);
    const editDistanceWhy = computeLevenshtein(aiWhyText, userWhy || aiWhyText);
    const editDistanceBox = computeLevenshtein(aiBoxText, userBox || aiBoxText);
    const editDistanceCyto = computeLevenshtein(aiCytoText, userCyto || aiCytoText);

    const logEntry = {
      sessionId,
      timestamp: new Date().toISOString(),
      caseId: item.caseId,
      roiIndex: item.roiIndex,
      decision: decisionState, // accept | reject | edit (overall; set below)
      elapsedMs,
      counters: { ...counters },
      aiThumbLength: aiThumbText.length,
      aiWhyLength: aiWhyText.length,
      aiBoxLength: aiBoxText.length,
      aiCytoLength: aiCytoText.length,
      userThumbLength: userThumb.length,
      userWhyLength: userWhy.length,
      userBoxLength: userBox.length,
      userCytoLength: userCyto.length,
      editDistanceThumb,
      editDistanceWhy,
      editDistanceBox,
      editDistanceCyto,
      rejected: decisionState === 'reject',
      rejectedROI: decisionState === 'reject',
      thumbnailDecision: 'accept',
      thumbnailRejectAllowed: false,
      thumbnailEditedText: userThumb,
      // paths for traceability
      thumb: item.thumb,
      box: item.box,
      cyto: item.cyto,
      decisions: { overall: decisionState },
    };
    // Overall decision: accept if all accepted; reject if any reject; else edit
    logEntry.decision = logEntry.rejected ? 'reject' : 'accept';

    logs.push(logEntry);
    saveLocal();
    decisionState = null;
    dom.btnNext.disabled = true;
    nextItem();
  }

  function saveLocal() {
    try {
      const key = 'hil_logs_' + sessionId;
      localStorage.setItem(key, JSON.stringify(logs));
    } catch (e) {
      // ignore
    }
  }

  function toCSV(rows) {
    const headers = [
      'sessionId','timestamp','caseId','roiIndex','decision','elapsedMs',
      'acceptClicks','rejectClicks','editClicks',
      'keystrokesThumb','keystrokesWhy','keystrokesBox','keystrokesCyto',
      'aiThumbLength','aiWhyLength','aiBoxLength','aiCytoLength',
      'userThumbLength','userWhyLength','userBoxLength','userCytoLength',
      'editDistanceThumb','editDistanceWhy','editDistanceBox','editDistanceCyto',
      'rejected','rejectedROI','thumbnailDecision','thumbnailRejectAllowed','thumb','box','cyto'
    ];
    const escape = (v) => '"' + String(v ?? '').replace(/"/g, '""') + '"';
    const lines = [headers.join(',')];
    for (const r of rows) {
      const line = [
        r.sessionId,r.timestamp,r.caseId,r.roiIndex,r.decision,r.elapsedMs,
        r.counters?.acceptClicks ?? 0,
        r.counters?.rejectClicks ?? 0,
        r.counters?.editClicks ?? 0,
        r.counters?.keystrokesThumb ?? 0,
        r.counters?.keystrokesWhy ?? 0,
        r.counters?.keystrokesBox ?? 0,
        r.counters?.keystrokesCyto ?? 0,
        r.aiThumbLength,
        r.aiWhyLength,
        r.aiBoxLength,
        r.aiCytoLength,
        r.userThumbLength,
        r.userWhyLength,
        r.userBoxLength,
        r.userCytoLength,
        r.editDistanceThumb,
        r.editDistanceWhy,
        r.editDistanceBox,
        r.editDistanceCyto,
        r.rejected,
        r.rejectedROI,
        r.thumbnailDecision,
        r.thumbnailRejectAllowed,
        r.thumb,
        r.box,
        r.cyto
      ].map(escape).join(',');
      lines.push(line);
    }
    return lines.join('\n');
  }

  function downloadBlob(content, mime, filename) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function init() {
    try {
      manifestItems = await loadManifest();
      if (!manifestItems.length) throw new Error('Manifest is empty');
      setFooter(`Loaded ${manifestItems.length} items. First: ${manifestItems[0]?.thumb || ''}`);
      dom.statusIndex.textContent = `Progress: 0/${manifestItems.length}`;
      // Auto-start to avoid confusion
      currentIndex = -1;
      nextItem();
    } catch (e) {
      setFooter(e.message || 'Failed to load manifest');
    }
  }

  // Event bindings
  dom.btnStart.addEventListener('click', () => {
    if (!manifestItems.length) return;
    currentIndex = -1;
    nextItem();
  });
  // Next always advances; interpret as accept unless rejected
  dom.btnNext.addEventListener('click', () => {
    if (!decisionState) {
      decisionState = 'accept';
    }
    recordAndAdvance();
  });
  // Single reject button
  const btnRejectAll = document.getElementById('btnRejectAll');
  btnRejectAll && btnRejectAll.addEventListener('click', () => {
    counters.rejectClicks += 1;
    decisionState = 'reject';
    dom.statusDecision.textContent = 'Decision: REJECT';
  });
  dom.btnEdit?.addEventListener && dom.btnEdit.addEventListener('click', handleEdit);
  dom.editThumb.addEventListener('input', handleKeystrokeThumb);
  dom.editWhy.addEventListener('input', handleKeystrokeWhy);
  dom.editBox.addEventListener('input', handleKeystrokeBox);
  dom.editCyto.addEventListener('input', handleKeystrokeCyto);

  dom.btnDownloadJSON.addEventListener('click', () => {
    downloadBlob(JSON.stringify(logs, null, 2), 'application/json', `hil_logs_${sessionId}.json`);
  });
  dom.btnDownloadCSV.addEventListener('click', () => {
    downloadBlob(toCSV(logs), 'text/csv', `hil_logs_${sessionId}.csv`);
  });

  // No per-section buttons in simplified UI

  init();
})();


