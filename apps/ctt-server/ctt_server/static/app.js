// SPDX-License-Identifier: BSD-2-Clause
// Copyright (C) 2026, Raspberry Pi
// ctt-server browser logic: Alpine components for capture, run and results.

// --- shared helpers --------------------------------------------------------
function sanitiseLabel(s) {
  return (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

// Sensor h/v flip persists across tabs/reloads (the camera resets it on every
// reconfigure, so the browser is the source of truth and re-enforces it).
function loadFlip() {
  try { return JSON.parse(localStorage.getItem('cttFlip') || '{}'); } catch (e) { return {}; }
}
function saveFlip(hflip, vflip) {
  try { localStorage.setItem('cttFlip', JSON.stringify({ hflip: !!hflip, vflip: !!vflip })); } catch (e) { /* ignore */ }
}

function detectType(name) {
  if (name.includes('alsc')) return 'alsc';
  if (name.includes('cac')) return 'cac';
  return 'macbeth';
}

// --- zoom loupe ------------------------------------------------------------
// Press-and-hold a preview to magnify the area under the cursor in a corner
// PiP, sampling the live <img> pixels via canvas drawImage (no backend round
// trip). Returned as a mixin so both the capture viewfinder and the results
// live-preview test can spread it in over their own element ids.
function createLoupe({ imgId, canvasId, markerId, zoom = 1.5 }) {
  return {
    loupe: { active: false, zoom },
    _loupeIds: { img: imgId, canvas: canvasId, marker: markerId },

    // The displayed content box for an object-fit:contain <img>: the scale +
    // letterbox offsets that map natural-image px ↔ displayed px.
    previewBox() {
      const img = document.getElementById(this._loupeIds.img);
      if (!img) return null;
      const cw = img.clientWidth, ch = img.clientHeight;
      const natW = img.naturalWidth || cw, natH = img.naturalHeight || ch;
      const scale = Math.min(cw / natW, ch / natH);
      return { img, cw, ch, natW, natH, scale, ox: (cw - natW * scale) / 2, oy: (ch - natH * scale) / 2 };
    },

    loupeStart(e) {
      const box = this.previewBox();
      if (!box || box.img.naturalWidth === 0) return;
      e.preventDefault();  // stop the browser's native image-drag from hijacking the gesture
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      this._lx = e.clientX; this._ly = e.clientY;
      this.loupe.active = true;
      this._loupeTick();
    },

    loupeMove(e) {
      if (!this.loupe.active) return;
      this._lx = e.clientX; this._ly = e.clientY;
      this.loupeRender();  // pan immediately on move (don't wait for the next frame)
    },

    loupeEnd(e) {
      if (!this.loupe.active) return;
      this.loupe.active = false;
      if (this._loupeRaf) cancelAnimationFrame(this._loupeRaf);
      this._loupeRaf = null;
      try { if (e && e.pointerId != null) e.currentTarget.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    },

    _loupeTick() {
      if (!this.loupe.active) return;
      this.loupeRender();
      this._loupeRaf = requestAnimationFrame(() => this._loupeTick());
    },

    loupeRender() {
      const cv = document.getElementById(this._loupeIds.canvas);
      const marker = document.getElementById(this._loupeIds.marker);
      const box = this.previewBox();
      if (!box || !cv) return;
      const rect = box.img.getBoundingClientRect();
      // Cursor → natural-image coords (undo the contain mapping), clamped to the image.
      let nx = (this._lx - rect.left - box.ox) / box.scale;
      let ny = (this._ly - rect.top - box.oy) / box.scale;
      nx = Math.max(0, Math.min(box.natW, nx));
      ny = Math.max(0, Math.min(box.natH, ny));
      const W = cv.width, H = cv.height;
      const cropW = W / this.loupe.zoom, cropH = H / this.loupe.zoom;
      let sx = Math.max(0, Math.min(box.natW - cropW, nx - cropW / 2));
      let sy = Math.max(0, Math.min(box.natH - cropH, ny - cropH / 2));
      const ctx = cv.getContext('2d');
      ctx.imageSmoothingEnabled = false;  // show real pixels so focus is judgeable
      ctx.clearRect(0, 0, W, H);
      try { ctx.drawImage(box.img, sx, sy, cropW, cropH, 0, 0, W, H); } catch (_) { /* frame not ready */ }
      ctx.strokeStyle = 'rgba(214,51,108,0.85)'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W / 2, H / 2 - 8); ctx.lineTo(W / 2, H / 2 + 8);
      ctx.moveTo(W / 2 - 8, H / 2); ctx.lineTo(W / 2 + 8, H / 2);
      ctx.stroke();
      if (marker) {
        marker.style.left = (box.ox + sx * box.scale) + 'px';
        marker.style.top = (box.oy + sy * box.scale) + 'px';
        marker.style.width = (cropW * box.scale) + 'px';
        marker.style.height = (cropH * box.scale) + 'px';
      }
      // Make the loupe follow the cursor (offset so it doesn't sit under the pointer),
      // flipping/clamping so it stays inside the preview.
      const wrap = cv.parentElement;
      if (wrap) {
        const wr = wrap.getBoundingClientRect();
        const lw = cv.offsetWidth || cv.width, lh = cv.offsetHeight || cv.height;
        const cxw = this._lx - wr.left, cyw = this._ly - wr.top;
        let lx = cxw + 24, ly = cyw + 24;
        if (lx + lw > wr.width) lx = cxw - lw - 24;
        if (ly + lh > wr.height) ly = cyw - lh - 24;
        lx = Math.max(0, Math.min(wr.width - lw, lx));
        ly = Math.max(0, Math.min(wr.height - lh, ly));
        cv.style.left = lx + 'px';
        cv.style.top = ly + 'px';
        cv.style.right = 'auto';
      }
    },
  };
}

// --- capture page ----------------------------------------------------------
function captureApp(cfg) {
  return {
    ...createLoupe({ imgId: 'previewImg', canvasId: 'loupeCanvas', markerId: 'loupeMarker' }),
    project: cfg.project,
    connected: null,                 // null = unknown until /api/health resolves
    captures: cfg.captures || [],
    counts: { macbeth: 0, alsc: 0, cac: 0 },
    form: { image_type: 'macbeth', colour_temp: 6500, lux: 1000 },
    controls: { exposure: 10000, gain: 1.0, auto_exposure: true, colour_temp: 0, lux: 0, ev: 0 },
    camera: { model: '', resolution: null },
    metered: { exposure: 0, gain: 0, colour_temp: 0, lux: 0, focus_fom: 0 },
    clip: { r: 0, g: 0, b: 0 },
    macbeth: { found: false, confidence: null, corners: null, small: false, saturated: false },
    lightbox: { present: false, channel: null, illuminant: '', intensity: 0, illuminants: {} },
    busy: false,
    error: '',
    hflip: false,
    vflip: false,
    previewTick: 0,  // bumped to reconnect the MJPEG <img> after a camera reconfigure

    async init() {
      this.updateCounts();
      // Restore the user's flip choice (persists across tabs/reloads).
      const f = loadFlip();
      this.hflip = !!f.hflip; this.vflip = !!f.vflip;
      try {
        const r = await fetch('/api/health');
        const h = await r.json();
        this.connected = !!h.camera;
        if (h.model) this.camera.model = h.model;
        if (h.resolution) this.camera.resolution = h.resolution;
        if (h.controls) this.metered = h.controls;
      } catch (e) {
        this.connected = false;
      }
      this.loadLightbox();  // independent of the camera
      if (this.connected) {
        // Returning from a Results-page preview test: restore the default tuning.
        // Idempotent server-side (no-op when already on the default).
        try { await fetch('/api/preview-default', { method: 'POST' }); } catch (e) { /* best effort */ }
        // The reload reset the sensor flip; re-enforce the stored choice.
        if (this.hflip || this.vflip) await this.applyTransform();
        this.loadControls();
        this.pollHistogram();
      }
    },

    async applyTransform() {
      // Reconfigure the camera with the new flip and reconnect the preview stream.
      saveFlip(this.hflip, this.vflip);
      try {
        await fetch('/api/transform', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ hflip: this.hflip, vflip: this.vflip }),
        });
        this.previewTick = Date.now();  // force the MJPEG <img> to reopen on the reconfigured camera
      } catch (e) { this.error = 'Failed to set flip'; }
    },

    async loadLightbox() {
      try {
        const r = await fetch('/api/lightbox');
        if (!r.ok) return;
        const lb = await r.json();
        const ch = lb.channel;  // capture before mutating: lb IS this.lightbox below
        this.lightbox = lb;
        if (lb.present && ch != null) {
          // Re-assert the active channel once the <select>'s x-for <option>s exist:
          // Alpine applies x-model before the options render, so the dropdown would
          // otherwise show the first illuminant. The -1 forces a reactive change so
          // the nextTick reassignment re-syncs the <select>. Then seed the colour temp.
          this.lightbox.channel = -1;
          this.$nextTick(() => { this.lightbox.channel = ch; this.seedColourTemp(); });
        }
      } catch (e) { /* lightbox is optional */ }
    },

    async postLightbox(body) {
      try {
        const r = await fetch('/api/lightbox', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.ok) { this.lightbox = await r.json(); }
        else { const d = await r.json().catch(() => ({})); this.error = d.error || 'Lightbox command failed'; }
      } catch (e) { this.error = 'Lightbox request failed'; }
    },

    setIlluminant() {
      this.seedColourTemp();  // seed the capture colour temp from the illuminant's nominal CCT
      this.postLightbox({ channel: Number(this.lightbox.channel) });
    },

    // Seed the Tag & capture colour temp from the selected illuminant's nominal
    // CCT (from the lightbox API); fall back to 6500 K when the channel has none.
    seedColourTemp() {
      const temps = this.lightbox.illuminant_temps || {};
      this.form.colour_temp = Number(temps[this.lightbox.channel] ?? 6500);
    },
    applyLightbox() { this.postLightbox({ channel: Number(this.lightbox.channel), percent: this.lightbox.intensity }); },
    lightboxOff() { this.postLightbox({ off: true }); },

    updateCounts() {
      const c = { macbeth: 0, alsc: 0, cac: 0 };
      for (const cap of this.captures) c[cap.image_type] = (c[cap.image_type] || 0) + 1;
      this.counts = c;
    },

    nextIndex(type, ct) {
      const prefix = `${type}_${ct}k_`;
      let max = -1;
      for (const c of this.captures) {
        if (c.filename.startsWith(prefix)) {
          const stem = c.filename.slice(prefix.length, -4);
          if (/^\d+$/.test(stem)) max = Math.max(max, parseInt(stem, 10));
        }
      }
      return max + 1;
    },

    previewName() {
      const ct = this.form.colour_temp || 0;
      if (this.form.image_type === 'alsc') return `alsc_${ct}k_${this.nextIndex('alsc', ct)}.dng`;
      if (this.form.image_type === 'cac') return `cac_${ct}k_${this.nextIndex('cac', ct)}.dng`;
      const label = sanitiseLabel(this.project) || 'mac';
      return `${label}_${ct}k_${this.form.lux || 0}l.dng`;
    },

    async loadControls() {
      try {
        const r = await fetch('/api/controls');
        if (r.ok) this.controls = await r.json();
      } catch (e) { /* preview not critical */ }
    },

    async applyControls() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.controls),
        });
        if (r.ok) this.controls = await r.json();
      } catch (e) { this.error = 'Failed to set controls'; }
    },

    async pollHistogram() {
      const tick = async () => {
        try {
          const hr = await fetch('/api/histogram');
          if (hr.ok) {
            const h = await hr.json();
            this.clip = h.clipping || this.clip;
            drawHistogram(document.getElementById('histCanvas'), h);
          }
          const cr = await fetch('/api/controls');  // live metered estimates for the info box
          if (cr.ok) this.metered = await cr.json();
          // Macbeth finder: only while the Macbeth image type is selected.
          if (this.form.image_type === 'macbeth') {
            const mr = await fetch('/api/macbeth');
            if (mr.ok) this.macbeth = await mr.json();
          } else if (this.macbeth.found) {
            this.macbeth = { found: false, confidence: null, corners: null };
          }
          this.drawMacbeth();
        } catch (e) { /* transient */ }
        setTimeout(tick, 1500);
      };
      tick();
    },

    macbethState() {
      // 'bad' = not found / low confidence; 'warn' = too small or clipping; else 'ok'.
      if (!this.macbeth.found || this.macbeth.confidence === null || this.macbeth.confidence < 0.75) return 'bad';
      if (this.macbeth.small || this.macbeth.saturated) return 'warn';
      return 'ok';
    },

    drawMacbeth() {
      const cv = document.getElementById('macbethOverlay');
      const box = this.previewBox();
      if (!box || !cv) return;
      cv.width = box.cw; cv.height = box.ch;
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, box.cw, box.ch);
      if (!this.macbeth.found || !this.macbeth.corners) return;
      // Map normalised frame coords → the image's rendered box.
      const dw = box.natW * box.scale, dh = box.natH * box.scale;
      const pts = this.macbeth.corners.map(([nx, ny]) => [box.ox + nx * dw, box.oy + ny * dh]);
      const st = this.macbethState();
      const col = st === 'ok' ? '#2fb344' : st === 'warn' ? '#f1a204' : '#e5484d';
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.lineWidth = 3;
      ctx.strokeStyle = col;
      ctx.stroke();
      ctx.fillStyle = col;
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }
    },

    async capture() {
      this.error = '';
      // Warn before overwriting: a re-capture at the same colour temp + lux
      // reuses the filename and replaces the existing image on disk.
      const name = this.previewName();
      if (this.captures.some((c) => c.filename === name) &&
          !confirm(`“${name}” already exists. Capturing again will overwrite it. Continue?`)) {
        return;
      }
      this.busy = true;
      try {
        const r = await fetch(`/projects/${this.project}/capture`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_type: this.form.image_type,
            colour_temp: this.form.colour_temp,
            lux: this.form.image_type === 'macbeth' ? this.form.lux : null,
          }),
        });
        const data = await r.json();
        if (!r.ok) { this.error = data.error || 'Capture failed'; return; }
        const entry = {
          filename: data.filename, image_type: data.image_type,
          colour_temp: data.colour_temp, lux: data.lux, label: data.label, valid: true,
          jpeg: data.jpeg || 'saved',
        };
        // Replace the existing entry on overwrite, otherwise add to the top.
        const idx = this.captures.findIndex((c) => c.filename === data.filename);
        if (idx >= 0) this.captures.splice(idx, 1, entry);
        else this.captures.unshift(entry);
        this.updateCounts();
      } catch (e) {
        this.error = 'Capture request failed';
      } finally {
        this.busy = false;
      }
    },

    checklist() {
      const alscCts = new Set(this.captures.filter((c) => c.image_type === 'alsc').map((c) => c.colour_temp));
      const macCts = new Set(this.captures.filter((c) => c.image_type === 'macbeth').map((c) => c.colour_temp));
      const mac = this.counts.macbeth;
      return [
        { key: 'alsc-temps', label: 'ALSC flat-fields at ≥2 colour temperatures',
          detail: `${alscCts.size} temps`, done: alscCts.size >= 2 },
        { key: 'mac-count', label: 'At least 3 Macbeth chart images',
          detail: `${mac} shots`, done: mac >= 3 },
        { key: 'mac-spread', label: 'Macbeth across ≥3 illuminants/temperatures',
          detail: `${macCts.size} temps`, done: macCts.size >= 3 },
      ];
    },

    pct(v) { return ((v || 0) * 100).toFixed(1) + '%'; },
  };
}

// --- images page -----------------------------------------------------------
function imagesApp(cfg) {
  return {
    ...createLoupe({ imgId: 'imgsViewerImg', canvasId: 'imgsViewerLoupe', markerId: 'imgsViewerMarker' }),
    project: cfg.project,
    captures: cfg.captures || [],
    rawpy: !!cfg.rawpy,
    busy: false,
    uploadMsg: '',
    viewer: { open: false, src: '', filename: '', developed: false, loading: false },
    exif: { open: false, filename: '', loading: false, error: '', summary: [], tags: [] },

    // Grid thumbnails: half-size develop for DNG-only captures (full-res is too
    // slow on a Pi); saved JPEGs are served as-is and scaled by the browser.
    thumbSrc(c) {
      return `/projects/${this.project}/captures/${encodeURIComponent(c.filename)}/jpeg?thumb=1`;
    },

    // Default preview source for a thumbnail click: the saved JPEG if there is
    // one, else a rawpy develop of the DNG.
    bestSource(c) { return c.jpeg === 'saved' ? 'jpeg' : 'raw'; },

    async uploadFiles(e) {
      const files = Array.from(e.target.files || []);
      e.target.value = '';  // let the same file be re-selected later
      if (!files.length) return;
      this.uploadMsg = '';
      this.busy = true;
      try {
        const fd = new FormData();
        for (const f of files) fd.append('files', f);
        const r = await fetch(`/projects/${this.project}/upload`, { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok) { this.uploadMsg = d.error || 'Upload failed'; return; }
        for (const entry of (d.added || [])) {
          const idx = this.captures.findIndex((c) => c.filename === entry.filename);
          if (idx >= 0) this.captures.splice(idx, 1, entry);
          else this.captures.unshift(entry);
        }
        const parts = [];
        if (d.added && d.added.length) parts.push(`Added ${d.added.length}.`);
        if (d.skipped && d.skipped.length) {
          parts.push('Skipped: ' + d.skipped.map((s) => `${s.filename} (${s.reason})`).join('; '));
        }
        this.uploadMsg = parts.join(' ');
      } catch (e) {
        this.uploadMsg = 'Upload request failed';
      } finally {
        this.busy = false;
      }
    },

    async remove(filename) {
      if (!confirm(`Delete ${filename}?`)) return;
      const r = await fetch(`/projects/${this.project}/captures/${encodeURIComponent(filename)}/delete`, { method: 'POST' });
      if (r.ok) this.captures = this.captures.filter((c) => c.filename !== filename);
    },

    openViewer(c, source) {
      this.viewer = {
        open: true,
        filename: c.filename,
        developed: source === 'raw',
        loading: source === 'raw',  // a rawpy develop takes seconds; cleared on <img> load
        src: `/projects/${this.project}/captures/${encodeURIComponent(c.filename)}/jpeg?source=${source}`,
      };
    },

    closeViewer() {
      this.viewer.open = false;
      this.viewer.src = '';  // free the (large) decoded image
      this.loupe.active = false;
    },

    async toggleExclude(c) {
      const r = await fetch(`/projects/${this.project}/captures/${encodeURIComponent(c.filename)}/exclude`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ excluded: !c.excluded }),
      });
      if (r.ok) c.excluded = (await r.json()).excluded;
    },

    async openExif(c) {
      this.exif = { open: true, filename: c.filename, loading: true, error: '', summary: [], tags: [] };
      try {
        const r = await fetch(`/projects/${this.project}/captures/${encodeURIComponent(c.filename)}/exif`);
        if (!r.ok) { this.exif.error = 'Failed to read EXIF metadata'; return; }
        const d = await r.json();
        this.exif.summary = d.summary || [];
        this.exif.tags = d.tags || [];
      } catch (e) {
        this.exif.error = 'EXIF request failed';
      } finally {
        this.exif.loading = false;
      }
    },
  };
}

function drawHistogram(canvas, h) {
  if (!canvas || !h) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const bins = (h.r || []).length;
  if (!bins) return;
  const maxAll = Math.max(1, ...(h.r || []), ...(h.g || []), ...(h.b || []));
  const channels = [['r', 'rgba(229,72,77,0.8)'], ['g', 'rgba(47,179,68,0.8)'], ['b', 'rgba(80,160,255,0.8)']];
  for (const [name, colour] of channels) {
    const data = h[name] || [];
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < bins; i++) {
      const x = (i / (bins - 1)) * W;
      const y = H - (data[i] / maxAll) * (H - 4);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.fillStyle = colour.replace('0.8', '0.18');
    ctx.fill();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
}

// --- run page --------------------------------------------------------------
function runApp(cfg) {
  return {
    project: cfg.project,
    targets: 'pisp,vc4',
    mode: 'full',
    greyworld: false,
    doAlscColour: true,
    luminanceStrength: 0.8,
    maxGain: 8.0,
    blacklevel: -1,
    matrixSelection: 'average',
    testPatches: '1,2,5,8,9,12,14',
    luxMode: 'single',          // 'single' = anchor on nearest-luxAnchor capture; 'average' = robust average
    luxAnchor: 1000,            // anchor lux for single mode
    luxMethod: 'trimmed-mean',  // robust-average method: 'trimmed-mean' | 'median'
    running: false,
    done: false,
    exitCode: null,
    source: null,

    start() {
      if (this.running) return;
      this.running = true; this.done = false; this.exitCode = null;
      const console = this.$refs.console;
      console.innerHTML = '';
      const params = new URLSearchParams({
        targets: this.targets, mode: this.mode,
        greyworld: this.greyworld ? '1' : '0',
        do_alsc_colour: this.doAlscColour ? '1' : '0',
        luminance_strength: this.luminanceStrength,
        max_gain: this.maxGain,
        blacklevel: this.blacklevel,
        matrix_selection: this.matrixSelection,
        test_patches: this.matrixSelection === 'patches' ? this.testPatches : '',
        lux_reference_target: this.luxMode === 'average' ? 0 : (this.luxAnchor || 1000),
        lux_reference_method: this.luxMethod,
      });
      this.source = new EventSource(`/projects/${this.project}/run/stream?${params}`);
      this.source.onmessage = (e) => {
        const line = JSON.parse(e.data);
        if (line.startsWith('CTT_EXIT ')) {
          this.exitCode = parseInt(line.slice(9), 10);
          this.running = false; this.done = this.exitCode === 0;
          this.source.close();
          return;
        }
        this.appendLine(console, line);
      };
      this.source.onerror = () => {
        if (this.running) { this.appendLine(console, 'ERROR: stream interrupted'); this.running = false; }
        if (this.source) this.source.close();
      };
    },

    appendLine(consoleEl, line) {
      const el = document.createElement('span');
      el.className = 'ln ' + classify(line);
      el.textContent = line;
      consoleEl.appendChild(el);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    },
  };
}

function classify(line) {
  if (line.startsWith('$ ')) return 'cmd';
  if (line.includes('✓') || line.includes('complete')) return 'ok';
  if (line.includes('✗') || line.startsWith('ERROR') || line.includes('Error')) return 'err';
  if (line.includes('WARNING') || line.includes('DISCARDED')) return 'warn';
  return '';
}

// --- results page ----------------------------------------------------------
function resultsApp(cfg) {
  return {
    ...createLoupe({ imgId: 'resultPreviewImg', canvasId: 'resultLoupeCanvas', markerId: 'resultLoupeMarker' }),
    project: cfg.project,
    targets: cfg.targets,
    target: cfg.targets[0],
    summary: null,
    charts: {},
    metrics: null,
    ccmCt: null,          // selected colour temp for the per-patch ΔE chart
    view: 'new',          // results page: 'new' (this calibration) | 'old' (built-in default)
    all: {},              // results page: target -> full /results/data response
    alscTarget: cfg.targets[0],  // ALSC card's own target toggle (PISP/VC4)
    alscHover: null,             // {col,row,r,g,b} gains under the cursor on the ALSC grid
    runs: cfg.runs || {}, // target -> {label, epoch}: when each tuning file was generated
    autoPreview: cfg.autoPreview || false,  // Preview page: start the live test on load
    _charts: {},
    // live preview test
    busy: false,
    testing: false,
    testTarget: null,
    testTuning: null,
    testKind: null,       // 'generated' | 'standard' — which tuning is live
    previewSrc: '',
    testError: '',
    hflip: false,         // preview page: sensor flips, applied to the live test camera
    vflip: false,
    camera: {},           // preview page: {model, resolution} for the live sensor-info box
    metered: { exposure: 0, gain: 0, colour_temp: 0, lux: 0 },  // live metered values
    controls: { auto_exposure: true, exposure: 0, gain: 1, ev: 0 },  // exposure panel state
    lightbox: { present: false, channel: null, intensity: 0, illuminants: {} },  // optional lightbox device
    _polling: false,      // guards a single metered-poll loop

    init() {
      const f = loadFlip();                            // restore flip choice (persists across tabs)
      this.hflip = !!f.hflip; this.vflip = !!f.vflip;
      if (this.autoPreview) { this.startPreviewTest(); this.loadLightbox(); }  // Preview page
      if (cfg.allTargets) this.loadAll();              // Results page: new/old + per-target ALSC
      else this.select(this.target);                   // Tuning/Preview: single target
    },

    // Results page: fetch every target's data up-front (needed for the per-target
    // ALSC toggle), then render the primary target under the current new/old view.
    async loadAll() {
      for (const t of this.targets) {
        try {
          const r = await fetch(`/projects/${this.project}/results/data?target=${t}`);
          if (r.ok) this.all[t] = await r.json();
        } catch (e) { /* skip missing target */ }
      }
      this.target = this.targets.find((t) => this.all[t]) || this.targets[0];
      this.alscTarget = this.target;
      this.applyView();
      this.$nextTick(() => requestAnimationFrame(() => this.renderCharts()));
    },

    // Point summary/charts/metrics at the selected target's new or old (default) data.
    applyView() {
      const d = this.all[this.target];
      if (!d) { this.summary = null; this.charts = {}; this.metrics = null; return; }
      if (this.view === 'old' && d.metrics && d.metrics.default) {
        this.summary = d.metrics.default.summary;
        this.charts = d.metrics.default.charts || {};
        // Reuse run-level fields (counts/coverage/warnings); swap CCM to the default's.
        this.metrics = { ...d.metrics, ccm: d.metrics.ccm_default || [], ccm_quality: d.metrics.ccm_default_quality || {} };
      } else {
        this.summary = d.summary; this.charts = d.charts || {}; this.metrics = d.metrics || null;
      }
      const cts = ((this.metrics && this.metrics.ccm) || []).map((c) => c.ct);
      if (!cts.includes(this.ccmCt)) this.ccmCt = cts.length ? cts[0] : null;
    },

    hasDefault() { const d = this.all[this.target]; return !!(d && d.metrics && d.metrics.default); },
    setView(v) {
      this.view = v;
      this.applyView();
      this.$nextTick(() => requestAnimationFrame(() => this.renderCharts()));
    },

    // ALSC card data for its own target toggle, honouring the new/old view.
    alscView() {
      const d = this.all[this.alscTarget];
      if (!d) return null;
      const charts = (this.view === 'old' && d.metrics && d.metrics.default) ? d.metrics.default.charts : d.charts;
      return charts ? charts.alsc : null;
    },
    setAlscTarget(t) {
      this.alscTarget = t;
      this.$nextTick(() => requestAnimationFrame(() => { try { this.renderAlsc(); } catch (e) { console.error(e); } }));
    },

    async startPreviewTest() {
      this.busy = true; this.testError = '';
      try {
        const r = await fetch(`/projects/${this.project}/preview-test`, { method: 'POST' });
        const d = await r.json();
        if (!r.ok) { this.testError = d.error || 'Failed to start preview'; return; }
        this.testTarget = d.target; this.testTuning = d.tuning; this.testKind = 'generated';
        this.testing = true;
        // Cache-bust so the <img> opens a fresh MJPEG stream on the new camera.
        this.previewSrc = '/api/preview?t=' + Date.now();
        if (this.hflip || this.vflip) await this._postTransform();  // re-apply flip on the fresh camera
        this._loadCamInfo(); this.pollMetered();
      } catch (e) {
        this.testError = 'Preview request failed';
      } finally {
        this.busy = false;
      }
    },

    genLabel() { return (this.runs[this.testTarget] || {}).label || ''; },

    // POST the current flip state and reopen the stream on the reconfigured camera.
    async _postTransform() {
      saveFlip(this.hflip, this.vflip);
      await fetch('/api/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hflip: this.hflip, vflip: this.vflip }),
      });
      this.previewSrc = '/api/preview?t=' + Date.now();
    },

    async applyTransform() {
      this.busy = true; this.testError = '';
      try { await this._postTransform(); }
      catch (e) { this.testError = 'Flip request failed'; }
      finally { this.busy = false; }
    },

    // Fetch sensor model/resolution (and seed metered values) for the info box.
    async _loadCamInfo() {
      try {
        const r = await fetch('/api/health');
        if (!r.ok) return;
        const h = await r.json();
        if (h.model) this.camera.model = h.model;
        if (h.resolution) this.camera.resolution = h.resolution;
        if (h.controls) { this.metered = { ...h.controls }; this.controls = { ...h.controls }; }
      } catch (e) { /* best effort */ }
    },

    async applyControls() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.controls),
        });
        if (r.ok) this.controls = await r.json();
      } catch (e) { this.testError = 'Failed to set controls'; }
    },

    // Poll live metered exposure/gain/CT/lux while the preview is running.
    pollMetered() {
      if (this._polling) return;
      this._polling = true;
      const tick = async () => {
        if (!this.testing) { this._polling = false; return; }
        try {
          const cr = await fetch('/api/controls');
          if (cr.ok) this.metered = await cr.json();
        } catch (e) { /* transient */ }
        if (this.testing) setTimeout(tick, 1500); else this._polling = false;
      };
      tick();
    },

    // --- optional lightbox device (same API as the Capture page) ----------
    async loadLightbox() {
      try {
        const r = await fetch('/api/lightbox');
        if (!r.ok) return;
        const lb = await r.json();
        const ch = lb.channel;  // capture before mutating: lb IS this.lightbox below
        this.lightbox = lb;
        if (lb.present && ch != null) {
          // Re-assert the active channel once the <select>'s x-for options exist
          // (Alpine applies x-model before they render, else it shows the first).
          this.lightbox.channel = -1;
          this.$nextTick(() => { this.lightbox.channel = ch; });
        }
      } catch (e) { /* lightbox is optional */ }
    },
    async postLightbox(body) {
      try {
        const r = await fetch('/api/lightbox', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.ok) this.lightbox = await r.json();
        else { const d = await r.json().catch(() => ({})); this.testError = d.error || 'Lightbox command failed'; }
      } catch (e) { this.testError = 'Lightbox request failed'; }
    },
    setIlluminant() { this.postLightbox({ channel: Number(this.lightbox.channel) }); },
    applyLightbox() { this.postLightbox({ channel: Number(this.lightbox.channel), percent: this.lightbox.intensity }); },
    lightboxOff() { this.postLightbox({ off: true }); },

    async previewStandard() {
      // Switch the live preview to the camera's default (built-in) tuning, for
      // an A/B against the generated one — stays in preview mode.
      this.busy = true; this.testError = '';
      this.previewSrc = '';  // close the stream before the camera restarts
      try {
        const r = await fetch('/api/preview-default', { method: 'POST' });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { this.testError = d.error || 'Failed to load standard tuning'; return; }
        this.testTuning = 'default (built-in)'; this.testTarget = null; this.testKind = 'standard';
        this.testing = true;
        this.previewSrc = '/api/preview?t=' + Date.now();
        if (this.hflip || this.vflip) await this._postTransform();  // re-apply flip on the fresh camera
        this._loadCamInfo(); this.pollMetered();
      } catch (e) {
        this.testError = 'Preview request failed';
      } finally {
        this.busy = false;
      }
    },

    async capturePng() {
      this.busy = true; this.testError = '';
      try {
        const r = await fetch(`/projects/${this.project}/preview-capture`);
        if (!r.ok) { this.testError = 'Capture failed'; return; }
        const blob = await r.blob();
        const cd = r.headers.get('Content-Disposition') || '';
        const m = cd.match(/filename="?([^"]+)"?/);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = m ? m[1] : 'preview.png';
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(a.href);
      } catch (e) {
        this.testError = 'Capture request failed';
      } finally {
        this.busy = false;
      }
    },

    async restoreDefault() {
      this.busy = true;
      this.previewSrc = '';  // close the MJPEG stream before the camera restarts
      try {
        await fetch('/api/preview-default', { method: 'POST' });
      } catch (e) { /* best effort */ }
      this.testing = false; this.testTarget = null; this.testTuning = null;
      this.busy = false;
    },

    async select(t) {
      this.target = t;
      const r = await fetch(`/projects/${this.project}/results/data?target=${t}`);
      if (!r.ok) return;
      const data = await r.json();
      this.summary = data.summary;
      this.charts = data.charts || {};
      this.metrics = data.metrics || null;
      // Keep the selected colour temperature across a target switch when the new
      // target also has it; otherwise fall back to its first CT.
      const cts = ((this.metrics && this.metrics.ccm) || []).map((c) => c.ct);
      if (!cts.includes(this.ccmCt)) this.ccmCt = cts.length ? cts[0] : null;
      // Wait for Alpine to apply x-show/x-if, then a layout frame, so each
      // canvas's container has its final size before Chart.js measures it.
      this.$nextTick(() => requestAnimationFrame(() => this.renderCharts()));
    },

    // The selected CT's per-patch entry, and its mean/worst ΔE for the stat row.
    ccmEntry() {
      const ccm = (this.metrics && this.metrics.ccm) || [];
      return ccm.find((c) => c.ct === this.ccmCt) || ccm[0] || null;
    },
    // Per-patch ΔE was added later; older sidecars only have per-CT aggregates.
    ccmHasPatches() { return ((this.metrics && this.metrics.ccm) || []).some((c) => c.patches && c.patches.length); },
    ccmStats() {
      const e = this.ccmEntry();
      if (!e) return { mean: null, worst: null, colour: null };
      const de = (e.patches || []).map((p) => p.de);
      const dn = (e.patches || []).map((p) => p.de_norm).filter((v) => v != null);
      const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      if (de.length) {
        return { mean: avg(de), worst: Math.max(...de), colour: dn.length ? avg(dn) : null };
      }
      return { mean: e.metric_after, worst: e.max_after, colour: null };  // legacy fallback
    },
    setCcmCt(ct) {
      this.ccmCt = ct;
      this._charts.ccm?.destroy();
      this.$nextTick(() => requestAnimationFrame(() => { try { this.renderCcm(); } catch (e) { console.error(e); } }));
    },

    // Format a number for display (2 sig-figs-ish), passing through non-numbers.
    fmt(v) { return (typeof v === 'number' && isFinite(v)) ? (Math.round(v * 100) / 100) : (v ?? '–'); },
    bandLabel(b) { return { good: 'Good', fair: 'Fair', poor: 'Poor' }[b] || ''; },
    ctRange() {
      const c = this.metrics && this.metrics.coverage;
      if (!c || c.ct_min == null || c.ct_max == null) return '–';
      return c.ct_min === c.ct_max ? `${c.ct_min}` : `${c.ct_min}–${c.ct_max}`;
    },
    runLabel() { return (this.runs[this.target] || {}).label || ''; },
    runsMismatch() {
      // Warn when the per-target tuning files are from runs > 2 min apart (one is stale).
      const ep = Object.values(this.runs).map((r) => r.epoch).filter((e) => e != null);
      return ep.length > 1 && (Math.max(...ep) - Math.min(...ep)) > 120;
    },

    renderCharts() {
      for (const k in this._charts) { this._charts[k].destroy?.(); }
      this._charts = {};
      // Each chart is independent: a failure in one must not block the others.
      try { if (this.charts.awb) this.renderAwb(); } catch (e) { console.error('AWB chart:', e); }
      try { if (this.alscView()) this.renderAlsc(); } catch (e) { console.error('ALSC chart:', e); }
      try {
        if (this.metrics && this.metrics.ccm && this.metrics.ccm.length) this.renderCcm();
      } catch (e) { console.error('CCM chart:', e); }
      try {
        if (this.metrics && this.metrics.lux && (this.metrics.lux.samples || []).length) this.renderLux();
      } catch (e) { console.error('Lux chart:', e); }
    },

    renderLux() {
      const lux = this.metrics.lux;
      const ctx = document.getElementById('luxChart');
      if (!ctx) return;
      const pts = lux.samples.slice().sort((a, b) => a.ct - b.ct);
      const xs = pts.map((p) => p.ct);
      const ref = lux.reference_slope;
      const opts = chartOpts('Colour temperature (K)', 'Y per lux·exp·gain');
      opts.scales.x.type = 'linear';
      opts.plugins.tooltip = { callbacks: {
        label: (i) => i.dataset.label === 'captures'
          ? `${pts[i.dataIndex].ct}K · ${pts[i.dataIndex].lux} lx · ${i.parsed.y.toFixed(5)}`
          : `calibration ${i.parsed.y.toFixed(5)}`,
      } };
      this._charts.lux = new Chart(ctx, {
        data: {
          datasets: [
            { type: 'scatter', label: 'captures',
              data: pts.map((p) => ({ x: p.ct, y: p.slope })),
              backgroundColor: '#3b82f6', pointRadius: 4 },
            { type: 'line', label: 'calibration',
              data: [{ x: Math.min(...xs), y: ref }, { x: Math.max(...xs), y: ref }],
              borderColor: '#e5484d', borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
          ],
        },
        options: opts,
      });
    },

    renderCcm() {
      const ctx = document.getElementById('ccmChart');
      if (!ctx) return;
      if (!this.ccmHasPatches()) { this.renderCcmByCt(ctx); return; }  // legacy sidecar
      const entry = this.ccmEntry();
      if (!entry || !entry.patches) return;
      const patches = entry.patches;
      const opts = chartOpts('Macbeth patch', 'ΔE (CIE2000)');
      opts.plugins.legend.display = false;
      opts.plugins.tooltip = { callbacks: { title: (i) => 'Patch ' + (i[0].dataIndex + 1) } };
      this._charts.ccm = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: patches.map((_, i) => i + 1),
          // Each bar is filled with the patch's reference colour; height is its ΔE.
          datasets: [{
            label: `ΔE · ${entry.ct}K`,
            data: patches.map((p) => p.de),
            backgroundColor: patches.map((p) => `rgb(${p.rgb[0]},${p.rgb[1]},${p.rgb[2]})`),
            borderColor: 'rgba(255,255,255,0.25)', borderWidth: 1,
          }],
        },
        options: opts,
      });
    },

    // Fallback for sidecars without per-patch data: worst ΔE per CT, before/after.
    renderCcmByCt(ctx) {
      const pts = this.metrics.ccm;
      this._charts.ccm = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: pts.map((p) => p.ct + 'K'),
          datasets: [
            { label: 'Worst ΔE before', data: pts.map((p) => p.max_before), backgroundColor: '#6b7888' },
            { label: 'Worst ΔE after', data: pts.map((p) => p.max_after), backgroundColor: '#2fb344' },
          ],
        },
        options: chartOpts('Colour temperature', 'ΔE (CIE2000)'),
      });
    },

    renderAwb() {
      // Plot the white-balance locus: R/G on x, B/G on y, joined in CT order.
      const pts = (this.charts.awb.points || []).slice().sort((a, b) => a.ct - b.ct);
      const ctx = document.getElementById('awbChart');
      if (!ctx) return;
      const opts = chartOpts('R/G', 'B/G');
      opts.scales.x.type = 'linear';  // x is a continuous R/G value, not a category
      opts.plugins.legend.display = false;
      opts.plugins.tooltip = { callbacks: { label: (i) => `${pts[i.dataIndex].ct}K  (R/G ${i.parsed.x.toFixed(3)}, B/G ${i.parsed.y.toFixed(3)})` } };
      this._charts.awb = new Chart(ctx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'CT curve',
            data: pts.map((p) => ({ x: p.r, y: p.b })),
            borderColor: '#d6336c', backgroundColor: '#d6336c',
            tension: 0.3, showLine: true, pointRadius: 4,
          }],
        },
        options: opts,
      });
    },

    renderAlsc() {
      const a = this.alscView();
      const canvas = document.getElementById('alscChart');
      if (!canvas || !a || !a.grid) return;
      // Custom heatmap: gain values mapped blue(low)→red(high).
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const rows = a.grid.length, cols = a.grid[0].length;
      const cw = W / cols, ch = H / rows;
      const lo = a.min, hi = a.max, span = (hi - lo) || 1;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const t = (a.grid[r][c] - lo) / span;
          ctx.fillStyle = heat(t);
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      // Hover read-out: map the cursor to a grid cell and expose its R/G/B gains.
      const self = this;
      canvas.onmousemove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left, py = e.clientY - rect.top;
        const c = Math.min(cols - 1, Math.max(0, Math.floor(px / rect.width * cols)));
        const r = Math.min(rows - 1, Math.max(0, Math.floor(py / rect.height * rows)));
        self.alscHover = {
          x: px, y: py,
          r: a.r ? a.r[r][c] : null,
          g: a.g ? a.g[r][c] : a.grid[r][c],
          b: a.b ? a.b[r][c] : null,
        };
      };
      canvas.onmouseleave = () => { self.alscHover = null; };
    },
  };
}

function heat(t) {
  // 0 → blue, 0.5 → green, 1 → red
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, t * 2));
  const b = Math.round(255 * Math.min(1, (1 - t) * 2));
  const g = Math.round(255 * (1 - Math.abs(t - 0.5) * 2) * 0.8);
  return `rgb(${r},${g},${b})`;
}

function chartOpts(xTitle, yTitle) {
  return {
    responsive: true,
    // The canvas lives in a fixed-height .chart-box; fill it rather than deriving
    // height from width (which blows the canvas up and can render blank when the
    // container is only just becoming visible).
    maintainAspectRatio: false,
    // Draw synchronously on creation; a deferred animation frame can otherwise
    // fire after Alpine has re-rendered the canvas away (getContext on null).
    animation: false,
    plugins: { legend: { labels: { color: '#9aa7b8' } } },
    scales: {
      x: { title: { display: true, text: xTitle, color: '#6b7888' }, ticks: { color: '#6b7888' }, grid: { color: '#2b3444' } },
      y: { title: { display: !!yTitle, text: yTitle || '', color: '#6b7888' }, ticks: { color: '#6b7888' }, grid: { color: '#2b3444' } },
    },
  };
}
