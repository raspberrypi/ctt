// SPDX-License-Identifier: BSD-2-Clause
// Copyright (C) 2026, Raspberry Pi
// ctt-server browser logic: Alpine components for capture, run and results.

// Chart.js has its own default font stack; match the app's Roboto.
if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = "Roboto, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
}

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

// Minimal JSON syntax highlighter for the tuning-file viewer: escape the text,
// then wrap strings (keys vs values), numbers and keywords in coloured spans.
// Oversized files are returned escaped-but-plain to keep the DOM manageable.
function highlightJson(text) {
  const esc = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  if (esc.length > 1_500_000) return esc;
  return esc.replace(
    /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g,
    (m, str, colon, kw) => {
      if (str) return colon ? `<span class="j-key">${str}</span>${colon}` : `<span class="j-str">${str}</span>`;
      if (kw) return `<span class="j-kw">${kw}</span>`;
      return `<span class="j-num">${m}</span>`;
    },
  );
}

// Colour a unified diff by line prefix (file headers, @@ hunks, +/- lines).
function highlightDiff(text) {
  const esc = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc.split('\n').map((ln) => {
    let cls = '';
    if (ln.startsWith('+++') || ln.startsWith('---')) cls = 'd-file';
    else if (ln.startsWith('@@')) cls = 'd-hunk';
    else if (ln.startsWith('+')) cls = 'd-add';
    else if (ln.startsWith('-')) cls = 'd-del';
    return cls ? `<span class="${cls}">${ln}</span>` : ln;
  }).join('\n');
}

function detectType(name) {
  if (name.includes('alsc')) return 'alsc';
  if (name.includes('cac')) return 'cac';
  if (name.includes('dark')) return 'dark';
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
    counts: { macbeth: 0, alsc: 0, cac: 0, dark: 0 },
    form: { image_type: 'macbeth', colour_temp: 6500, lux: 1000, frames: 1 },
    controls: { exposure: 10000, gain: 1.0, auto_exposure: true, colour_temp: 0, lux: 0, ev: 0 },
    fpsTarget: 30,  // framerate target; 0 = unconstrained (variable frame duration)
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
        let ch = lb.channel;  // capture before mutating: lb IS this.lightbox below
        // A fresh device reports channel 0 (nothing selected yet); snap to the
        // first illuminant so the model matches what the dropdown renders.
        if (lb.present && lb.illuminants && lb.illuminants[ch] == null) ch = Number(Object.keys(lb.illuminants)[0]);
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
      const c = { macbeth: 0, alsc: 0, cac: 0, dark: 0 };
      for (const cap of this.captures) c[cap.image_type] = (c[cap.image_type] || 0) + 1;
      this.counts = c;
    },

    nextIndex(type, ct) {
      const prefix = type === 'dark' ? 'dark_' : `${type}_${ct}k_`;
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
      if (this.form.image_type === 'dark') return `dark_${this.nextIndex('dark')}.dng`;
      const label = sanitiseLabel(this.project) || 'mac';
      const suffix = (this.form.frames || 1) > 1 ? '_<n>' : '';
      return `${label}_${ct}k_${this.form.lux || 0}l${suffix}.dng`;
    },

    async loadControls() {
      try {
        const r = await fetch('/api/controls');
        if (r.ok) { this.controls = await r.json(); this.fpsTarget = this.controls.fps || 0; }
      } catch (e) { /* preview not critical */ }
    },

    fpsMax() {
      // Fastest rate the current sensor mode allows (the frame duration clamps
      // to the mode's minimum, so a higher target silently runs slower).
      return this.metered.frame_duration_min ? 1000000 / this.metered.frame_duration_min : 0;
    },
    fpsUnachievable() {
      return !!(this.fpsTarget && this.fpsMax() && this.fpsTarget > this.fpsMax() + 0.05);
    },

    async applyFps() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fps: this.fpsTarget || 0 }),
        });
        if (r.ok) {
          const c = await r.json();
          this.fpsTarget = c.fps || 0;
          this.controls.fps = c.fps;  // keep applyControls' payload in step with the new target
        }
      } catch (e) { this.error = 'Failed to set framerate'; }
    },

    async applyControls() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.controls),
        });
        // In manual mode the form is the source of truth: the response carries
        // frame metadata that lags the request by a few frames, so writing it
        // back would bounce the sliders (the metered box shows the actuals).
        if (r.ok && this.controls.auto_exposure) this.controls = await r.json();
      } catch (e) { this.error = 'Failed to set controls'; }
    },

    // Slider bound for manual exposure: the camera's advertised limit,
    // additionally capped at one frame time when a fixed framerate is set.
    expMax() {
      if (!this.controls.exposure_max) return null;  // limits not fetched yet
      const fpsCap = this.fpsTarget ? Math.floor(1000000 / this.fpsTarget) : this.controls.frame_duration_max;
      return Math.min(this.controls.exposure_max, fpsCap);
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
          // In auto mode the greyed-out sliders track the values AEC is
          // actually using, so switching to manual continues from there.
          if (this.controls.auto_exposure) {
            this.controls.exposure = this.metered.exposure;
            this.controls.gain = this.metered.gain;
          }
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
      const frames = this.form.frames || 1;
      // Warn before overwriting: a single re-capture at the same colour temp +
      // lux reuses the filename. Burst frames get fresh indexed names instead.
      const name = this.previewName();
      if (frames === 1 && this.captures.some((c) => c.filename === name) &&
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
            frames,
          }),
        });
        const data = await r.json();
        if (!r.ok) { this.error = data.error || 'Capture failed'; return; }
        for (const entry of (data.added || [])) {
          entry.jpeg = entry.jpeg || 'saved';
          // Replace the existing entry on overwrite, otherwise add to the top.
          const idx = this.captures.findIndex((c) => c.filename === entry.filename);
          if (idx >= 0) this.captures.splice(idx, 1, entry);
          else this.captures.unshift(entry);
        }
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
    expandedGroups: {},  // burst group key -> true while expanded

    // Grid thumbnails: half-size develop for DNG-only captures (full-res is too
    // slow on a Pi); saved JPEGs are served as-is and scaled by the browser.
    thumbSrc(c) {
      return `/projects/${this.project}/captures/${encodeURIComponent(c.filename)}/jpeg?thumb=1`;
    },

    // Default preview source for a thumbnail click: the saved JPEG if there is
    // one, else a rawpy develop of the DNG.
    bestSource(c) { return c.jpeg === 'saved' ? 'jpeg' : 'raw'; },

    // Burst frames share a name apart from the trailing index: Macbeth
    // d65_5000k_800l_<n>.dng, and ALSC replicates alsc_<K>k_<n>.dng (which
    // CTT averages per colour temperature). Group them in the grid; dark
    // frames (dark_<n>.dng) form a single flat group. Uploads are renamed to
    // this scheme on import, so the canonical prefixes are exhaustive here.
    groupKey(c) {
      const mac = c.filename.match(/^(.*\dl)_\d+\.dng$/i);
      if (mac) return mac[1];
      const alsc = c.filename.match(/^(alsc_\d+k)_\d+\.dng$/i);
      if (alsc) return alsc[1];
      const dark = c.filename.match(/^(dark)_\d+\.dng$/i);
      if (dark) return dark[1];
      return null;
    },

    // The grid's render list: singles as cards; bursts as one collapsed card,
    // or (expanded) a full-width header bar followed by the member cards.
    renderItems() {
      const groups = new Map();
      const order = [];
      for (const c of this.captures) {
        const key = this.groupKey(c);
        if (!key) { order.push({ kind: 'card', key: c.filename, c }); continue; }
        if (!groups.has(key)) {
          const g = { kind: 'group', key, items: [] };
          groups.set(key, g);
          order.push(g);
        }
        groups.get(key).items.push(c);
      }
      const out = [];
      for (const it of order) {
        if (it.kind === 'card') { out.push(it); continue; }
        if (it.items.length === 1) {  // a lone _<n> file: just a normal card
          out.push({ kind: 'card', key: it.items[0].filename, c: it.items[0] });
        } else if (this.expandedGroups[it.key]) {
          out.push({ kind: 'bar', key: it.key, count: it.items.length });
          for (const c of it.items) out.push({ kind: 'card', key: c.filename, c, grouped: true });
        } else {
          out.push(it);
        }
      }
      return out;
    },

    toggleGroup(key) { this.expandedGroups[key] = !this.expandedGroups[key]; },

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
    update: false,              // update an existing tuning in place (preserve non-recalibrated sections)
    updateSource: 'project',    // 'project' = the project's own tuning; 'system' = an installed libcamera tuning; 'upload' = a supplied file
    updateFile: null,           // selected file when updateSource === 'upload'
    systemTunings: [],          // installed libcamera tunings (both ISP platforms): [{name, path, target}]
    systemTuning: '',           // selected system tuning path when updateSource === 'system'
    stOpen: false,              // system-tuning dropdown open state
    systemError: '',
    blSeed: null,               // black level measured from the project's dark frames
    blFrames: 0,
    running: false,
    done: false,
    exitCode: null,
    source: null,
    started: false,        // a run has been started this page-load (hides the plan preview)

    // A system/uploaded tuning is platform-specific, so the run target comes from
    // the file, not this control. Lock the Target buttons while one is selected.
    get targetLocked() {
      return this.update && (this.updateSource === 'system' || this.updateSource === 'upload');
    },

    // The selected system-tuning file object (for the dropdown's current label + pill).
    get selectedSystemFile() {
      return this.systemTunings.find((f) => f.path === this.systemTuning) || null;
    },

    // Human label for an ISP target, matching the Target-platform buttons.
    platformLabel(t) {
      return t === 'pisp' ? 'PiSP' : t === 'vc4' ? 'VC4' : (t || '').toUpperCase();
    },

    // The invocation the run will echo, shown in the empty console as a preview.
    get planLine() {
      return `$ ctt  targets=${this.targets}  mode=${this.mode}  update=${this.update ? 'on' : 'off'}`;
    },

    async init() {
      // Seed the black level field from the project's dark frames (if any).
      // Only an untouched field (-1 auto) is seeded; a user value is kept.
      try {
        const r = await fetch(`/projects/${this.project}/blacklevel`);
        if (!r.ok) return;
        const d = await r.json();
        if (d.black_level == null) return;
        this.blSeed = d.black_level;
        this.blFrames = (d.frames || []).length;
        if (this.blacklevel === -1) this.blacklevel = d.black_level;
      } catch (e) { /* seeding is best-effort */ }
    },

    // List the libcamera tuning files installed on this Pi for its ISP platform,
    // defaulting the selection to the one matching the live sensor.
    async loadSystemTunings() {
      if (this.systemTunings.length) return;  // fetched once per page load
      this.systemError = '';
      try {
        const r = await fetch(`/projects/${this.project}/run/system-tunings`);
        const d = await r.json();
        if (!r.ok) { this.systemError = d.error || 'Failed to list system tunings'; return; }
        this.systemTunings = d.files || [];
        this.systemTuning = d.default || (this.systemTunings[0] || {}).path || '';
        if (!this.systemTunings.length) this.systemError = 'No installed tuning files found.';
        this.syncSystemTarget();
      } catch (e) { this.systemError = 'Failed to list system tunings'; }
    },

    // Point the run target at the selected system tuning's platform (it is
    // platform-specific, so the update run must produce that target).
    syncSystemTarget() {
      const f = this.systemTunings.find((x) => x.path === this.systemTuning);
      if (f && f.target) this.targets = f.target;
    },

    async start() {
      if (this.running) return;
      this.running = true; this.done = false; this.exitCode = null;
      this.started = true;
      const console = this.$refs.console;
      console.innerHTML = '';
      // An uploaded or installed (system) tuning becomes the project's tuning for
      // its (detected) target, then the run updates it in place — restricted to
      // that target.
      let targets = this.targets;
      if (this.update && (this.updateSource === 'upload' || this.updateSource === 'system')) {
        const body = new FormData();
        if (this.updateSource === 'upload') {
          if (!this.updateFile) {
            this.appendLine(console, 'ERROR: choose a tuning file to upload');
            this.running = false;
            return;
          }
          body.append('file', this.updateFile);
        } else {
          if (!this.systemTuning) {
            this.appendLine(console, 'ERROR: choose a system tuning file');
            this.running = false;
            return;
          }
          body.append('system_path', this.systemTuning);
        }
        try {
          const r = await fetch(`/projects/${this.project}/run/import-tuning`, { method: 'POST', body });
          const d = await r.json();
          if (!r.ok) throw new Error(d.error || 'import failed');
          targets = d.target;
          const src = this.updateSource === 'system' ? 'system' : 'uploaded';
          this.appendLine(console, `Imported ${src} tuning as ${targets.toUpperCase()} project tuning`);
        } catch (e) {
          this.appendLine(console, `ERROR: ${e.message}`);
          this.running = false;
          return;
        }
      }
      const params = new URLSearchParams({
        targets: targets, mode: this.mode,
        update: this.update ? '1' : '0',
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
      // The log div nests inside the scrolling .console box; scroll the box.
      const box = consoleEl.closest('.console') || consoleEl;
      box.scrollTop = box.scrollHeight;
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
    gammaTarget: 'sRGB',  // gamma card: target transfer function (recomputed client-side)
    view: 'new',          // results page: 'new' (this calibration) | 'old' (built-in default)
    all: {},              // results page: target -> full /results/data response
    alscTarget: cfg.targets[0],  // ALSC card's own target toggle (PISP/VC4)
    alscHover: null,             // {col,row,r,g,b} gains under the cursor on the ALSC grid
    runs: cfg.runs || {}, // target -> {label, epoch}: when each tuning file was generated
    autoPreview: cfg.autoPreview || false,  // Preview page: start the live test on load
    variants: cfg.variants || {},  // Preview page: target -> [{id, label, stale}]
    hasTuning: cfg.hasTuning || false,  // Preview page: any generated tuning exists yet
    // Tuning page: original text + the selected variant's text/diff + variant list.
    tuningState: { original: null, custom: null, diff: null, variants: [], selected: null, existing: null },
    tuningSel: 'original', // dropdown value: 'original' | 'existing' | '<variant slug>'
    variantId: null,       // the selected variant's slug (null for Original/Existing)
    tuningHtml: '',       // highlighted text of the selected view
    diffHtml: '',
    editing: false,
    editorText: '',
    tuningError: '',
    _tuningReq: 0,        // monotonic token guarding against stale tuning-data fetches
    _charts: {},
    // live preview test
    busy: false,
    testing: false,
    testTarget: null,
    testTuning: null,
    testKind: null,       // 'generated' | 'custom' | 'standard' — which tuning is live
    testVariant: null,    // selected custom variant id (when testKind === 'custom')
    testLabel: null,      // selected custom variant label, for captions + the ΔE legend
    previewSrc: '',
    testError: '',
    hflip: false,         // preview page: sensor flips, applied to the live test camera
    vflip: false,
    camera: {},           // preview page: {model, resolution} for the live sensor-info box
    metered: { exposure: 0, gain: 0, colour_temp: 0, lux: 0 },  // live metered values
    controls: { auto_exposure: true, exposure: 0, gain: 1, ev: 0 },  // exposure panel state
    fpsTarget: 30,  // framerate target; 0 = unconstrained (variable frame duration)
    lightbox: { present: false, channel: null, intensity: 0, illuminants: {} },  // optional lightbox device
    colour: null,         // current live ΔE reading (for whichever tuning is loaded)
    liveColour: true,     // semi-live colour measurement while previewing
    chartSeen: null,      // null = no reading yet, false = chart not found, {confidence} = found
    _colourPolling: false,  // guards a single colour-poll loop
    _polling: false,      // guards a single metered-poll loop

    init() {
      const f = loadFlip();                            // restore flip choice (persists across tabs)
      this.hflip = !!f.hflip; this.vflip = !!f.vflip;
      // Preview page: auto-start the generated tuning, or the existing (built-in)
      // one when the project has no generated tuning yet.
      if (this.autoPreview) { (this.hasTuning ? this.startPreviewTest() : this.previewStandard()); this.loadLightbox(); }
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
      // Open the gamma card on the target the run recorded; the user can change it.
      this.gammaTarget = (this.metrics && this.metrics.gamma && this.metrics.gamma.target) || 'sRGB';
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

    async startPreviewTest(kind = 'generated', variant = null) {
      this.busy = true; this.testError = '';
      // Switching tuning while already previewing carries the exposure panel over.
      const keep = this._exposureState();
      try {
        const r = await fetch(`/projects/${this.project}/preview-test`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind, variant }),
        });
        const d = await r.json();
        if (!r.ok) { this.testError = d.error || 'Failed to start preview'; return; }
        this.testTarget = d.target; this.testTuning = d.tuning; this.testKind = d.kind || 'generated';
        this.testVariant = d.variant; this.testLabel = d.label;
        this.testing = true;
        // Cache-bust so the <img> opens a fresh MJPEG stream on the new camera.
        this.previewSrc = '/api/preview?t=' + Date.now();
        if (this.hflip || this.vflip) await this._postTransform();  // re-apply flip on the fresh camera
        this._loadCamInfo(keep); this.pollMetered(); this.pollColour();
      } catch (e) {
        this.testError = 'Preview request failed';
      } finally {
        this.busy = false;
      }
    },

    genLabel() { return (this.runs[this.testTarget] || {}).label || ''; },

    // Preview tab: the live-tuning dropdown's current value and its change handler.
    // Values are 'generated' (Tuned), 'standard' (Existing) or 'custom:<id>'.
    previewSel() {
      if (this.testKind === 'standard') return 'standard';
      if (this.testKind === 'custom') return `custom:${this.testVariant}`;
      return 'generated';
    },
    previewSelect(val) {
      if (val === 'standard') return this.previewStandard();
      if (val && val.startsWith('custom:')) return this.startPreviewTest('custom', val.slice(7));
      return this.startPreviewTest('generated');
    },

    // The exposure-panel state to carry across a tuning switch, or null on the
    // first start (nothing chosen yet — adopt the reloaded camera's defaults).
    _exposureState() {
      if (!this.testing) return null;
      const c = this.controls;
      return { auto_exposure: c.auto_exposure, exposure: c.exposure, gain: c.gain, ev: c.ev, fps: this.fpsTarget };
    },

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

    // Switch to one of the camera's advertised sensor modes ("WxH" from the
    // dropdown). Captures follow the selected mode's resolution.
    async applyMode(val) {
      const [width, height] = val.split('x').map(Number);
      this.busy = true; this.testError = '';
      if (this.testing) this.previewSrc = '';  // close the MJPEG stream before the camera reconfigures
      try {
        const r = await fetch('/api/mode', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ width, height }),
        });
        if (!r.ok) throw new Error();
        const m = await r.json();
        this.camera.resolution = m.resolution;
        await this._loadCamInfo();  // the mode changes the control limits (exposure, frame duration)
      } catch (e) { this.testError = 'Failed to switch sensor mode'; }
      finally {
        this.busy = false;
        if (this.testing) this.previewSrc = '/api/preview?t=' + Date.now();
      }
    },

    // Fetch sensor model/resolution (and seed metered values) for the info box.
    // preserve: the prior exposure-panel state to carry over a camera reload (so
    // switching tuning keeps the user's manual/gain/exposure/EV/fps choices),
    // re-applied over the reloaded camera's fresh control limits.
    async _loadCamInfo(preserve = null) {
      try {
        const r = await fetch('/api/health');
        if (!r.ok) return;
        const h = await r.json();
        if (h.model) this.camera.model = h.model;
        if (h.resolution) this.camera.resolution = h.resolution;
        if (h.modes) this.camera.modes = h.modes;
        if (h.controls) {
          this.metered = { ...h.controls };
          this.controls = { ...h.controls };
          this.fpsTarget = h.controls.fps || 0;
          if (preserve) {
            this.controls.auto_exposure = preserve.auto_exposure;
            this.controls.exposure = preserve.exposure;
            this.controls.gain = preserve.gain;
            this.controls.ev = preserve.ev;
            this.controls.fps = preserve.fps;
            this.fpsTarget = preserve.fps || 0;
            this.applyControls();
          }
        }
      } catch (e) { /* best effort */ }
    },

    async applyControls() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.controls),
        });
        // In manual mode the form is the source of truth: the response carries
        // frame metadata that lags the request by a few frames, so writing it
        // back would bounce the sliders (the metered box shows the actuals).
        if (r.ok && this.controls.auto_exposure) this.controls = await r.json();
      } catch (e) { this.testError = 'Failed to set controls'; }
    },

    // Slider bound for manual exposure: the camera's advertised limit,
    // additionally capped at one frame time when a fixed framerate is set.
    expMax() {
      if (!this.controls.exposure_max) return null;  // limits not fetched yet
      const fpsCap = this.fpsTarget ? Math.floor(1000000 / this.fpsTarget) : this.controls.frame_duration_max;
      return Math.min(this.controls.exposure_max, fpsCap);
    },

    fpsMax() {
      // Fastest rate the current sensor mode allows (the frame duration clamps
      // to the mode's minimum, so a higher target silently runs slower).
      return this.metered.frame_duration_min ? 1000000 / this.metered.frame_duration_min : 0;
    },
    fpsUnachievable() {
      return !!(this.fpsTarget && this.fpsMax() && this.fpsTarget > this.fpsMax() + 0.05);
    },

    async applyFps() {
      try {
        const r = await fetch('/api/controls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fps: this.fpsTarget || 0 }),
        });
        if (r.ok) {
          const c = await r.json();
          this.fpsTarget = c.fps || 0;
          this.controls.fps = c.fps;  // keep applyControls' payload in step with the new target
        }
      } catch (e) { this.testError = 'Failed to set framerate'; }
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
          // In auto mode the greyed-out sliders track the values AEC is
          // actually using, so switching to manual continues from there.
          if (this.controls.auto_exposure) {
            this.controls.exposure = this.metered.exposure;
            this.controls.gain = this.metered.gain;
          }
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
        let ch = lb.channel;  // capture before mutating: lb IS this.lightbox below
        // A fresh device reports channel 0 (nothing selected yet); snap to the
        // first illuminant so the model matches what the dropdown renders.
        if (lb.present && lb.illuminants && lb.illuminants[ch] == null) ch = Number(Object.keys(lb.illuminants)[0]);
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
      // Switching tuning while already previewing carries the exposure panel over.
      const keep = this._exposureState();
      this.previewSrc = '';  // close the stream before the camera restarts
      try {
        const r = await fetch('/api/preview-default', { method: 'POST' });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { this.testError = d.error || 'Failed to load standard tuning'; return; }
        // Keep testTarget so the Custom button stays visible for an A/B back to it.
        this.testTuning = 'default (built-in)'; this.testKind = 'standard';
        this.testing = true;
        this.previewSrc = '/api/preview?t=' + Date.now();
        if (this.hflip || this.vflip) await this._postTransform();  // re-apply flip on the fresh camera
        this._loadCamInfo(keep); this.pollMetered(); this.pollColour();
      } catch (e) {
        this.testError = 'Preview request failed';
      } finally {
        this.busy = false;
      }
    },

    // Semi-live colour accuracy: while the preview test runs, periodically
    // locate the Macbeth chart in a processed frame and ΔE its patches —
    // the current reading for whichever tuning is loaded.
    pollColour() {
      if (this._colourPolling) return;
      this._colourPolling = true;
      const tick = async () => {
        if (!this.testing || !this.liveColour) { this._colourPolling = false; return; }
        if (!document.hidden) {
          try {
            const r = await fetch('/api/macbeth-deltae');
            const d = await r.json().catch(() => ({}));
            if (r.ok && d.found) {
              this.chartSeen = { confidence: d.confidence };
              this.colour = d;
              this.$nextTick(() => requestAnimationFrame(() => { try { this.renderLiveDe(); } catch (e) { console.error(e); } }));
            } else {
              this.chartSeen = false;
            }
          } catch (e) { /* transient */ }
        }
        if (this.testing && this.liveColour) setTimeout(tick, 2000); else this._colourPolling = false;
      };
      tick();
    },

    renderLiveDe() {
      const ctx = document.getElementById('liveDeChart');
      const d = this.colour;
      if (!ctx || !d) return;
      const name = this.testKind === 'standard' ? 'system'
        : (this.testKind === 'custom' ? (this.testLabel || 'custom') : 'tuned');
      const label = `ΔE · ${name}`;
      const chart = this._charts.livede;
      if (chart) {
        // Update in place: a destroy/recreate every poll would flicker.
        chart.data.datasets[0].label = label;
        chart.data.datasets[0].data = d.patches.map((p) => p.de);
        chart.data.datasets[0].backgroundColor = d.patches.map((p) => `rgb(${p.rgb[0]},${p.rgb[1]},${p.rgb[2]})`);
        chart.update('none');
        return;
      }
      const opts = chartOpts('Macbeth patch', 'ΔE (CIE2000)');
      opts.plugins.legend.display = false;
      opts.plugins.tooltip = { callbacks: { title: (i) => 'Patch ' + (i[0].dataIndex + 1) } };
      this._charts.livede = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: d.patches.map((_, i) => i + 1),
          datasets: [{
            label,
            data: d.patches.map((p) => p.de),
            backgroundColor: d.patches.map((p) => `rgb(${p.rgb[0]},${p.rgb[1]},${p.rgb[2]})`),
            borderColor: 'rgba(255,255,255,0.25)', borderWidth: 1,
          }],
        },
        options: opts,
      });
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
      if (cfg.showJson) this.loadTuningData(t);  // adopt the target's default selection
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

    // Tuning page: fetch the selected target's original text + the chosen
    // variant's text/diff. variantId null/undefined sends no ?variant; the server
    // returns the first variant, which an initial load (undefined) adopts.
    // Load the tuning-data for target t and show selection `sel`: 'original',
    // 'existing' (built-in), or a variant slug. undefined keeps/adopts the
    // default (the server's first variant, else Original).
    async loadTuningData(t, sel = undefined) {
      this.tuningHtml = ''; this.diffHtml = ''; this.editing = false; this.tuningError = '';
      const token = ++this._tuningReq;
      const wantVariant = (sel && sel !== 'original' && sel !== 'existing') ? sel : null;
      try {
        const q = wantVariant ? `?variant=${encodeURIComponent(wantVariant)}` : '';
        const r = await fetch(`/projects/${this.project}/tuning-data/${t}${q}`);
        if (!r.ok) return;
        const s = await r.json();
        if (this.target !== t || token !== this._tuningReq) return;  // ignore a stale fetch
        this.tuningState = s;
        this.tuningSel = sel !== undefined ? sel : (s.selected || 'original');
        this._syncVariantId();
        this.renderTuning();
      } catch (e) { /* box stays empty */ }
    },

    // variantId is the slug only when a real variant is shown (drives Edit/Copy/
    // Remove/Download); null for Original or Existing.
    _syncVariantId() {
      this.variantId = (this.tuningSel === 'original' || this.tuningSel === 'existing') ? null : this.tuningSel;
    },

    // The text currently shown in the contents box (original / existing / variant).
    _shownText() {
      const s = this.tuningState;
      if (this.tuningSel === 'existing') return s.existing || '';
      if (this.variantId != null && s.custom != null) return s.custom;
      return s.original || '';
    },

    renderTuning() {
      this.tuningHtml = highlightJson(this._shownText());
      // Only a saved variant has a diff against the generated original.
      this.diffHtml = (this.variantId != null && this.tuningState.diff) ? highlightDiff(this.tuningState.diff) : '';
    },

    // Switch the contents box: 'original' | 'existing' | '<variant slug>'.
    selectTuning(v) {
      this.loadTuningData(this.target, v || 'original');
    },

    startEdit() {
      this.editorText = this._shownText();  // seed from whatever is shown (original/existing/variant)
      this.editing = true; this.tuningError = '';
    },

    // Adopt a fresh tuning-state response: update state, select the affected
    // variant (or fall back to Original), leave the editor.
    _applyTuningState(d) {
      this.tuningState = d;
      this.tuningSel = d.selected || 'original';
      this._syncVariantId();
      this.editing = false; this.tuningError = '';
      this.renderTuning();
    },

    // Save the editor contents back to the selected variant, in place.
    async saveEdit() {
      if (this.variantId == null) return this.saveAsNew();  // nothing selected → it needs a label
      try {
        const r = await fetch(`/projects/${this.project}/tuning/custom/${this.target}/${this.variantId}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ json: this.editorText }),
        });
        const d = await r.json();
        if (!r.ok) { this.tuningError = d.error || 'Save failed'; return; }
        this._applyTuningState(d);
      } catch (e) { this.tuningError = 'Save request failed'; }
    },

    // Save the editor contents as a brand-new labelled variant.
    async saveAsNew() {
      const label = (prompt('Name this custom tuning:') || '').trim();
      if (!label) return;
      try {
        const r = await fetch(`/projects/${this.project}/tuning/custom/${this.target}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, json: this.editorText }),
        });
        const d = await r.json();
        if (!r.ok) { this.tuningError = d.error || 'Save failed'; return; }
        this._applyTuningState(d);
      } catch (e) { this.tuningError = 'Save request failed'; }
    },

    // Duplicate the selected variant under a new label.
    async copyVariant() {
      if (this.variantId == null) return;
      const label = (prompt('Name for the copy:') || '').trim();
      if (!label) return;
      try {
        const r = await fetch(`/projects/${this.project}/tuning/custom/${this.target}/${this.variantId}/copy`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label }),
        });
        const d = await r.json();
        if (!r.ok) { this.tuningError = d.error || 'Copy failed'; return; }
        this._applyTuningState(d);
      } catch (e) { this.tuningError = 'Copy request failed'; }
    },

    // Remove the selected variant; the server returns the new default selection.
    async removeVariant() {
      if (this.variantId == null) return;
      if (!confirm('Remove this custom tuning?')) return;
      try {
        const r = await fetch(`/projects/${this.project}/tuning/custom/${this.target}/${this.variantId}/delete`, { method: 'POST' });
        if (!r.ok) return;
        this._applyTuningState(await r.json());
      } catch (e) { /* keep the current view */ }
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
      this._charts.gamut?.destroy();
      this.$nextTick(() => requestAnimationFrame(() => {
        try { this.renderCcm(); } catch (e) { console.error(e); }
        try { this.renderGamut(); } catch (e) { console.error(e); }
      }));
    },
    // The CCM step records per-patch chromaticity (u'v') only in newer sidecars.
    gamutHasData() {
      const e = this.ccmEntry();
      return !!(e && (e.patches || []).some((p) => p.uv && p.uv_ref));
    },

    renderGamut() {
      const entry = this.ccmEntry();
      const ctx = document.getElementById('gamutChart');
      if (!ctx) return;
      this._charts.gamut?.destroy();
      const patches = ((entry && entry.patches) || []).filter((p) => p.uv && p.uv_ref);
      const ref = this.metrics && this.metrics.gamut_reference;
      if (!patches.length || !ref) return;
      const xy = ([u, v]) => ({ x: u, y: v });
      const closed = (pts) => [...pts, pts[0]].map(xy);  // close the polygon
      const colours = patches.map((p) => `rgb(${p.rgb[0]},${p.rgb[1]},${p.rgb[2]})`);
      // Reference->corrected as one line dataset; null breaks split the segments.
      const vectors = [];
      patches.forEach((p) => vectors.push(xy(p.uv_ref), xy(p.uv), { x: null, y: null }));
      const datasets = [
        { type: 'line', label: 'spectral locus', data: closed(ref.locus),
          borderColor: '#39455a', borderWidth: 1, pointRadius: 0, fill: false, tension: 0 },
        { type: 'line', label: 'sRGB / Rec.709', data: closed(ref.srgb),
          borderColor: '#9aa7b8', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0 },
        { type: 'line', label: 'Rec.2020', data: closed(ref.rec2020),
          borderColor: '#6b7888', borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0 },
        { type: 'scatter', label: 'D65', data: [xy(ref.d65)],
          pointStyle: 'rectRot', pointRadius: 6, backgroundColor: '#e8ecf3', borderColor: '#e8ecf3' },
        { type: 'line', label: 'error', data: vectors, spanGaps: false,
          borderColor: 'rgba(229,72,77,0.55)', borderWidth: 1, pointRadius: 0, fill: false },
        { type: 'scatter', label: 'reference', data: patches.map((p) => xy(p.uv_ref)),
          pointRadius: 4, backgroundColor: 'transparent', borderColor: colours, borderWidth: 1.5 },
        { type: 'scatter', label: 'corrected', data: patches.map((p) => xy(p.uv)),
          pointRadius: 4, backgroundColor: colours, borderColor: 'rgba(255,255,255,0.4)', borderWidth: 1 },
      ];
      // Equal ranges on both axes + a square box (.chart-box.square) keep one
      // unit of u' the same length as one unit of v', so the locus isn't skewed.
      const opts = chartOpts("CIE u'", "CIE v'");
      opts.scales.x.type = 'linear'; opts.scales.x.min = 0; opts.scales.x.max = 0.65;
      opts.scales.y.min = 0; opts.scales.y.max = 0.65;
      this._charts.gamut = new Chart(ctx, { data: { datasets }, options: opts });
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
      try { if (this.gamutHasData()) this.renderGamut(); } catch (e) { console.error('Gamut chart:', e); }
      try {
        if (this.metrics && this.metrics.lux && (this.metrics.lux.samples || []).length) this.renderLux();
      } catch (e) { console.error('Lux chart:', e); }
      try {
        if (this.metrics && this.metrics.black_level && (this.metrics.black_level.frames || []).length
            && !this.blSinglePoint()) {
          this.renderBlackLevel();
        }
      } catch (e) { console.error('Black level chart:', e); }
      try {
        if (this.metrics && this.metrics.gamma && (this.metrics.gamma.curve || this.metrics.gamma.tone)) {
          this.renderGamma();
        }
      } catch (e) { console.error('Gamma chart:', e); }
    },

    // Opto-electronic transfer function (linear 0-1 -> code 0-1) for the selected
    // target. Mirrors ctt/algorithms/gamma_check.py so the target can be changed in
    // the browser without re-running the calibration (the measured data is fixed).
    gammaOetf(name) {
      const clamp = (x) => Math.min(1, Math.max(0, x));
      if (name === 'rec709') return (x) => clamp(x < 0.018 ? 4.5 * x : 1.099 * Math.pow(x, 0.45) - 0.099);
      if (name === 'rec2020') {
        const a = 1.09929682680944, b = 0.018053968510807;
        return (x) => clamp(x < b ? 4.5 * x : a * Math.pow(x, 0.45) - (a - 1));
      }
      if (name.startsWith('power:')) {
        const n = parseFloat(name.slice(6)) || 2.2;
        return (x) => clamp(Math.pow(Math.max(x, 0), 1 / n));
      }
      return (x) => clamp(x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);  // sRGB
    },

    // The standard targets plus whatever the run recorded (e.g. a custom power law).
    gammaTargetOptions() {
      const base = ['sRGB', 'rec709', 'rec2020', 'power:2.2', 'power:2.4'];
      const cur = this.metrics && this.metrics.gamma && this.metrics.gamma.target;
      if (cur && !base.includes(cur)) base.unshift(cur);
      return base;
    },
    gammaTargetLabel(t) {
      return { sRGB: 'sRGB', rec709: 'Rec.709', rec2020: 'Rec.2020' }[t] || t.replace('power:', 'Gamma ');
    },
    setGammaTarget(t) {
      this.gammaTarget = t;
      this.$nextTick(() => requestAnimationFrame(() => { try { this.renderGamma(); } catch (e) { console.error(e); } }));
    },

    // Per-grey reflectance ratios (white = 1) for recomputing the target output.
    // Prefer tone.patches; fall back to linearity.patches for sidecars written
    // before tone carried 'reflectance' (so the target is still recomputable).
    gammaReflRatios() {
      const g = (this.metrics && this.metrics.gamma) || {};
      const tone = (g.tone && g.tone.patches) || [];
      if (tone.length && tone[0].reflectance != null) return tone.map((p) => p.reflectance);
      const lin = (g.linearity && g.linearity.patches) || [];
      if (lin.length) { const mx = Math.max(...lin.map((p) => p.reflectance)); return lin.map((p) => p.reflectance / mx); }
      return tone.map(() => null);
    },

    // Headline figures, recomputed for the selected target. Linearity is a property
    // of the sensor (target-independent), so only tone/curve deviation move.
    gammaStats() {
      const g = (this.metrics && this.metrics.gamma) || {};
      const oetf = this.gammaOetf(this.gammaTarget);
      const rms = (a) => (a.length ? Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length) : NaN);
      const out = [];
      if (g.tone && g.tone.patches && g.tone.patches.length) {
        const rr = this.gammaReflRatios();
        const errs = g.tone.patches
          .map((p, i) => (rr[i] == null ? null : p.measured - oetf(rr[i]) * 255))
          .filter((e) => e != null);
        out.push(['Tone RMS (8-bit)', errs.length ? rms(errs).toFixed(1) : '—']);
      }
      if (g.curve && g.curve.points) {
        out.push(['Curve RMS (8-bit)', rms(g.curve.points.map((p) => (p.measured - oetf(p.x)) * 255)).toFixed(1)]);
      }
      if (g.linearity) out.push(['Linearity R²', g.linearity.r2.toFixed(4)]);
      return out;
    },

    renderGamma() {
      const g = this.metrics.gamma;
      const ctx = document.getElementById('gammaChart');
      if (!ctx) return;
      // Re-renders (target change) reuse the canvas; Chart.js needs the old chart
      // destroyed first or new Chart() throws and the stale chart stays.
      this._charts.gamma?.destroy();
      const oetf = this.gammaOetf(this.gammaTarget);
      const datasets = [];
      if (g.curve && g.curve.points) {
        const pts = g.curve.points;
        datasets.push({
          type: 'line', label: 'gamma curve',
          data: pts.map((p) => ({ x: p.x * 255, y: p.measured * 255 })),
          borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, tension: 0,
        });
        datasets.push({
          type: 'line', label: this.gammaTargetLabel(this.gammaTarget),
          data: pts.map((p) => ({ x: p.x * 255, y: oetf(p.x) * 255 })),
          borderColor: '#9aa7b8', borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0, tension: 0,
        });
      }
      if (g.tone && g.tone.patches) {
        datasets.push({
          type: 'scatter', label: 'measured greys',
          data: g.tone.patches.map((p) => ({ x: p.input * 255, y: p.measured })),
          backgroundColor: '#e5484d', pointRadius: 4,
        });
      }
      const opts = chartOpts('Input (linear, 8-bit)', 'Output code (8-bit)');
      opts.scales.x.type = 'linear';
      opts.scales.x.min = 0; opts.scales.x.max = 255;
      opts.scales.y.min = 0; opts.scales.y.max = 255;
      this._charts.gamma = new Chart(ctx, { data: { datasets }, options: opts });
    },

    // All dark frames at a single total exposure: a trend plot is meaningless,
    // so the card shows the measured per-channel values as stat boxes instead.
    blSinglePoint() {
      const f = (this.metrics && this.metrics.black_level && this.metrics.black_level.frames) || [];
      return f.length > 0 && new Set(f.map((p) => p.total_exposure)).size === 1;
    },

    blStats() {
      const bl = (this.metrics && this.metrics.black_level) || {};
      const f = bl.frames || [];
      if (!f.length) return [];
      const avg = (ch) => Math.round(f.reduce((s, p) => s + (p[ch] || 0), 0) / f.length);
      const out = ('y' in f[0])
        ? [['Y measured', avg('y')]]
        : [['R measured', avg('r')], ['G measured', avg('g')], ['B measured', avg('b')]];
      out.push(['Used', bl.black_level]);
      return out;
    },

    renderBlackLevel() {
      const bl = this.metrics.black_level;
      const ctx = document.getElementById('blackLevelChart');
      if (!ctx) return;
      const pts = bl.frames.slice().sort((a, b) => a.total_exposure - b.total_exposure);
      const xs = pts.map((p) => p.total_exposure / 1000);  // metrics carry µs; plot in ms
      // Colour sensors carry r/g/b per frame; mono frames a single y.
      const series = ('y' in (pts[0] || {}))
        ? [['y', '#9aa7b8']]
        : [['r', '#e5484d'], ['g', '#2fb344'], ['b', '#50a0ff']];
      const opts = chartOpts('Total exposure (ms × gain)', 'Black level (16-bit)');
      opts.scales.x.type = 'linear';
      opts.plugins.tooltip = { callbacks: {
        label: (i) => {
          const p = pts[i.dataIndex];
          return i.dataset.label === 'calibration'
            ? `calibration ${i.parsed.y}`
            : `${p.name} · ${i.dataset.label} ${i.parsed.y} · ${(p.exposure / 1000).toFixed(2)} ms × ${p.gain}`;
        },
      } };
      const datasets = series.map(([ch, colour]) => ({
        type: 'scatter', label: ch,
        data: pts.map((p) => ({ x: p.total_exposure / 1000, y: p[ch] })),
        backgroundColor: colour, pointRadius: 4,
      }));
      datasets.push({
        type: 'line', label: 'calibration',
        data: [{ x: Math.min(...xs), y: bl.black_level }, { x: Math.max(...xs), y: bl.black_level }],
        borderColor: '#9aa7b8', borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0,
      });
      this._charts.black_level = new Chart(ctx, { data: { datasets }, options: opts });
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

// --- MTF (slanted edge) card on the Results page ----------------------------
function mtfApp(cfg) {
  return {
    project: cfg.project,
    busy: false,
    measuring: false,
    hasCapture: false,
    connected: null,   // null until /api/health resolves
    live: false,       // true = live viewfinder, false = captured results view
    liveTick: 0,       // cache-bust so the MJPEG <img> reopens on re-entry
    camera: { model: '', resolution: null },
    metered: { exposure: 0, gain: 0, colour_temp: 0, lux: 0, focus_fom: 0 },
    hflip: false,
    vflip: false,
    previewSrc: '',
    rois: [],          // sensor-pixel coords {x, y, w, h}
    results: null,
    error: '',
    _nat: { w: 0, h: 0 },  // captured image natural (sensor) size
    _chart: null,

    async init() {
      // A chart captured in an earlier session is reusable; probe for it.
      this.previewSrc = `/projects/${this.project}/mtf/preview?t=${Date.now()}`;
      const f = loadFlip();  // restore the flip choice (persists across tabs)
      this.hflip = !!f.hflip; this.vflip = !!f.vflip;
      try {
        const r = await fetch('/api/health');
        const h = await r.json();
        this.connected = r.ok && h.camera === true;
        if (h.model) this.camera.model = h.model;
        if (h.resolution) this.camera.resolution = h.resolution;
      } catch (e) {
        this.connected = false;
      }
      if (this.connected) {
        // The page load reset the sensor flip; re-enforce the stored choice.
        if (this.hflip || this.vflip) await this.applyTransform();
        this.liveView();
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
        this.liveTick = Date.now();  // reopen the MJPEG stream on the reconfigured camera
      } catch (e) { this.error = 'Failed to set flip'; }
    },

    liveView() {
      this.live = true;
      this.liveTick = Date.now();
      this.pollMetered();
    },

    // Poll metered exposure/gain/Focus FoM while the viewfinder is up, so the
    // chart can be focused right here before measuring.
    pollMetered() {
      if (this._polling) return;
      this._polling = true;
      const tick = async () => {
        if (!this.live || !this.connected) { this._polling = false; return; }
        try {
          const r = await fetch('/api/controls');
          if (r.ok) this.metered = await r.json();
        } catch (e) { /* transient */ }
        if (this.live) setTimeout(tick, 1500); else this._polling = false;
      };
      tick();
    },

    // Detect patches: capture a raw DNG of whatever is framed and place
    // measurement regions on the edges it finds. Measurement is a separate,
    // deliberate step so manual patches can be added/removed first.
    async detectLive() {
      this.busy = true; this.error = ''; this.results = null; this.rois = [];
      try {
        const r = await fetch(`/projects/${this.project}/mtf/capture`, { method: 'POST' });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          this.error = d.error || 'Capture failed — is a camera connected?';
          return;
        }
        this.live = false;  // switch to the patch-editing view; the MJPEG stream closes
        this.previewSrc = `/projects/${this.project}/mtf/preview?t=${Date.now()}`;  // imageLoaded re-fires
        await this.autoDetect();
      } catch (e) {
        this.error = 'Capture request failed';
      } finally {
        this.busy = false;
      }
    },

    // Default ROI box size: ~6% of the short sensor edge.
    roiSize() {
      return Math.max(96, Math.min(384, Math.round(Math.min(this._nat.w, this._nat.h) * 0.06)));
    },

    imageLoaded(e) {
      this._nat = { w: e.target.naturalWidth, h: e.target.naturalHeight };
      this.hasCapture = true;
      this.$nextTick(() => this.drawRois());
    },

    async autoDetect() {
      // Places patches only (unmeasured, pink); Measure produces the numbers.
      this.busy = true; this.error = ''; this.results = null;
      try {
        const r = await fetch(`/projects/${this.project}/mtf/auto`, { method: 'POST' });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { this.error = d.error || 'Edge detection failed'; return; }
        this.rois = d.rois.map((r2) => ({ x: r2.x, y: r2.y, w: r2.w, h: r2.h }));
        if (!d.rois.length) this.error = 'No measurable edges found — check focus/framing, or slant the chart a few degrees.';
        this.drawRois();
      } catch (e) {
        this.error = 'Edge detection request failed';
      } finally {
        this.busy = false;
      }
    },

    clearRois() { this.rois = []; this.results = null; this.drawRois(); },

    // Pointer position -> sensor coords.
    _toSensor(e) {
      const img = document.getElementById('mtfChartImg');
      if (!img || !this._nat.w) return null;
      const rect = img.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) * (this._nat.w / rect.width),
        y: (e.clientY - rect.top) * (this._nat.h / rect.height),
      };
    },

    // Click adds a default-size box (or removes the box under the cursor);
    // click-drag draws a box of any size.
    roiDown(e) {
      const p = this._toSensor(e);
      if (!p) return;
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      this._drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, moved: false };
    },

    roiMove(e) {
      if (!this._drag) return;
      const p = this._toSensor(e);
      if (!p) return;
      this._drag.x1 = p.x; this._drag.y1 = p.y;
      // Ignore jitter below ~6 display px so a click stays a click.
      const img = document.getElementById('mtfChartImg');
      const k = img ? this._nat.w / img.getBoundingClientRect().width : 1;
      if (Math.abs(p.x - this._drag.x0) > 6 * k || Math.abs(p.y - this._drag.y0) > 6 * k) this._drag.moved = true;
      if (this._drag.moved) this.drawRois();
    },

    roiUp(e) {
      const d = this._drag;
      this._drag = null;
      if (!d) return;
      if (d.moved) {
        const x = Math.round(Math.max(Math.min(d.x0, d.x1), 0));
        const y = Math.round(Math.max(Math.min(d.y0, d.y1), 0));
        const w = Math.round(Math.abs(d.x1 - d.x0));
        const h = Math.round(Math.abs(d.y1 - d.y0));
        if (w >= 32 && h >= 32) this.rois.push({ x, y, w, h });
      } else {
        // Plain click: remove the box under the cursor, else add a default box.
        const hit = this.rois.findIndex(
          (r) => d.x0 >= r.x && d.x0 <= r.x + r.w && d.y0 >= r.y && d.y0 <= r.y + r.h);
        if (hit >= 0) this.rois.splice(hit, 1);
        else {
          const s = this.roiSize();
          this.rois.push({
            x: Math.round(Math.min(Math.max(d.x0 - s / 2, 0), this._nat.w - s)),
            y: Math.round(Math.min(Math.max(d.y0 - s / 2, 0), this._nat.h - s)),
            w: s, h: s,
          });
        }
      }
      this.results = null;
      this.drawRois();
    },

    drawRois() {
      const img = document.getElementById('mtfChartImg');
      const cv = document.getElementById('mtfOverlay');
      if (!img || !cv || !this._nat.w) return;
      cv.width = img.clientWidth; cv.height = img.clientHeight;
      const kx = cv.width / this._nat.w, ky = cv.height / this._nat.h;
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.lineWidth = 2;
      ctx.font = '600 13px sans-serif';
      this.rois.forEach((r, i) => {
        const ok = this.results ? this.results[i] && this.results[i].ok : null;
        ctx.strokeStyle = ok === null ? '#f06595' : ok ? '#2fb344' : '#e5484d';
        ctx.strokeRect(r.x * kx, r.y * ky, r.w * kx, r.h * ky);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(`${i + 1}`, r.x * kx + 4, r.y * ky + 16);
      });
      // Rubber band while drawing a new box.
      const d = this._drag;
      if (d && d.moved) {
        ctx.strokeStyle = '#f06595';
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(Math.min(d.x0, d.x1) * kx, Math.min(d.y0, d.y1) * ky,
                       Math.abs(d.x1 - d.x0) * kx, Math.abs(d.y1 - d.y0) * ky);
        ctx.setLineDash([]);
      }
    },

    async measure() {
      this.measuring = true; this.error = '';
      try {
        const r = await fetch(`/projects/${this.project}/mtf/measure`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rois: this.rois }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { this.error = d.error || 'Measurement failed'; return; }
        this.results = d.rois;
        // The server snaps each box onto the strongest nearby edge; show that.
        this.rois = d.rois.map((r2) => ({ x: r2.x, y: r2.y, w: r2.w, h: r2.h }));
        this.drawRois();  // recolour boxes by result, at their snapped positions
        this.$nextTick(() => requestAnimationFrame(() => this.renderCurves()));
      } catch (e) {
        this.error = 'Measurement request failed';
      } finally {
        this.measuring = false;
      }
    },

    renderCurves() {
      const ctx = document.getElementById('mtfCurveChart');
      if (!ctx || !this.results) return;
      this._chart?.destroy();
      const palette = ['#f06595', '#3b82f6', '#2fb344', '#f1a204', '#a78bfa', '#22d3ee'];
      const opts = chartOpts('Spatial frequency (cycles/pixel)', 'MTF');
      opts.scales.x.type = 'linear';
      opts.scales.x.max = 0.5;
      opts.scales.y.min = 0;
      opts.scales.y.max = 1.05;
      this._chart = new Chart(ctx, {
        type: 'line',
        data: {
          datasets: this.results.map((r, i) => ({ r, i })).filter(({ r }) => r.ok).map(({ r, i }) => ({
            label: `#${i + 1}${r.zone ? ' ' + r.zone : ''}` + (r.mtf50 !== null ? ` (${r.mtf50.toFixed(3)})` : ''),
            data: r.curve.map((p) => ({ x: p.f, y: p.mtf })),
            borderColor: palette[i % palette.length],
            backgroundColor: palette[i % palette.length],
            pointRadius: 0, borderWidth: 2, tension: 0.2,
          })),
        },
        options: opts,
      });
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
