# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Slanted-edge MTF (e-SFR) measurement, ISO 12233 style.
#
# Operates on the raw DNG green plane: linear sensor data with no ISP
# sharpening or denoise, so the result is the true lens+sensor MTF. The edge
# must be slanted a few degrees off vertical/horizontal — the slant is what
# yields a sub-pixel-supersampled edge profile. Pipeline per ROI:
#
#   per-row edge centroid -> line fit (angle) -> project pixels onto the
#   edge normal -> 4x oversampled ESF -> derivative (LSF) -> Hamming window
#   -> FFT -> MTF curve -> MTF50.
#
# The green plane samples every other sensor pixel on each axis, so plane
# frequencies are halved to report in cycles per *sensor* pixel; thanks to
# the slant oversampling the measurement stays valid up to sensor Nyquist.

import numpy as np

# Bin width of the supersampled ESF, in (plane) pixels.
_OVERSAMPLE = 4
# Acceptable edge slant range (ISO 12233 recommends ~2-10°): too square gives
# aliased bins; steeper than ~10° is usually a wedge/curve, not a slanted edge.
_MIN_ANGLE_DEG, _MAX_ANGLE_DEG = 0.5, 10.0
# A green-plane sample sits every 2 sensor pixels on each axis.
_PLANE_STEP = 2
# Sensor pixels per degree of view for the acutance weighting: roughly a
# full-screen viewing distance. It places the CSF peak (~8 cy/deg) in the
# mid-frequency band sharpening boosts; only ratios of acutance are compared,
# so the exact viewing assumption is not critical.
_CSF_PX_PER_DEG = 40.0


def _fail(reason: str) -> dict:
    return {'ok': False, 'reason': reason}


def analyse_edge(
    roi: np.ndarray,
    *,
    plane_step: int = _PLANE_STEP,
    min_monotonicity: float = 0.4,
    edge_line: tuple[float, float] | None = None,
) -> dict:
    """Measure the MTF of a single slanted edge in a 2D plane ROI.

    plane_step is how many sensor pixels one plane sample spans: 2 for the raw
    green plane (the default), 1 for a full-resolution processed luma plane.
    min_monotonicity is the single-clean-edge acceptance bar: the fraction of
    the (smoothed) ESF's total variation the net level change must account
    for. The default rejects bars/wedges/zone plates in unvalidated ROIs;
    callers measuring a pre-validated edge that may carry heavy sharpening
    halos and high-gain noise should relax it (halos legitimately add
    variation on top of the step). edge_line optionally fixes the edge
    geometry to a previously fitted (slope, x0) — repeated measurements of a
    static scene should not re-estimate the line from each noisy capture,
    where a marginal fit can flap across the angle/border guards; the fitted
    line is returned as 'edge_line' for exactly this reuse.

    Returns {'ok': True, 'angle_deg', 'mtf50', 'mtf_peak', 'overshoot',
    'undershoot', 'edge_low', 'edge_high', 'curve': [{'f', 'mtf'}, ...]} with
    frequencies in cycles per sensor pixel, or {'ok': False, 'reason'}.
    mtf_peak is the maximum of the DC-normalised curve (> 1 means over-unity,
    i.e. sharpened, response); overshoot/undershoot are the ESF halo excursions
    beyond the bright/dark plateaus as a fraction of the edge step — ~0 on an
    unsharpened edge. Both halo directions are reported because the sharpening
    hardware deliberately applies more negative than positive gain.
    """
    roi = np.asarray(roi, dtype=np.float64)
    if roi.ndim != 2 or min(roi.shape) < 16:
        return _fail('ROI too small')

    # Work with a near-vertical edge: if the gradient is predominantly along
    # rows (a horizontal edge), transpose.
    gy = np.abs(np.diff(roi, axis=0)).mean()
    gx = np.abs(np.diff(roi, axis=1)).mean()
    if gy > gx:
        roi = roi.T
    h, w = roi.shape

    # Contrast sanity check: an edge needs a real step.
    lo, hi = np.percentile(roi, (5, 95))
    if hi - lo <= 1e-9 or (hi - lo) / max(hi, 1e-9) < 0.2:
        return _fail('No edge found (ROI contrast too low)')

    # Per-row edge position: centroid of the squared row derivative. The
    # good-row mask applies in both modes — rows whose gradient energy lies
    # on other content (a neighbouring feature in the box) would smear the
    # ESF projection.
    deriv = np.diff(roi, axis=1)
    weights = deriv**2
    row_strength = weights.sum(axis=1)
    good = row_strength > 0.1 * row_strength.max()
    if good.sum() < 8:
        return _fail('No edge found (too few usable rows)')
    rows = np.arange(h)[good]

    if edge_line is not None:
        # Fixed geometry from a previous fit of the same (static) edge; the
        # angle was validated back then.
        slope, x0 = float(edge_line[0]), float(edge_line[1])
        angle = float(np.degrees(np.arctan(slope)))
    else:
        xs = np.arange(w - 1) + 0.5
        centroids = (weights[good] * xs).sum(axis=1) / row_strength[good]

        # Fit the edge line x = slope*y + x0 and check the slant angle.
        slope, x0 = np.polyfit(rows, centroids, 1)
        angle = float(np.degrees(np.arctan(slope)))
        if not (_MIN_ANGLE_DEG <= abs(angle) <= _MAX_ANGLE_DEG):
            return _fail(f'Edge angle {angle:.1f}° outside {_MIN_ANGLE_DEG}–{_MAX_ANGLE_DEG}° (slant the chart)')

    # Project pixels onto the edge normal and bin into the oversampled ESF.
    # Only rows where the edge was actually detected take part: rows beyond the
    # end of a bar (or containing other content) would otherwise smear their
    # values across every distance bin and corrupt the ESF.
    cc, rr = np.meshgrid(np.arange(w, dtype=np.float64), rows.astype(np.float64))
    dist = (cc - (x0 + slope * rr)) * np.cos(np.arctan(slope))
    span = min(16.0, (w / 2) - 2)
    idx = np.round(dist * _OVERSAMPLE).astype(int)
    keep = np.abs(idx) <= int(span * _OVERSAMPLE)
    idx = idx[keep] + int(span * _OVERSAMPLE)
    vals = roi[good][keep]
    nbins = 2 * int(span * _OVERSAMPLE) + 1
    sums = np.bincount(idx, weights=vals, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    esf = np.full(nbins, np.nan)
    nz = counts > 0
    esf[nz] = sums[nz] / counts[nz]
    if nz.sum() < nbins * 0.8:
        return _fail('Edge projection too sparse')
    # Fill any empty bins by interpolation.
    bins = np.arange(nbins)
    esf = np.interp(bins, bins[nz], esf[nz])

    # A real edge has an essentially monotonic ESF: the net level change should
    # dominate the total variation. Oscillating content (zone plates, wedges, a
    # whole square inside the ROI) fails this even when it has strong gradients.
    # Validity and halo statistics use a lightly smoothed copy (one plane px)
    # so per-bin noise doesn't masquerade as structure; the MTF path below
    # keeps the raw ESF.
    kernel = np.ones(_OVERSAMPLE + 1) / (_OVERSAMPLE + 1)
    # Edge-pad before convolving: zero padding would drag both ends of the
    # smoothed ESF towards zero and corrupt the net-change measurement.
    pad = len(kernel) // 2
    esf_smooth = np.convolve(np.pad(esf, pad, mode='edge'), kernel, mode='valid')
    total_variation = np.abs(np.diff(esf_smooth)).sum()
    if total_variation <= 0 or abs(esf_smooth[-1] - esf_smooth[0]) / total_variation < min_monotonicity:
        return _fail('ROI is not a single clean edge')

    # Halo metrics from the ESF. Plateau levels come from the outer quarters
    # (well clear of any few-pixel sharpening halo around the edge) via
    # medians; the halo extrema are searched only in the central half so
    # plateau noise can never register as a halo.
    oriented = esf_smooth if esf_smooth[-1] >= esf_smooth[0] else esf_smooth[::-1]
    quarter = nbins // 4
    edge_low = float(np.median(oriented[:quarter]))
    edge_high = float(np.median(oriented[-quarter:]))
    step = edge_high - edge_low
    core_esf = oriented[quarter : nbins - quarter]
    overshoot = undershoot = None
    if step > 1e-9 and core_esf.size:
        overshoot = max(0.0, (float(core_esf.max()) - edge_high) / step)
        undershoot = max(0.0, (edge_low - float(core_esf.min())) / step)

    # LSF, windowed around its peak to suppress noise far from the edge. The
    # peak is located on the smoothed profile so a single noisy bin cannot
    # drag the window to the ROI border; with fixed geometry the edge is at
    # the projection centre by construction, so the search stays in the
    # central half (residual off-line content must not steal the window).
    lsf = np.gradient(esf)
    search = np.abs(np.gradient(esf_smooth))
    if edge_line is not None:
        q = nbins // 4
        peak = q + int(np.argmax(search[q : nbins - q]))
    else:
        peak = int(np.argmax(search))
    half = min(peak, nbins - 1 - peak)
    if half < 4 * _OVERSAMPLE:
        return _fail('Edge too close to the ROI border')
    window = np.zeros(nbins)
    window[peak - half : peak + half + 1] = np.hamming(2 * half + 1)
    lsf = lsf * window

    # MTF: FFT magnitude, DC-normalised. Plane freq -> sensor freq (/plane_step).
    mtf = np.abs(np.fft.rfft(lsf))
    if mtf[0] <= 0:
        return _fail('Degenerate edge response')
    mtf = mtf / mtf[0]
    freqs = np.fft.rfftfreq(nbins, d=1.0 / _OVERSAMPLE) / plane_step

    # Keep the curve up to sensor Nyquist (0.5 cycles/pixel).
    upto = freqs <= 0.5
    freqs, mtf = freqs[upto], mtf[upto]

    # Perceptual sharpness: the CSF-weighted mean of the MTF (a simplified
    # CPIQ acutance, Mannos-Sakrison CSF). Sharpening lifts the mid/high
    # band and this integral follows it, where MTF50 — the 50% crossing,
    # often pinned by the optics — barely moves.
    f_deg = freqs * _CSF_PX_PER_DEG
    csf = 2.6 * (0.0192 + 0.114 * f_deg) * np.exp(-((0.114 * f_deg) ** 1.1))
    acutance = float((mtf * csf).sum() / max(csf.sum(), 1e-9))

    # MTF50: first 0.5 crossing, linearly interpolated.
    mtf50 = None
    below = np.where(mtf < 0.5)[0]
    if below.size and below[0] > 0:
        i = below[0]
        f0, f1, m0, m1 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
        mtf50 = float(f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0))

    return {
        'ok': True,
        'angle_deg': round(angle, 2),
        'edge_line': [round(float(slope), 6), round(float(x0), 3)],
        'mtf50': round(mtf50, 4) if mtf50 is not None else None,
        'mtf_peak': round(float(mtf.max()), 4),
        'acutance': round(acutance, 4),
        'overshoot': round(overshoot, 4) if overshoot is not None else None,
        'undershoot': round(undershoot, 4) if undershoot is not None else None,
        'edge_low': round(edge_low, 2),
        'edge_high': round(edge_high, 2),
        'curve': [{'f': round(float(f), 4), 'mtf': round(float(m), 4)} for f, m in zip(freqs, mtf, strict=True)],
    }


def green_plane(dng_path: str) -> np.ndarray:
    """Extract a black-level-subtracted green plane from a DNG (linear floats).

    Uses one of the two green Bayer positions; the plane samples every other
    sensor pixel on each axis (callers index it with sensor coords // 2).
    """
    import rawpy  # noqa: PLC0415 (heavy import; server also lazy-imports it)

    with rawpy.imread(dng_path) as raw:
        pattern = raw.raw_pattern
        ys, xs = np.where(pattern == 1)  # rawpy colour index 1 = first green
        y, x = int(ys[0]), int(xs[0])
        plane = raw.raw_image[y::2, x::2].astype(np.float64)
        colour = int(pattern[y, x])
        black = float(raw.black_level_per_channel[colour])
    return np.maximum(plane - black, 0.0)


def _snap_roi(plane: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int]:
    """Fine-tune an ROI so the strongest edge *inside it* sits at its centre.

    Searches only within the box the user drew — the adjusted box always
    overlaps the original footprint, so it can centre a slightly-off edge but
    can never wander off to some stronger edge elsewhere on the chart.
    """
    ph, pw = plane.shape
    crop = plane[y : min(y + h, ph), x : min(x + w, pw)]
    if crop.size == 0 or min(crop.shape) < 4:
        return x, y
    col_strength = np.abs(np.diff(crop, axis=1)).sum(axis=0)
    row_strength = np.abs(np.diff(crop, axis=0)).sum(axis=1)
    kernel = np.ones(5) / 5
    col_strength = np.convolve(col_strength, kernel, mode='same')
    row_strength = np.convolve(row_strength, kernel, mode='same')
    if col_strength.max() >= row_strength.max():
        x += int(np.argmax(col_strength)) - w // 2  # centre on the vertical edge
    else:
        y += int(np.argmax(row_strength)) - h // 2  # centre on the horizontal edge
    return min(max(x, 0), pw - w), min(max(y, 0), ph - h)


def _zone(x: int, y: int, w: int, h: int, pw: int, ph: int) -> str:
    """Human-readable frame zone ('top-left', 'centre', ...) for a box."""
    fx, fy = (x + w / 2) / pw, (y + h / 2) / ph
    col = 'left' if fx < 1 / 3 else 'right' if fx > 2 / 3 else 'centre'
    row = 'top' if fy < 1 / 3 else 'bottom' if fy > 2 / 3 else 'middle'
    return 'centre' if (col, row) == ('centre', 'middle') else f'{row}-{col}'


def auto_detect(dng_path: str, max_regions: int = 9) -> list[dict]:
    """Find measurable slanted edges on a DNG's green plane (sensor px)."""
    return detect_edges(green_plane(dng_path), plane_step=_PLANE_STEP, max_regions=max_regions)


def detect_edges(plane: np.ndarray, *, plane_step: int = _PLANE_STEP, max_regions: int = 9) -> list[dict]:
    """Find measurable slanted edges automatically in any 2D plane.

    Tiles the plane with edge-shaped candidate boxes (tall for vertical
    edges, wide for horizontal), keeps the ones that pass the full analysis,
    removes overlapping duplicates preferring higher-contrast edges, and
    spreads the survivors across frame zones so corners are represented, not
    just the strongest cluster. Returns measure_rois-shaped dicts with
    coordinates in sensor pixels (plane px x plane_step) and each candidate's
    plane-level contrast, so callers can classify edges by step height.
    """
    ph, pw = plane.shape
    white = np.percentile(plane, 99)
    candidates = []
    for bw, bh in ((32, 64), (64, 32)):  # plane px: tall boxes catch vertical edges, wide catch horizontal
        for y in range(0, ph - bh + 1, bh):
            for x in range(0, pw - bw + 1, bw):
                crop = plane[y : y + bh, x : x + bw]
                lo, hi = float(crop.min()), float(crop.max())
                if hi - lo < 0.25 * white:  # cheap pre-filter: needs real contrast
                    continue
                sx, sy = _snap_roi(plane, x, y, bw, bh)
                result = analyse_edge(plane[sy : sy + bh, sx : sx + bw], plane_step=plane_step)
                if result['ok'] and result['mtf50'] is not None:
                    candidates.append({'x': sx, 'y': sy, 'w': bw, 'h': bh, 'contrast': hi - lo, **result})

    # Dedupe: snapped boxes converge onto the same edges; keep the higher-contrast one.
    candidates.sort(key=lambda c: -c['contrast'])
    accepted = []
    for c in candidates:
        if all(abs(c['x'] - a['x']) >= c['w'] or abs(c['y'] - a['y']) >= c['h'] for a in accepted):
            accepted.append(c)

    # Spread across frame zones (round-robin) so the pick isn't one bright cluster.
    by_zone: dict[str, list[dict]] = {}
    for c in accepted:
        by_zone.setdefault(_zone(c['x'], c['y'], c['w'], c['h'], pw, ph), []).append(c)
    picked = []
    while len(picked) < max_regions and any(by_zone.values()):
        for zone in list(by_zone):
            if by_zone[zone] and len(picked) < max_regions:
                c = by_zone[zone].pop(0)
                picked.append({**c, 'zone': zone})

    out = []
    for c in picked:
        out.append(
            {
                **c,
                'contrast': round(float(c['contrast']), 2),
                'x': c['x'] * plane_step,
                'y': c['y'] * plane_step,
                'w': c['w'] * plane_step,
                'h': c['h'] * plane_step,
            }
        )
    out.sort(key=lambda c: (c['y'], c['x']))
    return out


def measure_rois(dng_path: str, rois: list[dict]) -> list[dict]:
    """Measure each ROI ({'x','y','w','h'} in sensor pixels) of a DNG's green plane.

    Each ROI is first snapped onto the strongest edge near it; the echoed
    x/y in the results are the snapped (sensor-pixel) positions.
    """
    plane = green_plane(dng_path)
    ph, pw = plane.shape
    out = []
    for roi in rois:
        x, y = int(roi['x']) // _PLANE_STEP, int(roi['y']) // _PLANE_STEP
        w, h = max(int(roi['w']) // _PLANE_STEP, 1), max(int(roi['h']) // _PLANE_STEP, 1)
        x, y = min(max(x, 0), max(pw - w, 0)), min(max(y, 0), max(ph - h, 0))
        sx, sy = _snap_roi(plane, x, y, w, h)
        crop = plane[sy : min(sy + h, ph), sx : min(sx + w, pw)]
        result = analyse_edge(crop) if crop.size else _fail('ROI outside the image')
        ex, ey = sx, sy
        if not result['ok'] and (sx, sy) != (x, y):
            # The snap can occasionally centre on the wrong feature; fall back
            # to exactly the box the user drew before reporting a failure.
            retry = analyse_edge(plane[y : min(y + h, ph), x : min(x + w, pw)])
            if retry['ok']:
                result = retry
            ex, ey = x, y  # failed boxes stay where the user put them
        elif not result['ok']:
            ex, ey = x, y
        zone = _zone(ex, ey, w, h, pw, ph)
        out.append({**roi, 'x': ex * _PLANE_STEP, 'y': ey * _PLANE_STEP, 'zone': zone, **result})
    return out
