# Sensor characterisation: experiments and measurements

Notes from a full characterisation of a raw image sensor, recorded here as a
reference for adding characterisation measurements to the tuning tool. The
methodology is sensor-agnostic: every metric below is computed off-device from
bursts of raw Bayer frames, so the same capture protocol supports all of them.

## General capture protocol

All experiments share one capture recipe; get this right and every measurement
below follows from the same kind of data.

- **Raw only, processing off.** Capture unprocessed Bayer frames with
  auto-exposure, auto-white-balance and lens shading correction disabled, so the
  sensor is measured rather than an image pipeline. On libcamera/picamera2 this
  means a still configuration with a raw stream (e.g. `SRGGB12` requested,
  delivered as 16-bit unpacked) and `AeEnable=False`, `AwbEnable=False`.
- **Native bit depth.** Undo the 16-bit left-justification of the packed raw and
  quote everything in native DN (e.g. 0--4095 for a 12-bit sensor).
- **Bursts, with settle frames.** For every operating point, discard a few
  frames after changing exposure/gain (controls take 1--3 frames to land — read
  back the applied exposure from frame metadata to confirm), then capture a
  burst (8--16 frames) for temporal statistics.
- **Central ROI.** Use a centred region (e.g. 25% of the frame) for photometric
  work: it bounds payload size, avoids lens shading and vignetting at the edges,
  and keeps flat-field non-uniformity out of the temporal statistics.
- **Single green channel.** Compute all noise and photon-transfer statistics on
  one green channel only (Gr). Averaging Gr and Gb halves the per-pixel variance
  and doubles the apparent conversion gain — a classic trap.
- **Two kinds of statistics.** *Temporal* statistics (per pixel, across a burst)
  isolate read + shot noise. *Spatial* statistics (across pixels of the
  time-averaged frame, after removing a low-order shading fit) give fixed-pattern
  non-uniformity (PRNU/DSNU).
- **Stimuli.** A diffuse flat field, enclosed with the sensor against ambient
  light, for all photometric/noise work; a ColorChecker Classic under multiple
  measured illuminants for colour. An independent light meter (lux + CCT)
  provides the illuminant reference — do not trust illuminant labels (see
  pitfalls).

## Experiments

### 1. Black level and dark behaviour

Source off, sensor enclosed against ambient. Capture dark bursts and measure:

- **Black level** per Bayer channel (should be uniform across R/Gr/Gb/B and
  match the manufacturer's declared pedestal; measured 200 DN on the test
  sensor).
- **DSNU** (dark-signal non-uniformity): spatial standard deviation of the
  time-averaged dark frame after removing a low-order shading term. Quote in DN
  and electrons (measured 0.13 DN / 1.4 e-).

Do this first: a clean, flat black is the foundation every other measurement
rests on, and a wrong pedestal (e.g. light leak) corrupts everything downstream.

### 2. Photon transfer curve (PTC): conversion gain and read noise

On the flat field, sweep exposure from near-dark to saturation at fixed gain.
At each point compute the mean signal above black and the temporal variance
(per-pixel, across the burst, mean over the ROI), single green channel.

- Plot temporal variance vs mean signal. For a shot-noise-limited sensor this
  is a straight line: **slope = 1 / conversion gain** (DN per e-, so
  conversion gain K = 1/slope in e-/DN), **intercept = read noise squared**.
- Exclude clipped points: as pixels saturate the variance collapses towards
  zero and drags the fit down.
- Cross-check the intercept-derived read noise against the direct measurement
  from dark frames (temporal standard deviation of darks, converted to
  electrons via K). Measured: 10.4 e-/DN conversion gain, 9.8 e- read noise
  at unity gain, agreeing with the PTC intercept (~1 DN).
- **SNR curve**: from the same sweep, plot SNR = mean/temporal-sigma vs signal.
  It should follow the square-root (shot-noise) law up to a peak near full well
  (~46 dB measured).

### 3. Linearity, full well and dynamic range

Same exposure sweep on the flat field, central ROI:

- **Linearity**: straight-line fit of mean signal vs exposure time; quote R^2
  (measured 1.000). Verify the *applied* exposure from frame metadata at every
  point — the request is not always what the sensor did.
- **Full well**: the hard saturation knee in DN above black (measured 3895 DN),
  converted to electrons via the conversion gain (40.4 ke-). Saturation should
  be a hard knee; a gradual shoulder usually indicates a measurement problem
  (non-uniform scene, ambient light), not the sensor.
- **Dynamic range**: 20*log10(full well / read noise), both in electrons
  (measured 72 dB at gain 1).

### 4. Analogue gain sweep

Repeat the dark-noise and signal measurements across the gain range
(e.g. 1, 2, 4, 8, 16):

- Mean signal should scale linearly with gain.
- Read noise in *DN* rises with gain (measured 0.95 -> 6.6 DN over 1--16x), but
  input-referred read noise in *electrons* should fall — the signature of a true
  analogue multiplier, and what makes gain genuinely useful in low light.
- The PTC slope at each gain gives the effective conversion gain, which should
  scale as 1/gain; this is also the cross-check that reported gain is real
  (see pitfalls — gain labels can lie).

### 5. Colour response: white balance and CCM

ColorChecker Classic under several illuminants spanning the CCT range of
interest (five illuminants, 2776--5650 K measured, on the test run). For each
illuminant:

- Measure the true illuminant CCT and lux with an independent meter; the
  lightbox's labelled daylight channels measured meaningfully warmer than their
  CIE nominals (D65 channel: 5650 K measured vs 6500 K label), so always plot
  against the measured value.
- Locate the 24 patches in the raw frame. With a small, rotated or
  lens-distorted chart, interpolate the patch grid from the four corner patches
  rather than assuming an axis-aligned layout.
- **White balance**: grey-point gains from the neutral patches (make a neutral
  read R = G = B). Gains should track measured CCT monotonically.
- **CCM**: least-squares 3x3 matrix mapping white-balanced raw to linear sRGB
  reference values, fitted per illuminant. Quote mean CIE Lab Delta E before
  (WB only) and after the CCM. Measured: Delta E ~20 -> ~6.5; a production
  tuning with a flat-on, well-framed chart typically reaches 2--3, so treat
  the residual as chart/model-limited, not sensor-limited.
- Note the fitted matrix absorbs an overall scale (row sums well below 1 are
  normal for a least-squares fit); a downstream white-preserving normalisation
  restores it.

### 6. Photo-response non-uniformity (PRNU)

Flat field at mid-signal, time-average a burst, remove the low-order
lens-shading gradient (polynomial fit), then quote the residual spatial
standard deviation as a percentage of mean signal (measured 0.48%). Time
averaging is essential: it suppresses the temporal noise so only the fixed
pattern remains.

### 7. Dual conversion gain (if the sensor supports it)

For a DCG pixel (switchable sense-node capacitance: LCG default, HCG option),
a **three-arm experiment** separates the conversion-gain step from ordinary
amplification. All arms use the identical PTC analysis:

1. **LCG at minimum gain** — the baseline (reproduces experiment 2).
2. **LCG at the HCG-matched gain** — because HCG typically raises the minimum
   analogue gain (to ~2.3x on the test sensor), capture an LCG arm at that same
   gain.
3. **HCG at the same gain** — the difference between arms 2 and 3 is then the
   conversion-gain effect alone.

Per arm, report conversion gain, read noise (e-), full well (e-) and dynamic
range from the PTC. Expected signatures on the test sensor: arm 1 -> arm 2
conversion gain scales exactly by the applied analogue gain (9.69 -> 4.30
e-/DN at 2.25x); arm 2 -> arm 3 shows the sense-node switch (4.30 -> 0.52
e-/DN, an 8.2x step at matched gain), with read noise falling from 6.81 to
1.15 e- but full well collapsing ~18x (to ~2000 e-), so HCG's dynamic range
is *lower* than LCG at minimum gain. HCG buys a lower noise floor for shadows,
not headroom — which is why HDR modes read both gains and fuse them.

Two DCG-specific cautions: the reported analogue gain may be wrong in HCG mode
(a requested 1x was really 2.3x, with 1--2.3x an unusable dead zone), so trust
the PTC-measured conversion gain, never the label; and all arms clip at the
same DN ceiling, which corresponds to very different electron full wells.

## Pitfalls (all encountered, all material)

- **Ambient flicker.** Fluorescent room lighting corrupted an early flat-field
  run: darks were not dark (green pedestal ~1220 DN) and 100 Hz flicker
  inflated the temporal variance, giving nonsense read noise. Enclose the
  sensor and flat field against ambient; verify darks read the expected
  pedestal before anything else.
- **Non-uniform scenes confound photometry.** A cluttered scene makes the
  whole-frame mean roll off gradually as bright regions saturate first,
  mimicking non-linearity. Use a flat field and a central ROI.
- **Trust readbacks, not requests.** An apparent exposure clamp turned out not
  to exist — the applied exposure (from metadata) followed the request exactly
  and the roll-off was a scene artefact. Always log the metadata-reported
  exposure/gain alongside each capture.
- **Two-green averaging** halves variance and doubles apparent conversion
  gain; use a single green channel throughout.
- **Illuminant labels lie.** Measure CCT with an independent reference; plot
  against measurements, not channel names.
- **Payload size.** Multi-frame raw bursts are large; a centred ROI keeps them
  manageable without hurting any of the above measurements.
