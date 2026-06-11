# Raspberry Pi Camera Tuning Tool (CTT)

Camera Tuning Tool for generating JSON tuning files for Raspberry Pi cameras.
Takes DNG calibration images and produces tuning parameters for the PiSP or VC4
ISP platforms.

## Installation

Requires Python 3.11+.

```bash
pip install rpi-ctt
```

For development, install in editable mode from the repository:

```bash
pip install -e .
```

## Usage

### Full calibration

```bash
python3 -m ctt -i <image_dir> [-o <output_dir>] [--name <name>] [-t pisp] [-t vc4] [-c config.json]
```

All outputs (JSON tuning file, log file, Macbeth chart PNGs) are written to the
output directory. If `-o` is not specified, the current directory is used.

Output filenames are `<name>_<target>.json` and `<name>_<target>.log`. If
`--name` is not specified, the name is derived from the input directory. You
can pass `-t` multiple times or use a comma-separated list (e.g. `-t pisp,vc4`).
If no `-t` is given, both targets are run (e.g. `imx219_pisp.json` and
`imx219_vc4.json`).

### ALSC-only calibration

```bash
python3 -m ctt --alsc-only -i <image_dir> [-o <output_dir>] [-t pisp] [-t vc4]
```

### Colour-only calibration (AWB + CCM)

```bash
python3 -m ctt --colour-only -i <image_dir> [-o <output_dir>] [-t pisp] [-t vc4]
```

### Black-level-only calibration

Measures the sensor black level from dark frames (`dark_<n>.dng`, captured with
the lens cap on / no light) and writes it to `rpi.black_level`:

```bash
python3 -m ctt --blacklevel-only -i <image_dir> [-o <output_dir>] [-t pisp] [-t vc4]
```

### Update an existing tuning file

Re-run calibrations using an existing file as a template. The target (pisp or vc4) is read from the tuning file; do not use `-t` with `--update`. The file is updated **in place**.

```bash
python3 -m ctt -i <image_dir> --update <existing.json>

python3 -m ctt -i <image_dir> --update <existing.json> -o <output_dir> --name imx219
```

With `--alsc-only`, `--colour-only` or `--blacklevel-only`, only that section is re-calibrated; all other algorithm blocks in the file are left unchanged.

```bash
python3 -m ctt --alsc-only -i <image_dir> --update <existing.json>
```

### Use a custom template

```bash
python3 -m ctt -i <image_dir> -o <output_dir> -t pisp --template my_base.json
```

### Convert between VC4 and PiSP

Interpolates ALSC grids, swaps denoise/AGC/HDR blocks:

```bash
python3 -m ctt --convert -t pisp input_vc4.json output_pisp.json
python3 -m ctt --convert -t vc4 input_pisp.json                    # in-place
```

### Prettify a tuning file

```bash
python3 -m ctt --prettify [-t pisp|vc4] <input.json> [output.json]
```

## Options

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Calibration image directory |
| `-o`, `--output` | Output directory (default: current directory) |
| `--name` | Base name for output files (default: derived from input directory) |
| `-t`, `--target` | Target platform(s): repeat for multiple (e.g. `-t pisp -t vc4`) or use `-t pisp,vc4`. Default: both. |
| `-c`, `--config` | Configuration file (see below) |
| `--template` | Custom template JSON file |
| `--update` | Existing tuning file to update in place (target taken from file; cannot use `-t`) |
| `--plot` | Show matplotlib debug plot for algorithm (e.g. `awb`, `alsc`, `geq`, `noise`). Can be repeated or comma-separated. |
| `--alsc-only` | Run only ALSC (lens shading) calibration |
| `--colour-only` | Run only AWB and CCM calibrations |
| `--blacklevel-only` | Run only the black level measurement (requires dark frames) |
| `--convert` | Convert tuning file between VC4 and PiSP |
| `--prettify` | Prettify an existing tuning file |

## Configuration file

An optional JSON config file controls calibration behaviour. See
`apps/ctt/ctt/data/config_example.json` for the full set of options:

```json
{
    "disable": [],
    "plot": [],
    "alsc": {
        "do_alsc_colour": 1,
        "luminance_strength": 0.8,
        "max_gain": 8.0
    },
    "awb": {
        "greyworld": 0
    },
    "blacklevel": -1,
    "macbeth": {
        "small": 0,
        "show": 0
    }
}
```

- **disable** - List of algorithm keys to skip (e.g. `["rpi.noise"]`)
- **plot** - List of algorithm names for matplotlib debug plots. Use short names (`awb`, `alsc`, `geq`, `noise`) or full keys (`rpi.awb`, etc.). Supported: **rpi.awb** (colour curve hatspace + dehatspace), **rpi.alsc** (3D vignetting surfaces), **rpi.geq** (fit and scaled fit), **rpi.noise** (per-image noise fit). Can also be set via `--plot` on the command line.
- **alsc.do_alsc_colour** - Enable colour shading calibration (default: 1)
- **alsc.luminance_strength** - Luminance correction strength, 0.0-1.0 (default: 0.8)
- **alsc.max_gain** - Maximum lens shading gain (default: 8.0)
- **awb.greyworld** - Use grey world AWB instead of Macbeth-based (default: 0)
- **blacklevel** - Override black level; -1 to auto-detect (default: -1)
- **macbeth.small** - Use small Macbeth chart detection (default: 0)
- **macbeth.show** - Display detected Macbeth chart (default: 0)

## Calibration images

The tool looks only in the **root** of the input directory and uses **`.dng` files only** (no subdirectories, no JPEGs).

- **ALSC images**: Filenames containing `alsc` and a colour temperature (e.g. `alsc_3000k_0.dng`). Uniform flat-field images at multiple colour temperatures.
- **Macbeth images**: Filenames must encode colour temperature and lux in the form `...<temp>k_<lux>l.dng` (e.g. `d65_5858k_1344l.dng`). Images of a Macbeth ColorChecker chart for AWB, CCM, noise, lux, and GEQ.
- **Dark frames**: Filenames containing `dark` (e.g. `dark_0.dng`), no other tags needed. Zero-light captures (lens cap on) used to measure the sensor black level; the measured value replaces the DNG metadata value throughout the run. Capture several at different shutter/gain settings to check black level stability against exposure.

If a file is skipped (e.g. missing colour temp/lux in the filename, or Macbeth chart not found in the image), the tool prints a short message (e.g. colour temp/lux not in filename, or Macbeth not found) with the filename.

## Calibrations performed

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Black level | `rpi.black_level` | Sensor black level measured from dark frames |
| ALSC | `rpi.alsc` | Lens shading correction (colour and luminance) |
| AWB | `rpi.awb` | Auto white balance calibration |
| CCM | `rpi.ccm` | Colour correction matrices per illuminant |
| CAC | `rpi.cac` | Chromatic aberration correction (PiSP only) |
| Noise | `rpi.noise` | Noise profile characterisation |
| Lux | `rpi.lux` | Lux level calibration |
| GEQ | `rpi.geq` | Green equalisation threshold |

## Web frontend (ctt-server)

`ctt-server` is an optional web UI for capturing, tagging and tuning calibration
images, served from the Raspberry Pi. It runs as a single process on the Pi: the
server previews and captures DNGs in-process with Picamera2, files them with
CTT-correct filenames, runs the tuner in-process, and serves downloadable tuning
files with result visualisations. The client is just a web browser on any machine
on the network.

### Install and run (on the Pi)

```bash
pip install "rpi-ctt[server]"    # from PyPI
# or, from a local checkout:
pip install -e ".[server]"
ctt-server                       # HTTPS on 0.0.0.0:5000
```

`ctt-server` is **HTTPS-only**; on first run it generates a self-signed
certificate under `<workspace>/.tls` (pass `--cert`/`--key` to use your own, or
`--port` to change the port). Browse to `https://<pi-hostname>:5000` and accept
the one-time self-signed warning. Picamera2 ships with Raspberry Pi OS and is
imported lazily; if you use a virtualenv, create it with `--system-site-packages`
(or `apt install python3-picamera2`) so picamera2 is visible.

### Workflow

1. **Project** — create one per sensor (e.g. `imx708_wide`); the name becomes the
   output filename base (`imx708_wide_pisp.json`).
2. **Capture** — frame with the live preview and histogram, set exposure/gain (or
   leave on auto), and capture. In Macbeth mode a live finder overlays the detected
   chart and flags low confidence or a too-small chart. Each shot is filed with a
   CTT-correct filename.
3. **Run** — pick targets (PiSP/VC4/both) and mode (full / ALSC-only / colour-only);
   CTT progress streams live in the console.
4. **Results** — download the tuning `.json`/`.log` or the whole project as a zip,
   and inspect the AWB curve, per-CT CCM matrices, ALSC shading heatmap and
   lux/noise references parsed from the output JSON.

When a supported lightbox is attached (see below), the capture page shows a
**Lightbox** control to pick the illuminant and set intensity; it is hidden when no
device is present.

## Lightbox control

`ctt.devices` provides a small, generic API for controlling an illumination lightbox
over USB from the Pi, so a calibration run can set repeatable illuminants
programmatically. Consumers use the abstract `Lightbox` interface and the
device-agnostic factory; concrete drivers (currently **Image Engineering
lightSTUDIO-S**) plug in behind it. A new lightbox is added as a driver package under
`apps/ctt/ctt/devices/<model>/` registered in `apps/ctt/ctt/devices/registry.py` — no
change to the generic API or its consumers.

### Install (on the Pi)

```bash
pip install "rpi-ctt[devices]"            # from PyPI (or "rpi-ctt[server,devices]" with the web UI)
# or, from a local checkout:
pip install -e ".[devices]"               # or ".[server,devices]" with the web UI
sudo apt install libusb-1.0-0             # pyusb's backend

# allow non-root USB access (lightSTUDIO-S)
sudo cp apps/ctt/ctt/devices/lightstudio_s/contrib/99-lightstudio.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

If the udev rule is missing but pyusb is installed and the device is plugged in, the
probe gets far enough to fail with `usb.core.USBError: [Errno 13] Access denied`
(rather than reporting absent) — install the rule above. If `udevadm trigger` doesn't
re-apply to the already-enumerated device, unplug and replug the lightbox so the rule
runs on re-enumeration. The user running the server must be in the `plugdev` group.

### Use

```python
from ctt.devices import get_lightbox

with get_lightbox() as box:               # first attached, supported lightbox
    box.set_illuminant('D65')             # name → channel, at its default intensity
    box.set_intensity(4, 50)              # channel 4 (D65) at 50 %
    print(box.info())
    box.off()
```

```bash
ctt-lightbox probe                        # find the box, list illuminants
ctt-lightbox status
ctt-lightbox set 1 50                     # channel 1 (F12) → 50 %
ctt-lightbox illuminant D65               # switch to D65 at its default
ctt-lightbox off
```

### lightSTUDIO-S channels

| Ch | Illuminant | Default % |    | Ch | Illuminant | Default % |
|----|------------|-----------|----|----|------------|-----------|
| 1  | F12        | 100       |    | 5  | Halogen (10 lux)  | 2   |
| 2  | F11        | 100       |    | 6  | Halogen (100 lux) | 25  |
| 3  | D50        | 100       |    | 7  | Halogen (400 lux) | 100 |
| 4  | D65        | 100       |    | 8  | Halogen + blue filter (400 lux) | 100 |

## Development

### Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint errors
python3 -m ruff check .

# Auto-fix lint errors
python3 -m ruff check --fix .

# Format code
python3 -m ruff format .

# Check formatting without modifying files
python3 -m ruff format --check .
```

These same checks run in CI. To catch problems before committing, enable the
[pre-commit](https://pre-commit.com/) hooks (they run ruff lint + format on each
commit, mirroring CI):

```bash
pip install -e ".[dev]"
pre-commit install            # one-time, per clone
pre-commit run --all-files    # optional: check the whole tree now
```

### Running tests

Install with the test extra and run pytest:

```bash
pip install -e ".[test]"
pytest -v
```

### Building a wheel package

```bash
pip install build
python3 -m build
```

This produces a `.whl` file in the `dist/` directory.

## License

BSD-2-Clause - Copyright (C) 2026, Raspberry Pi
