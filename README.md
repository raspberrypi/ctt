# Raspberry Pi Camera Tuning Tool (CTT)

Camera Tuning Tool for generating JSON tuning files for Raspberry Pi cameras.
Takes DNG calibration images and produces tuning parameters for the PiSP or VC4
ISP platforms.

## Dependencies

Requires Python 3.11+. Install the package and its dependencies with:

```bash
pip install -e .
```

This installs ctt in editable mode with all required packages (see `pyproject.toml`).

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

### Update an existing tuning file

Re-run calibrations using an existing file as a template. The target (pisp or vc4) is read from the tuning file; do not use `-t` with `--update`. The file is updated **in place**.

```bash
python3 -m ctt -i <image_dir> --update <existing.json>

python3 -m ctt -i <image_dir> --update <existing.json> -o <output_dir> --name imx219
```

With `--alsc-only` or `--colour-only`, only that section is re-calibrated; all other algorithm blocks in the file are left unchanged.

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
| `--convert` | Convert tuning file between VC4 and PiSP |
| `--prettify` | Prettify an existing tuning file |

## Configuration file

An optional JSON config file controls calibration behaviour. See
`ctt/data/config_example.json` for the full set of options:

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

If a file is skipped (e.g. missing colour temp/lux in the filename, or Macbeth chart not found in the image), the tool prints a short message (e.g. colour temp/lux not in filename, or Macbeth not found) with the filename.

## Calibrations performed

| Algorithm | Key | Description |
|-----------|-----|-------------|
| ALSC | `rpi.alsc` | Lens shading correction (colour and luminance) |
| AWB | `rpi.awb` | Auto white balance calibration |
| CCM | `rpi.ccm` | Colour correction matrices per illuminant |
| CAC | `rpi.cac` | Chromatic aberration correction (PiSP only) |
| Noise | `rpi.noise` | Noise profile characterisation |
| Lux | `rpi.lux` | Lux level calibration |
| GEQ | `rpi.geq` | Green equalisation threshold |

## Package structure

```
ctt/
    __main__.py          Entry point (python -m ctt)
    ctt.py               CLI + orchestration
    core/
        camera.py        Camera data holder + image management
        image.py         Image dataclass
        image_loader.py  DNG image loading
    algorithms/
        base.py          CalibrationAlgorithm ABC
        alsc.py          Lens shading correction
        awb.py           Auto white balance
        ccm.py           Colour correction matrices
        cac.py           Chromatic aberration correction
        noise.py         Noise profiling
        lux.py           Lux calibration
        geq.py           Green equalisation
    detection/
        macbeth.py       Macbeth chart locator
        ransac.py        RANSAC geometry helpers
        dots.py          CAC dot detection
        patches.py       ALSC patch extraction
    platforms/
        base.py          PlatformConfig class
        pisp.py          PiSP configuration
        vc4.py           VC4 configuration
    output/
        json_formatter.py  JSON pretty-printer
        converter.py       VC4 <-> PiSP conversion
    utils/
        tools.py         Shared utilities
        colorspace.py    RGB to LAB conversion
        config.py        Config file parser
        errors.py        Error classes
    data/
        ctt_ref.pgm      Reference Macbeth chart image
        pisp_template.json  Default PiSP tuning template
        vc4_template.json   Default VC4 tuning template
        config_example.json Example configuration file
```

## Development

### Install in editable mode

```bash
pip install -e .
```

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

### Building a wheel package

```bash
pip install build
python3 -m build
```

This produces a `.whl` file in the `dist/` directory.

## License

BSD-2-Clause - Copyright (C) 2026, Raspberry Pi
