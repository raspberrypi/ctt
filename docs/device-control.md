# Device control

The top-level `devices` package provides small, generic APIs for two kinds of USB
device on the Pi: a controllable illumination **lightbox** (so a calibration run can
set repeatable illuminants programmatically) and a **light meter** (so it can read back
illuminance and colour temperature). Consumers use the abstract interfaces and the
device-agnostic factories; concrete drivers plug in behind them — currently **Image
Engineering lightSTUDIO-S** (lightbox) and **Konica Minolta CL-70F** (light meter). A
new device is added as a driver package under `devices/<model>/` registered in
`devices/registry.py` — no change to the generic API or its consumers.

## Install (on the Pi)

```bash
pip install "rpi-ctt[devices]"            # from PyPI (or "rpi-ctt[server,devices]" with the web UI)
# or, from a local checkout:
pip install -e ".[devices]"               # or ".[server,devices]" with the web UI
sudo apt install libusb-1.0-0             # pyusb's backend

# allow non-root USB access (one rule per device)
sudo cp devices/lightstudio_s/contrib/99-lightstudio.rules /etc/udev/rules.d/
sudo cp devices/cl70f/contrib/99-cl70f.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

If the udev rule is missing but pyusb is installed and the device is plugged in, the
probe gets far enough to fail with `usb.core.USBError: [Errno 13] Access denied`
(rather than reporting absent) — install the rule above. If `udevadm trigger` doesn't
re-apply to the already-enumerated device, unplug and replug the lightbox so the rule
runs on re-enumeration. The user running the server must be in the `plugdev` group.

## Use

Every method that takes an illuminant accepts its name (case-insensitive — the
short name, the descriptive label or a driver alias), a channel number, or a
member of the driver's `Illuminant` enum. Note that only one channel is lit at
a time, and setting an intensity is what switches channels — intensities cannot
be staged on an inactive channel (the only stored per-channel value is the
device's power-on default).

```python
from devices import get_lightbox
from devices.lightstudio_s import Illuminant

with get_lightbox() as box:               # first attached, supported lightbox
    box.set_illuminant('D65')             # switch to D65 at its default intensity
    box.set_illuminant('D65', 50)         # ... at 50 %
    box.set_illuminant(Illuminant.HalogenBF, 80)  # typo-safe enum (a plain str)
    box.set_illuminant(4, 50)             # channel numbers work too

    state = box.get_state()               # LightboxState(channel, illuminant, intensity)
    print(f'{state.illuminant} at {state.intensity:.0f}%')
    box.get_default_intensity('D65')      # a channel's power-on default (100.0)
    print(box.info())                     # full snapshot: identity, illuminant maps, state

    box.off()
```

```bash
ctt-lightbox probe                        # find the box, list illuminants
ctt-lightbox status
ctt-lightbox set F12 50                   # F12 (channel 1) → 50 %
ctt-lightbox set 1 50                     # same, by channel number
ctt-lightbox illuminant D65               # switch to D65 at its default
ctt-lightbox off
```

## lightSTUDIO-S channels

| Ch | Name | Description | Default % |
|----|------|-------------|-----------|
| 1  | `F12`        | F12 fluorescent                 | 100 |
| 2  | `F11`        | F11 fluorescent                 | 100 |
| 3  | `D50`        | D50 daylight                    | 100 |
| 4  | `D65`        | D65 daylight                    | 100 |
| 5  | `Halogen10`  | Halogen (10 lux)                | 2   |
| 6  | `Halogen100` | Halogen (100 lux)               | 17  |
| 7  | `Halogen400` | Halogen (400 lux)               | 67  |
| 8  | `HalogenBF`  | Halogen + blue filter (400 lux) | 99  |

The default % values are the device's stored power-on levels, reported per channel by
`Lightbox.info()` as `illuminant_defaults` (and used to prefill the web auto-capture
dialog); they may vary slightly between units and calibrations.

## Light meter

A light meter measures rather than controls: each `measure()` returns a `Measurement`
with illuminance (lux) and correlated colour temperature always present, plus optional
colorimetry (Δuv, CIE 1931 x/y, CIE 1976 u′v′, CIE 1960 uv), foot-candles, tristimulus
XYZ, dominant wavelength, excitation purity, CRI (Ra and R1–R15) and the 380–780 nm
spectrum where the device supplies them. `read_latest()` returns the device's last
reading without triggering a new one.

```python
from devices import get_lightmeter

with get_lightmeter() as meter:            # first attached, supported meter
    reading = meter.measure()               # trigger and read one measurement
    print(reading.illuminance_lux, reading.cct)
    print(reading.to_dict())                # JSON-friendly, omitting unset fields
    meter.read_latest()                     # last stored reading, or None
```

```bash
ctt-lightmeter probe                        # find the meter, show identity
ctt-lightmeter measure                      # one reading (--json for the full dump)
ctt-lightmeter sample --interval 5          # a reading every 5 s until interrupted
ctt-lightmeter sample --interval 5 --count 10
ctt-lightmeter calibrate                    # run a dark calibration
```

Non-root USB access to the CL-70F needs its udev rule (installed in the
[Install](#install-on-the-pi) step above; see the file for details).
