# Device control

`ctt.devices` provides a small, generic API for controlling an illumination lightbox
over USB from the Pi, so a calibration run can set repeatable illuminants
programmatically. Consumers use the abstract `Lightbox` interface and the
device-agnostic factory; concrete drivers (currently **Image Engineering
lightSTUDIO-S**) plug in behind it. A new lightbox is added as a driver package under
`apps/ctt/ctt/devices/<model>/` registered in `apps/ctt/ctt/devices/registry.py` — no
change to the generic API or its consumers.

## Install (on the Pi)

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

## Use

Every method that takes an illuminant accepts its name (case-insensitive — the
short name, the descriptive label or a driver alias), a channel number, or a
member of the driver's `Illuminant` enum. Note that only one channel is lit at
a time, and setting an intensity is what switches channels — intensities cannot
be staged on an inactive channel (the only stored per-channel value is the
device's power-on default).

```python
from ctt.devices import get_lightbox
from ctt.devices.lightstudio_s import Illuminant

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
| 6  | `Halogen100` | Halogen (100 lux)               | 25  |
| 7  | `Halogen400` | Halogen (400 lux)               | 100 |
| 8  | `HalogenBF`  | Halogen + blue filter (400 lux) | 100 |
