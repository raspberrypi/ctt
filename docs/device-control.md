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

## lightSTUDIO-S channels

| Ch | Illuminant | Default % |    | Ch | Illuminant | Default % |
|----|------------|-----------|----|----|------------|-----------|
| 1  | F12        | 100       |    | 5  | Halogen (10 lux)  | 2   |
| 2  | F11        | 100       |    | 6  | Halogen (100 lux) | 25  |
| 3  | D50        | 100       |    | 7  | Halogen (400 lux) | 100 |
| 4  | D65        | 100       |    | 8  | Halogen + blue filter (400 lux) | 100 |
