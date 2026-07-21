# Installation

Requires Python 3.11+.

## Debian package dependencies

On Debian/Raspberry Pi OS, the following apt packages are needed alongside the
pip install:

| Package | Needed by | Notes |
|---------|-----------|-------|
| `libgl1`, `libglib2.0-0` | Core CLI | Runtime libraries for the `opencv-python` wheel. Already present on desktop images; usually only missing on Lite. |
| `python3-picamera2` | [CTT server](ctt-server.md) | Camera capture. Ships with Raspberry Pi OS; pulls in the libcamera Python bindings (`python3-libcamera`), which are not available from PyPI. |
| `libusb-1.0-0` | [Device control](device-control.md) | The libusb backend used by pyusb. |

```bash
sudo apt install libgl1 libglib2.0-0     # core CLI (opencv runtime)
sudo apt install python3-picamera2      # CTT server
sudo apt install libusb-1.0-0           # device control
```

## From PyPI

```bash
pip install rpi-ctt
```

Optional extras enable the additional components:

| Extra | Installs | Use for |
|-------|----------|---------|
| `rpi-ctt[server]` | Flask | The [CTT server](ctt-server.md) web UI (run on the Pi) |
| `rpi-ctt[devices]` | pyusb | [USB lightbox and light-meter control](device-control.md) |
| `rpi-ctt[server,devices]` | both | Web UI with lightbox and light-meter support |

## From a local checkout

For development, install in editable mode from the repository root:

```bash
pip install -e .                    # core CLI only
pip install -e ".[server]"          # with the web UI
pip install -e ".[server,devices]"  # with the web UI and lightbox/light-meter control
```

See [Developers](developers.md) for the `dev` and `test` extras (pre-commit
hooks and pytest).

## Notes

- **Picamera2** (used by the CTT server for capture) ships with Raspberry Pi OS
  and is imported lazily, so it is intentionally not a pip dependency. If you
  use a virtualenv, create it with `--system-site-packages` (or
  `apt install python3-picamera2`) so picamera2 is visible.
- **Lightbox and light-meter control** additionally need the libusb backend and a
  udev rule — see [Device control](device-control.md#install-on-the-pi).
