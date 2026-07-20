# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the generic light-meter API (devices.lightmeter). Hardware-free: a
# fake driver exercises the base class, result dataclass and context manager.

from devices import LightMeter, Measurement, Spectrum


class FakeMeter(LightMeter):
    """A minimal in-memory light meter (no hardware)."""

    def __init__(self):
        self.closed = False
        self._latest = None

    @classmethod
    def probe(cls, serial=None):
        return cls()

    @property
    def model(self) -> str:
        return 'FakeMeter'

    @property
    def serial(self) -> str | None:
        return 'FM-1'

    def measure(self) -> Measurement:
        self._latest = Measurement(illuminance_lux=123.4, cct=6500.0, duv=0.0012)
        return self._latest

    def read_latest(self):
        return self._latest

    def close(self) -> None:
        self.closed = True


def test_measurement_to_dict_drops_none():
    m = Measurement(illuminance_lux=100.0, cct=5000.0, cie1931_xy=(0.3457, 0.3585))
    d = m.to_dict()
    assert d['illuminance_lux'] == 100.0
    assert d['cct'] == 5000.0
    assert d['cie1931_xy'] == (0.3457, 0.3585)
    # Unset optional fields are omitted entirely.
    assert 'duv' not in d
    assert 'cri_ra' not in d
    assert 'spectrum_1nm' not in d


def test_measurement_carries_rich_fields():
    spec = Spectrum(start_nm=380.0, step_nm=5.0, values=(0.1, 0.2, 0.3))
    m = Measurement(illuminance_lux=1.0, cct=3000.0, cri_ri=(95.0,) * 15, spectrum_5nm=spec)
    assert m.spectrum_5nm is spec
    d = m.to_dict()
    assert d['cri_ri'] == (95.0,) * 15
    # Nested Spectrum is flattened to a dict so the result is JSON-serialisable.
    assert d['spectrum_5nm'] == {'start_nm': 380.0, 'step_nm': 5.0, 'values': (0.1, 0.2, 0.3)}


def test_info_snapshot():
    assert FakeMeter().info() == {'model': 'FakeMeter', 'serial': 'FM-1'}


def test_default_read_latest_is_none():
    class Bare(FakeMeter):
        pass  # uses FakeMeter.read_latest

    class Minimal(LightMeter):
        @classmethod
        def probe(cls, serial=None):
            return cls()

        def measure(self):
            return Measurement(illuminance_lux=0.0, cct=0.0)

        def close(self):
            pass

    # The base class default returns None when a driver does not override read_latest.
    assert Minimal().read_latest() is None


def test_context_manager_closes():
    meter = FakeMeter()
    with meter as m:
        assert m is meter
        assert m.measure().illuminance_lux == 123.4
        assert m.read_latest() is not None
    assert meter.closed is True
