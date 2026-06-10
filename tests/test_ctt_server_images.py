# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt_server Images tab.

from ctt_server import sessions


def _client_with_project(tmp_path):
    from ctt_server.app import create_app

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    proj.add_capture(b'DNG', 'alsc', 5000, jpeg_bytes=b'JPGDATA')
    proj.add_capture(b'DNG', 'macbeth', 5858, lux=1344, label='d65')
    return create_app(str(tmp_path)).test_client(), proj


def test_images_page_renders(tmp_path):
    client, _ = _client_with_project(tmp_path)
    r = client.get('/projects/cam/images')
    assert r.status_code == 200
    assert b'imagesApp' in r.data
    assert b'alsc_5000k_0.dng' in r.data
    assert b'd65_5858k_1344l.dng' in r.data


def test_images_page_unknown_project_404(tmp_path):
    client, _ = _client_with_project(tmp_path)
    assert client.get('/projects/nope/images').status_code == 404


def test_capture_page_no_longer_lists_images(tmp_path):
    # The captured-images grid moved to the Images tab; the Capture page keeps
    # the coverage checklist (which still needs the captures data) and links over.
    client, _ = _client_with_project(tmp_path)
    r = client.get('/projects/cam')
    assert r.status_code == 200
    assert b'img-grid' not in r.data  # the image grid lives on the Images tab now
    assert b'Upload images' not in r.data  # upload moved with it
    assert b'/projects/cam/images' in r.data


def test_capture_jpeg_thumb_param(tmp_path):
    # ?thumb=1 must not break the saved-JPEG path (served verbatim).
    client, _ = _client_with_project(tmp_path)
    r = client.get('/projects/cam/captures/alsc_5000k_0.dng/jpeg?thumb=1')
    assert r.status_code == 200
    assert r.data == b'JPGDATA'


# --- explicit preview sources (?source=) ---


def test_capture_jpeg_source_jpeg_only(tmp_path):
    from ctt_server.app import _RAWPY

    client, proj = _client_with_project(tmp_path)
    # Saved JPEG present: served verbatim.
    r = client.get('/projects/cam/captures/alsc_5000k_0.dng/jpeg?source=jpeg')
    assert r.status_code == 200
    assert r.data == b'JPGDATA'
    # DNG-only capture: source=jpeg must 404 rather than fall back to rawpy.
    proj.import_capture('alsc_3000k_0.dng', b'NOTAREALDNG')
    assert client.get('/projects/cam/captures/alsc_3000k_0.dng/jpeg?source=jpeg').status_code == 404
    # ...whereas the default (auto) still falls back (500 on junk bytes with rawpy).
    r = client.get('/projects/cam/captures/alsc_3000k_0.dng/jpeg')
    assert r.status_code == (500 if _RAWPY else 404)


def test_capture_jpeg_source_raw_skips_saved_jpeg(tmp_path):
    from ctt_server.app import _RAWPY

    client, _ = _client_with_project(tmp_path)
    # The capture has a saved JPEG, but source=raw must develop the DNG instead:
    # the junk DNG bytes give 500 with rawpy (and 404 without it) — never b'JPGDATA'.
    r = client.get('/projects/cam/captures/alsc_5000k_0.dng/jpeg?source=raw')
    assert r.status_code == (500 if _RAWPY else 404)


# --- EXIF endpoint ---


def test_capture_exif_missing_file_404(tmp_path):
    client, _ = _client_with_project(tmp_path)
    assert client.get('/projects/cam/captures/nope_5000k_0.dng/exif').status_code == 404
    assert client.get('/projects/cam/captures/foo.txt/exif').status_code == 404


def test_dng_exif_summary_extraction(tmp_path, monkeypatch):
    # No real DNG fixtures exist, so unit-test the curated extraction against a
    # fake exifread tag dict, covering the IFD-prefix tolerance (rpicam puts
    # geometry in 'EXIF SubIFD0', PiDNG in 'Image').
    import ctt_server.app as app_mod

    class Tag:
        def __init__(self, values, text=None):
            self.values = values
            self._text = text if text is not None else str(values)

        def __str__(self):
            return self._text

    class Ratio:
        num, den = 1, 100  # 10 ms

    tags = {
        'Image Model': Tag(['imx708'], 'imx708'),
        'EXIF SubIFD0 ImageWidth': Tag([4608]),
        'EXIF SubIFD0 ImageLength': Tag([2592]),
        'EXIF ExposureTime': Tag([Ratio()]),
        'EXIF ISOSpeedRatings': Tag([400]),
        'EXIF SubIFD0 Tag 0xC61D': Tag([4095]),
        'JPEGThumbnail': b'\xff\xd8...',  # raw bytes: must be excluded from the dump
    }
    monkeypatch.setattr(app_mod.exifread, 'process_file', lambda f, details=False: tags)
    dng = tmp_path / 'x.dng'
    dng.write_bytes(b'X')

    out = app_mod._dng_exif(dng)
    summary = {row['label']: row['value'] for row in out['summary']}
    assert summary['Camera model'] == 'imx708'
    assert summary['Width'] == '4608'
    assert summary['Height'] == '2592'
    assert summary['Exposure time'] == '10.00 ms'
    assert summary['ISO'] == '400'
    assert summary['White level'] == '4095'
    assert 'Make' not in summary  # missing tags are skipped, not errors
    assert all(t['key'] != 'JPEGThumbnail' for t in out['tags'])


def test_capture_exif_route(tmp_path, monkeypatch):
    import ctt_server.app as app_mod

    client, _ = _client_with_project(tmp_path)
    monkeypatch.setattr(app_mod, '_dng_exif', lambda path: {'summary': [], 'tags': []})
    r = client.get('/projects/cam/captures/alsc_5000k_0.dng/exif')
    assert r.status_code == 200
    body = r.get_json()
    assert body['filename'] == 'alsc_5000k_0.dng'
    assert body['summary'] == [] and body['tags'] == []
