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
