# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# `ctt-server` entry point: launch the Flask web UI.

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog='ctt-server', description='ctt-server web UI')
    # Default to all interfaces: the Pi is the server and the browser is usually remote.
    parser.add_argument('--host', default='0.0.0.0', help='Bind address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--workspace', default=None, help='Workspace root (default: ~/ctt-server-workspace)')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    # The server is HTTPS-only. By default a self-signed cert is generated/reused;
    # pass your own with --cert/--key.
    parser.add_argument('--cert', default=None, help='TLS certificate (PEM); default: auto self-signed')
    parser.add_argument('--key', default=None, help='TLS private key (PEM); default: auto self-signed')
    parser.add_argument(
        '--libcamera-log',
        action='store_true',
        help='Let libcamera log to the console. ctt-server then leaves LIBCAMERA_LOG_LEVELS '
        'untouched so you set the level yourself (e.g. LIBCAMERA_LOG_LEVELS=INFO ctt-server '
        '--libcamera-log); without this flag libcamera is quietened to WARN.',
    )
    args = parser.parse_args()

    # Keep the console quiet by default. With --libcamera-log we leave
    # LIBCAMERA_LOG_LEVELS alone so the user controls libcamera's verbosity from the
    # command line. libcamera reads the level once when it first loads, so this must
    # run before importing the app (which pulls in the camera module).
    if not args.libcamera_log:
        os.environ.setdefault('LIBCAMERA_LOG_LEVELS', 'WARN')

    # Flask is an optional extra; give a clear hint rather than an ImportError
    # traceback if ctt-server is launched from a bare `pip install rpi-ctt`.
    try:
        from .app import create_app
    except ImportError as err:
        missing = getattr(err, 'name', None) or 'a dependency'
        print(
            f"ctt-server needs the 'server' extra (missing: {missing}).\n"
            "Install it with:  pip install 'rpi-ctt[server]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from err

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ssl_context = _ssl_context(args)
    logging.getLogger(__name__).info('Serving on https://%s:%d', args.host, args.port)
    app = create_app(args.workspace)
    # threaded=True so streaming endpoints (preview MJPEG, CTT log SSE) don't block.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, ssl_context=ssl_context)


def _ssl_context(args):
    """Return an ssl_context for app.run (the server is HTTPS-only).

    Uses an explicit --cert/--key pair if given, otherwise reuses or generates a
    persistent self-signed certificate under <workspace>/.tls.
    """
    if args.cert and args.key:
        return (args.cert, args.key)

    from .sessions import resolve_workspace

    ws = resolve_workspace(args.workspace)
    tls = ws / '.tls'
    tls.mkdir(parents=True, exist_ok=True)
    cert, key = tls / 'cert.pem', tls / 'key.pem'
    if not (cert.exists() and key.exists()):
        _generate_self_signed(cert, key)
        logging.getLogger(__name__).info('Generated self-signed certificate in %s', tls)
    return (str(cert), str(key))


def _generate_self_signed(cert, key) -> None:
    """Generate a persistent self-signed cert+key via openssl (present on Pi OS)."""
    import shutil
    import socket
    import subprocess

    host = socket.gethostname()
    san = ['DNS:localhost', f'DNS:{host}', 'IP:127.0.0.1']
    try:
        ip = socket.gethostbyname(host)
        if ip and not ip.startswith('127.'):
            san.append(f'IP:{ip}')
    except OSError:
        pass

    openssl = shutil.which('openssl')
    if not openssl:
        raise SystemExit('HTTPS needs a certificate: install openssl, or pass --cert/--key.')
    subprocess.run(
        [
            openssl,
            'req',
            '-x509',
            '-newkey',
            'rsa:2048',
            '-nodes',
            '-keyout',
            str(key),
            '-out',
            str(cert),
            '-days',
            '3650',
            '-subj',
            '/CN=ctt-server',
            '-addext',
            'subjectAltName=' + ','.join(san),
        ],
        check=True,
        capture_output=True,
    )


if __name__ == '__main__':
    main()
