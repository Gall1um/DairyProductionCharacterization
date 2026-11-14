#!/usr/bin/env python3
"""MilkWise isolated launcher: create a local Python environment, then run the web UI.

All Python packages are installed only in .milkwise_env/ beside this file.
No VS Code project files or global site-packages are modified.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
ENV = ROOT / '.milkwise_env'
MARKER = ENV / '.milkwise_setup.json'
REQUIREMENTS = ROOT / 'requirements.txt'


def info(message: str) -> None:
    print(f'\n[MilkWise] {message}', flush=True)


def local_python() -> Path:
    if os.name == 'nt':
        return ENV / 'Scripts' / 'python.exe'
    return ENV / 'bin' / 'python'


def ensure_environment() -> Path:
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or newer is required. Install Python 3.12 or 3.13 from python.org, then relaunch.')
    requirements_sha = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    expected = {
        'folder': str(ROOT),
        'python': str(Path(sys.executable).resolve()),
        'version': '.'.join(str(x) for x in sys.version_info[:2]),
        'requirements_sha256': requirements_sha,
    }
    py = local_python()
    try:
        actual = json.loads(MARKER.read_text())
    except (OSError, ValueError):
        actual = None
    valid = actual == expected and py.exists()
    if valid:
        probe = subprocess.run([str(py), '-c', 'import streamlit, sklearn, pandas, openpyxl'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        valid = (probe.returncode == 0)
    if not valid:
        if ENV.exists():
            info('Rebuilding the private environment inside this folder (for example after moving the app).')
            shutil.rmtree(ENV)
        info('Setting up a private environment inside MilkWise_Portable ...')
        venv.create(ENV, with_pip=True, clear=False)
        if not py.exists():
            raise RuntimeError('The private Python environment could not be created.')
        info('Installing the app dependencies into the private environment ...')
        env = os.environ.copy()
        env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        env['PIP_NO_INPUT'] = '1'
        result = subprocess.run([str(py), '-m', 'pip', 'install', '--no-cache-dir', '-r',
                                 str(REQUIREMENTS)], cwd=str(ROOT), env=env)
        if result.returncode != 0:
            raise RuntimeError('Dependency installation failed. Check the internet connection and Python version, then relaunch.')
        MARKER.write_text(json.dumps(expected, indent=2))
    return py


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def launch(py: Path) -> int:
    port = free_local_port()
    url = f'http://127.0.0.1:{port}'
    environment = os.environ.copy()
    environment['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    environment['PYTHONNOUSERSITE'] = '1'
    command = [str(py), '-m', 'streamlit', 'run', str(ROOT/'app.py'),
               '--server.address=127.0.0.1', f'--server.port={port}',
               '--server.headless=true', '--server.fileWatcherType=none',
               '--browser.gatherUsageStats=false']
    info('Starting MilkWise. Keep this window open while using the app.')
    process = subprocess.Popen(command, cwd=str(ROOT), env=environment)
    try:
        healthy = False
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('The app exited before it could start. See the messages above.')
            try:
                with urllib.request.urlopen(url + '/_stcore/health', timeout=0.5) as response:
                    if response.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(0.5)
        if not healthy:
            raise RuntimeError('The app did not respond. See the messages above for details.')
        info(f'Opening {url} in your browser. Close this window or press Ctrl+C to stop.')
        if not webbrowser.open(url):
            info('If the browser did not open, copy this URL into your browser: ' + url)
        return process.wait()
    except KeyboardInterrupt:
        info('Stopping MilkWise ...')
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    info('Portable edition — runs separately from VS Code.')
    try:
        return launch(ensure_environment())
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        info('ERROR: ' + str(exc))
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
