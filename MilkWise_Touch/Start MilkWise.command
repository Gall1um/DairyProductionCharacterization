#!/bin/bash
# Double-click on macOS. The app uses only this folder, not your VS Code setup.
cd "$(dirname "$0")" || exit 1
for PY in python3.13 python3.12 python3.11 python3.10 python3.14 python3; do
  if command -v "$PY" >/dev/null 2>&1; then
    if "$PY" -c 'import sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
      "$PY" bootstrap.py
      STATUS=$?
      if [ "$STATUS" -ne 0 ]; then
        echo
        echo 'MilkWise could not start. See the error above.'
        read -r -p 'Press Return to close this window...' _
      fi
      exit "$STATUS"
    fi
  fi
done
echo 'MilkWise needs Python 3.10 or newer.'
echo 'Install Python 3.12 or 3.13 from https://www.python.org/downloads/ and try again.'
read -r -p 'Press Return to close this window...' _
exit 1
