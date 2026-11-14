#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
for PY in python3.13 python3.12 python3.11 python3.10 python3.14 python3; do
  if command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
    exec "$PY" bootstrap.py
  fi
done
echo 'MilkWise needs Python 3.10+; install Python and then run this launcher.'
exit 1
