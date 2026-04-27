#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python scripts/run_local_hgdn_wallclock_resolver.py "$@"
