#!/usr/bin/env bash
# Serial re-run of p8(Western)/p9(asian_kyc)/p10(SCUT) M2 vs the FAITHFUL :8001 FakeVLM detector,
# AFTER the is_server_up rstrip bug fix + fatal-fallback guard. Runs sequentially (no parallel /health
# starvation). Each script writes its own timestamped outputs/ dir, so prior (invalid-viviai) runs stay.
set -uo pipefail
cd /data/disk4/lyx_ICML/self_evolution_forgery

echo "=== pre-flight: :8001 health ==="
code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8001/health || echo 000)
if [ "$code" != "200" ]; then
  echo "FATAL: :8001 /health returned $code — start scripts/fakevlm_raw_server.py first. Aborting."
  exit 1
fi
echo "  :8001 /health=200 OK"

for s in scripts/p8_faithful.sh scripts/p9_asian_ab.sh scripts/p10_scut_ab.sh; do
  echo "===== $(date +%H:%M:%S) START $s ====="
  bash "$s"
  rc=$?
  echo "===== $(date +%H:%M:%S) END   $s (rc=$rc) ====="
  if [ $rc -ne 0 ]; then
    echo "WARN: $s exited rc=$rc — continuing to next (check its run.log for the fatal-fallback guard)."
  fi
done
echo "=== ALL SERIAL RERUNS DONE $(date +%H:%M:%S) ==="
