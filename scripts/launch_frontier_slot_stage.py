#!/usr/bin/env python3
"""Extract and stage the new auxiliary-slot graph after successful export."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/lift-sd-20260916')
ART=ROOT/'artifacts/frontier-20260917'


def main():
    ready=ART/'export-slot-v4/frontier-export-manifest.json'
    while not ready.exists():
        if datetime.now(timezone.utc)>=datetime(2026,9,16,23,tzinfo=timezone.utc):
            raise RuntimeError('Autonomous window ended before export completion')
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'scripts/extract_frontier_context.py'),
        str(ART/'export-slot-v4'),str(ART/'context-slot-v4'),'--width','151936',
        '--params',str(ART/'context-packed-v6/params.json'),'--physical-block','8',
        '--logical-block','7','--inject-rows','16'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'scripts/stage_frontier_slot_assets.py'),
        '--context',str(ART/'context-slot-v4'),'--base-stage',str(RAW/'stage-frontier-v2.json'),
        '--output',str(RAW/'stage-frontier-slot-v1.json')],check=True,cwd=ROOT)


if __name__=='__main__':main()
