"""Pack only fully measured precision pairs while later exports continue."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/'MobiSpec.cpp/docs/results/raw/arlen-ddtree-20260928/v1'
deadline=datetime.fromisoformat('2026-09-28T23:20:00+00:00').timestamp()
for precision in ['W4A16','W4A8','W8A16']:
    while not all((RAW/f'main-{precision}-a{width}-complete.json').exists() for width in [32,64]):
        if time.time()>deadline:
            raise TimeoutError('Archive queue deadline')
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'scripts/archive_arlen_contexts.py'),
                    '--pack',precision],check=True)
print('ARCHIVE_QUEUE_COMPLETE',flush=True)
