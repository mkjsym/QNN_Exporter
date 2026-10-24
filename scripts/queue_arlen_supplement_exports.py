"""Export the predeclared deployed control, then optional A16 sweeps if time permits."""
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/arlen-ddtree-20260928'
RAW = ROOT.parent / 'MobiSpec.cpp/docs/results/raw/arlen-ddtree-20260928/v1'


def before(hour_utc):
    return time.time() < datetime.fromisoformat(hour_utc).timestamp()


while not (OUT / 'queue-complete.json').exists():
    if not before('2026-09-28T20:00:00+00:00'):
        raise TimeoutError('Mandatory exports not ready by 05:00 KST')
    time.sleep(30)

jobs = [('W4A16-deployed', 32), ('W4A16-deployed', 64),
        ('W4A8-protected', 32), ('W4A8-protected', 64)]
for precision, width in jobs:
    subprocess.run([sys.executable, str(ROOT / 'scripts/queue_arlen_exports.py'),
                    '--only', f'{precision}:{width}'], check=True)

def controls_archived():
    gate_path = RAW / 'protected-quality-gate.json'
    gate_failed = gate_path.exists() and not json.loads(gate_path.read_text())['promote']
    return ((OUT / 'W4A16-deployed-archive.json').exists() and
            ((OUT / 'W4A8-protected-archive.json').exists() or gate_failed))


def first_width_has_space():
    if not (OUT / 'W4A16-deployed-archive.json').exists():
        return False
    protected = sum(record['bytes'] for width in [32,64] for name,record in
                    json.loads((OUT/f'W4A8-protected-a{width}/complete.json').read_text())['files'].items()
                    if name.endswith('.bin'))
    # Preserve the archive's full-original-size reserve plus 6 GiB even after the first A16 export.
    return shutil.disk_usage(OUT).free > protected + int(3.2*(1<<30)) + (6<<30)


while before('2026-09-28T21:00:00+00:00'):
    if controls_archived() or first_width_has_space():
        break
    time.sleep(30)

# Require two hours for the entire optional three-precision width before launching it.
if before('2026-09-28T21:00:00+00:00'):
    jobs = [(p, 16) for p in ['W4A16', 'W4A8', 'W8A16']]
    plan = dict(widths=[16], jobs=jobs, decided_utc=datetime.now(timezone.utc).isoformat(),
                reason='All mandatory and both mixed-control targets compiled before 06:00; first A16 may overlap protected measurement only with explicit archive-space reserve. Remaining A16 exports wait for control archives.')
    (OUT / 'extra-widths-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for index, (precision, width) in enumerate(jobs):
        if index:
            while not controls_archived():
                if not before('2026-09-28T22:00:00+00:00'):
                    raise TimeoutError('Control archives not ready for remaining optional exports by 07:00 KST')
                time.sleep(30)
        subprocess.run([sys.executable, str(ROOT / 'scripts/queue_arlen_exports.py'),
                        '--only', f'{precision}:{width}'], check=True)
    (OUT / 'extra-exports-complete.json').write_text(json.dumps(plan, indent=2) + '\n')

(OUT / 'supplement-export-complete.json').write_text(json.dumps(dict(complete=True)) + '\n')
print('SUPPLEMENT_EXPORTS_COMPLETE', flush=True)
