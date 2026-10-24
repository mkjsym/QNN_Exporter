"""Stage the immutable ten-query context on the current S26 boot."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    adb = ['adb', '-H', '127.0.0.1', '-P', '10037', '-s', 'R5KL20FMLCN']
    remote = '/data/local/tmp/vlm-dflash-20260913/horizon-20260917/ctx-q10-v1'
    names = ['forward_0.bin', 'params.json', 'forward_0_json.json', 'io-contract.json']
    files = {str(a.context/n): remote+'/'+n for n in names}
    expected = {dest: sha(source) for source, dest in files.items()}
    plan = dict(files=files, hashes=expected, source_sha256=sha(__file__))
    a.output.with_name(a.output.stem+'-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    def shell(command):
        return subprocess.check_output(adb+['shell', command], text=True, timeout=120)
    with open('/tmp/mobispec-s26-experiments.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        boot = shell('cat /proc/sys/kernel/random/boot_id').strip()
        assert boot == '76bd5c5a-576b-4ba2-817b-6f7c3e280f1a'
        battery = int(re.search(r'^\s*level:\s*(\d+)', shell('dumpsys battery'), re.M)[1])
        assert battery >= 20, battery
        free_kib = int(shell('df -k /data/local/tmp').splitlines()[-1].split()[3])
        needed = sum(Path(f).stat().st_size for f in files)
        assert free_kib*1024 > needed + 1024**3, (free_kib, needed)
        subprocess.run(adb+['shell', 'test ! -e '+shlex.quote(remote)], check=True)
        subprocess.run(adb+['shell', 'mkdir -p '+shlex.quote(remote)], check=True)
        for source, dest in files.items():
            subprocess.run(adb+['push', source, dest], check=True)
        actual = {line.split()[1]: line.split()[0] for line in shell(shlex.join(['sha256sum', *expected])).splitlines()}
        assert actual == expected
        assert shell('cat /proc/sys/kernel/random/boot_id').strip() == boot
        plan.update(boot=boot, battery_percent=battery, remote=remote, device_hashes_match=True)
        a.output.write_text(json.dumps(plan, indent=2)+'\n')
    print('HORIZON_CONTEXT_STAGED', remote, flush=True)


if __name__ == '__main__':
    main()
