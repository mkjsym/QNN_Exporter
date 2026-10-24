#!/usr/bin/env python3
"""Stage the frozen secondary bridge without replacing any measured artifact."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from prepare_frontier_bridge_assets import sha

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/home/youngmin/dev/llm/MobiSpec.cpp/docs/results/raw/lift-sd-20260916')


def main():
    assets=ROOT/'artifacts/frontier-20260917/bridge-dense-prefix-r512-v1'
    source=assets/'frontier-bridge-f32.gguf'
    base=json.loads((RAW/'stage-frontier-v2.json').read_text())
    design=json.loads((RAW/'frontier-secondary-design-v1.json').read_text())
    manifest=json.loads((assets/'manifest.json').read_text())
    assert manifest['adapter_sha256']==design['adapter_sha256']
    assert manifest['outputs'][source.name]==sha(source)
    contract=json.loads((RAW/'frontier-runtime-contract-dense-r512-v1/results.json').read_text())
    assert contract['assets_sha256']==sha(source) and len(contract['cases'])>=2
    output=RAW/'stage-frontier-refinement-v1.json';assert not output.exists()
    remote=base['remote']+'/bridge-dense-prefix-r512-v1.gguf'
    expected=dict(base['hashes']);expected[remote]=sha(source)
    adb=['adb','-H','127.0.0.1','-P','10037','-s','R5KL20FMLCN']
    with open('/tmp/mobispec-s26-experiments.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        boot=subprocess.check_output(adb+['shell','cat /proc/sys/kernel/random/boot_id'],text=True).strip()
        assert boot==base['boot']
        text=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*base['hashes']])],text=True)
        assert {s.split()[1]:s.split()[0] for s in text.splitlines()}==base['hashes']
        subprocess.run(adb+['shell','test ! -e '+shlex.quote(remote)],check=True)
        subprocess.run(adb+['push',str(source),remote],check=True)
        text=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*expected])],text=True)
        assert {s.split()[1]:s.split()[0] for s in text.splitlines()}==expected
        output.write_text(json.dumps(dict(boot=boot,remote=base['remote'],hashes=expected,
            base_stage_sha256=sha(RAW/'stage-frontier-v2.json'),design_sha256=sha(RAW/'frontier-secondary-design-v1.json'),
            source_sha256=sha(__file__),manifest_sha256=sha(assets/'manifest.json'),device_hashes_match=True),indent=2)+'\n')
    print('FRONTIER_REFINEMENT_STAGED',remote,flush=True)


if __name__=='__main__':main()
