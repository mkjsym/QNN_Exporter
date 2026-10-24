#!/usr/bin/env python3
"""Stage an additional FCS graph while preserving the first measured graph."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from prepare_frontier_bridge_assets import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--base-stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    contract=json.loads((args.context/'io-contract.json').read_text())
    assert contract['width']==151936 and contract['physical_block']==8
    assert len(contract['primary'])==1 and contract['primary'][0]['dataType']=='QNN_DATATYPE_FLOAT_32'
    base=json.loads(args.base_stage.read_text())
    assert args.context.name.startswith('context-slot-v')
    remote=base['remote']+'/'+args.context.name.replace('context-slot-','ctx-slot-')
    files={str(args.context/name):remote+'/'+name for name in ['forward_0.bin','forward_0_json.json','params.json']}
    expected=dict(base['hashes'])
    expected.update({dest:sha(source) for source,dest in files.items()})
    adb=['adb','-H','127.0.0.1','-P','10037','-s','R5KL20FMLCN']
    with open('/tmp/mobispec-s26-experiments.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        boot=subprocess.check_output(adb+['shell','cat /proc/sys/kernel/random/boot_id'],text=True).strip()
        assert boot==base['boot']
        actual=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*base['hashes']])],text=True)
        assert {line.split()[1]:line.split()[0] for line in actual.splitlines()}==base['hashes']
        subprocess.run(adb+['shell','test ! -e '+shlex.quote(remote)],check=True)
        subprocess.run(adb+['shell','mkdir -p '+shlex.quote(remote)],check=True)
        for source,dest in files.items():subprocess.run(adb+['push',source,dest],check=True)
        actual=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*expected])],text=True)
        assert {line.split()[1]:line.split()[0] for line in actual.splitlines()}==expected
        args.output.write_text(json.dumps(dict(boot=boot,remote=remote,hashes=expected,
            base_stage_sha256=sha(args.base_stage),source_sha256=sha(__file__),files=files,
            context_contract_sha256=sha(args.context/'io-contract.json'),device_hashes_match=True),indent=2)+'\n')
    print('FRONTIER_SLOT_ASSETS_STAGED',remote,flush=True)


if __name__=='__main__':main()
