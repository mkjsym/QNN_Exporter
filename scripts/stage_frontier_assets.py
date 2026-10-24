#!/usr/bin/env python3
"""Stage a new frontier graph and immutable bridge weights under the S26 lock."""
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
    p.add_argument('--adapters',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repair-from',type=Path)
    args=p.parse_args()
    assert not args.output.exists()
    adb=['adb','-H','127.0.0.1','-P','10037','-s','R5KL20FMLCN']
    remote='/data/local/tmp/vlm-dflash-20260913/frontier-20260917'
    files={str(args.context/'forward_0.bin'):remote+'/ctx-packed-v6/forward_0.bin',
           str(args.context/'params.json'):remote+'/ctx-packed-v6/params.json',
           str(args.context/'forward_0_json.json'):remote+'/ctx-packed-v6/forward_0_json.json'}
    for adapter in args.adapters:
        assert adapter.name=='frontier-bridge-f32.gguf'
        files[str(adapter)]=remote+'/'+adapter.parent.name+'.gguf'
    expected={dest:sha(source) for source,dest in files.items()}
    plan=dict(files=files,hashes=expected,source_sha256=sha(__file__),
              context_contract_sha256=sha(args.context/'io-contract.json'))
    args.output.with_name(args.output.stem+'-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with open('/tmp/mobispec-s26-experiments.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        boot=subprocess.check_output(adb+['shell','cat /proc/sys/kernel/random/boot_id'],text=True).strip()
        assert boot=='76bd5c5a-576b-4ba2-817b-6f7c3e280f1a'
        previous={}
        if args.repair_from:
            previous=json.loads(args.repair_from.read_text())['hashes']
            assert all(expected.get(path)==digest for path,digest in previous.items())
            observed=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*previous])],text=True)
            assert {line.split()[1]:line.split()[0] for line in observed.splitlines()}==previous
            plan['repair_from_sha256']=sha(args.repair_from)
        else:
            subprocess.run(adb+['shell','test ! -e '+shlex.quote(remote)],check=True)
        subprocess.run(adb+['shell','mkdir -p '+shlex.quote(remote+'/ctx-packed-v6')],check=True)
        for source,dest in files.items():
            if dest not in previous:
                subprocess.run(adb+['shell','test ! -e '+shlex.quote(dest)],check=True)
                subprocess.run(adb+['push',source,dest],check=True)
        observed=subprocess.check_output(adb+['shell',shlex.join(['sha256sum',*expected])],text=True)
        hashes={line.split()[1]:line.split()[0] for line in observed.splitlines()}
        assert hashes==expected
        plan.update(boot=boot,remote=remote,device_hashes_match=True)
        args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print('FRONTIER_ASSETS_STAGED',remote,flush=True)


if __name__=='__main__':main()
