"""Losslessly pack completed width contexts, or restore one package for replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/arlen-ddtree-20260928'
RAW=ROOT.parent/'MobiSpec.cpp/docs/results/raw/arlen-ddtree-20260928/v1'


def digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(8<<20),b''):
        value.update(block)
    return value.hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return digest(stream)


def inventory(precision):
    records={}
    for width in [32,64]:
        package=OUT/f'{precision}-a{width}'
        data=json.loads((package/'complete.json').read_text())
        for name,info in data['files'].items():
            if name.endswith('.bin'):
                records[f'{precision}-a{width}/context/{name}']=info
    return records


def read_archive(path, records, restore_width=None):
    process=subprocess.Popen(['zstd','-d','-c','--long=31',str(path)],stdout=subprocess.PIPE)
    seen=set()
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in records and member.name not in seen
                info=records[member.name]
                assert member.size==info['bytes']
                stream=archive.extractfile(member)
                destination=OUT/member.name
                restore=restore_width is not None and member.name.split('/')[0].endswith(f'-a{restore_width}')
                if restore and not destination.exists():
                    temp=destination.with_suffix('.restore-tmp')
                    with temp.open('wb') as output:
                        shutil.copyfileobj(stream,output,8<<20)
                    assert sha(temp)==info['sha256']
                    temp.replace(destination)
                    print('RESTORED',member.name,flush=True)
                else:
                    assert digest(stream)==info['sha256']
                    if restore:
                        assert sha(destination)==info['sha256']
                seen.add(member.name)
        assert seen==set(records)
        process.stdout.close()
        assert process.wait()==0
    finally:
        if process.poll() is None:
            process.terminate();process.wait()


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pack',choices=['W4A16','W4A8','W8A16','W4A16-deployed','W4A8-protected'])
    mode.add_argument('--restore',help='PRECISION:WIDTH')
    parser.add_argument('--allow-staged',action='store_true',help='Pack A32-complete/A64-smoke-verified contexts already resident on the exclusive phone runner')
    args=parser.parse_args()
    precision=args.pack or args.restore.split(':')[0]
    assert precision in ['W4A16','W4A8','W8A16','W4A16-deployed','W4A8-protected']
    target=OUT/f'{precision}-contexts.tar.zst'
    manifest=OUT/f'{precision}-archive.json'
    records=inventory(precision)
    if args.restore:
        width=int(args.restore.split(':')[1]);assert width in [32,64]
        info=json.loads(manifest.read_text())
        assert records==info['files'] and sha(target)==info['archive_sha256']
        needed=sum(r['bytes'] for n,r in records.items() if n.split('/')[0].endswith(f'-a{width}') and not (OUT/n).exists())
        assert shutil.disk_usage(OUT).free>needed+(6<<30)
        read_archive(target,records,width)
        return
    for width in [32,64]:
        if (RAW/f'main-{precision}-a{width}-complete.json').exists():
            continue
        assert args.allow_staged and width==64, 'Do not pack an unstaged or unmeasured target'
        assert (RAW/f'main-{precision}-a32-complete.json').exists()
        assert (RAW/f'smoke-{precision}-a64-complete.json').exists()
        active=json.loads((RAW/'active-target.json').read_text())
        source=OUT/f'{precision}-a64/complete.json'
        manifest_data=json.loads(source.read_text())
        assert active['precision']==precision and active['width']==64
        assert active['export_manifest_sha256']==sha(source) and active['files']==manifest_data['files']
        (OUT/f'{precision}-early-archive.json').write_text(json.dumps(dict(
            utc=datetime.now(timezone.utc).isoformat(),active_target=active,
            reason='A32 complete, A64 staged with byte hashes and exact-output smoke; current phone run needs no host context reads. Preserve full verified archive before removing duplicate server bins.'),indent=2)+'\n')
    if manifest.exists():
        assert json.loads(manifest.read_text())['files']==records
        assert sha(target)==json.loads(manifest.read_text())['archive_sha256']
        for name, record in records.items():
            original = OUT/name
            if original.exists():
                assert sha(original) == record['sha256']
                original.unlink()
        print('Archive verified; restored originals compacted.');return
    assert not target.exists()
    total=sum(r['bytes'] for r in records.values())
    assert shutil.disk_usage(OUT).free>total+(6<<30), 'Retain originals until full round-trip verification'
    for name,record in records.items():
        assert sha(OUT/name)==record['sha256']
    temporary=target.with_suffix('.zst.tmp')
    if not temporary.exists():
        with temporary.open('xb') as output:
            process=subprocess.Popen(['zstd','-T2','-6','--long=31','-c'],stdin=subprocess.PIPE,stdout=output)
            try:
                with tarfile.open(fileobj=process.stdin,mode='w|') as archive:
                    for name in sorted(records,key=lambda n:(Path(n).name,n)):
                        archive.add(OUT/name,arcname=name,recursive=False)
                        print('PACKED',name,flush=True)
                process.stdin.close()
                assert process.wait()==0
            finally:
                if process.poll() is None:
                    process.terminate();process.wait()
    read_archive(temporary,records)
    temporary.replace(target)
    manifest.write_text(json.dumps(dict(files=records,archive_sha256=sha(target),
        original_bytes=total,archive_bytes=target.stat().st_size,round_trip_verified=True),indent=2)+'\n')
    for name,record in records.items():
        path=OUT/name
        assert sha(path)==record['sha256']
        path.unlink()
    print('ARCHIVE_VERIFIED',precision,'saved_GiB',(total-target.stat().st_size)/(1<<30),flush=True)


if __name__=='__main__':
    main()
