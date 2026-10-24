"""Export user-requested 20k full/none comparators with the established recipe."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MOBI = ROOT.parent / 'MobiSpec.cpp'
OUT = ROOT / 'artifacts/vlm-output-timing-20261008/export'
OLD = ROOT / 'artifacts/vlm-data-scale-20261006/export'
ET = ROOT.parent / 'executorch'
SDK = Path('/home/youngmin/qairt/2.37.1-v81')
AW = '/home/youngmin/anaconda3/envs/awbench/bin/python'
PY = '/home/youngmin/anaconda3/envs/etqnn/bin/python'


def configure(arm):
    import vlm_data_scale_export_common as common
    assert arm in ('20k-full', '20k-none')
    cp = common.checkpoint(arm, 3000) / 'model.safetensors'
    value = dict(selected=True, arm=arm, step=3000, mode=arm.split('-')[1],
                 key=arm + '-step3000', checkpoint_sha256=common.sha(cp),
                 reason='Explicit user request: add both20k interfaces to the timing comparison')
    common.OUT = OUT
    common.candidate = lambda: value
    sys.modules['vlm_residual8_export_common'] = common
    return common, value


def execute(phase, arm, extra):
    common, c = configure(arm)
    sys.argv = [__file__, *extra]
    if phase == 'convert':
        import prepare_vlm_data_scale_export as prepare
        prepare.torch.set_num_threads(4)
        prepare.convert()
        path = OUT / c['key'] / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['selection_sha256'] = common.sha(OUT / 'requested-checkpoints.json')
        manifest['selection_source'] = 'User-requested comparator matrix, not host-screen selection'
        common.write(path, manifest)
    elif phase == 'reference':
        import prepare_vlm_data_scale_export_references as references
        references.main()
    elif phase == 'audit':
        from check_vlm_residual8_export import main
        sys.argv = [__file__, '--arm', c['key']]
        main()
    elif phase == 'export':
        from export_vlm_residual8_draft import main
        main()
    else:
        raise ValueError(phase)


def queue(arms):
    sys.path.insert(0, str(MOBI / 'tools/training'))
    from vlm_data_scale import checkpoint, sha, write
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), *[ROOT / 'scripts' / n for n in (
        'vlm_data_scale_export_common.py', 'prepare_vlm_data_scale_export.py',
        'prepare_vlm_data_scale_export_references.py', 'check_vlm_residual8_export.py',
        'export_vlm_residual8_draft.py', 'extract_variable_draft_context.py')]]
    design = dict(arms=['20k-full', '20k-none'], step=3000,
                  checkpoint_sha256={a: sha(checkpoint(a,3000)/'model.safetensors') for a in ('20k-full','20k-none')},
                  calibration_manifest_sha256=sha(OLD/'calibration/manifest.json'),
                  source_sha256={str(p): sha(p) for p in sources},
                  root_reserve_GiB=50, precision='Unchanged mixed W4/W8 A16, headW8, KV8',
                  native_B7=True, physical_B8=True, open_rows=7, injection_rows=16)
    path = OUT / 'requested-checkpoints.json'
    if path.exists():
        assert json.loads(path.read_text()) == design
    else:
        write(path, design)
    calibration = OUT / 'calibration'
    if not calibration.exists():
        calibration.symlink_to(OLD / 'calibration', target_is_directory=True)
    assert calibration.resolve() == (OLD / 'calibration').resolve()
    manifest = json.loads((calibration/'manifest.json').read_text())
    train = {json.loads(line)['id'] for line in (MOBI/'docs/results/raw/vlm-data-scale-20261006/data/train20k.jsonl').read_text().splitlines()}
    assert len(manifest['records']) == 16
    for row in manifest['records']:
        assert row['id'] in train and row['split']=='train' and sha(row['path'])==row['sha256']
    for filename, digest in manifest['outputs'].items():
        assert sha(calibration/filename)==digest
    env = dict(os.environ, EXECUTORCH_ROOT=str(ET), QNN_SDK_ROOT=str(SDK),
               PYTHONPATH=f'{ET.parent}:{ET}:{MOBI}/tools/training',
               LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6')
    for arm in arms:
        key = arm + '-step3000'
        asset = OUT/key
        artifact = OUT/('npu-export-'+key)
        context = OUT/('ctx-'+key)
        jobs = [(phase, [AW if phase=='reference' else PY, str(Path(__file__)),
                         phase, '--arm', arm]) for phase in ('convert','reference','audit')]
        jobs.append(('export', [PY, '-u', str(Path(__file__)), 'export', '--arm', arm,
            '--decoder_model', 'qwen3-4b-dflash-w4', '--model_mode', 'hybrid',
            '--prefill_ar_len', '8', '--max_context_len', '1024', '--soc_model', 'SM8850',
            '--build_folder', str(ET/'build-android'), '--artifact', str(artifact),
            '--compile_only', '--checkpoint', str(asset/'draft-meta.pth'),
            '--calib_samples', str(calibration/'calibration-prompts.json'),
            '--dflash_hidden', str(calibration/'calibration-hidden.pt'), '--prompt', 'Once upon a time']))
        jobs.append(('extract', [PY, str(ROOT/'scripts/extract_variable_draft_context.py'),
                    str(artifact), str(context), '--width', '151936', '--params',
                    str(ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json'),
                    '--physical-block', '8', '--logical-block', '7', '--inject-rows', '16']))
        state = dict(status='running', pid=os.getpid(), arm=arm, key=key)
        try:
            for phase, cmd in jobs:
                seal = OUT/(key+'-'+phase+'-complete.json')
                if seal.exists():
                    assert json.loads(seal.read_text())['command']==cmd
                    continue
                assert shutil.disk_usage(OUT).free>50*(1<<30), 'Storage guard; no deletion'
                state.update(phase=phase, command=cmd)
                write(OUT/'queue-status.json', state)
                with (OUT/(key+'-'+phase+'.log')).open('x') as log:
                    child = subprocess.Popen(cmd,cwd=ET,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid'] = child.pid
                write(OUT/'queue-status.json', state)
                print('EXPORT_PHASE_START',key,phase,flush=True)
                start = time.monotonic()
                while child.poll() is None:
                    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                    if shutil.disk_usage(OUT).free<50*(1<<30) or mem['MemAvailable']<24*(1<<20) or time.monotonic()-start>7200:
                        os.killpg(child.pid,signal.SIGTERM)
                        child.wait(timeout=30)
                        raise RuntimeError('Resource/time guard; no deletion')
                    time.sleep(5)
                assert child.returncode==0, phase+' failed; inspect log'
                write(seal,dict(command=cmd,passed=True,log_sha256=sha(OUT/(key+'-'+phase+'.log'))))
                print('EXPORT_PHASE_PASS',key,phase,flush=True)
            value = json.loads((asset/'manifest.json').read_text())
            write(context/'runtime-contract.json',dict(context_sha256=sha(context/'forward_0.bin'),
                  params_sha256=sha(context/'params.json'),asset_manifest_sha256=sha(asset/'manifest.json'),
                  checkpoint_sha256=value['checkpoint_sha256'],mode=value['mode'],
                  physical_block=8,open_block=7,output_rows=list(range(7)),target_ar_len=32,
                  ddtree_budget=31,ddtree_k=4,summary_sha256=None))
            state.update(status='complete',context=str(context),context_sha256=sha(context/'forward_0.bin'))
            write(OUT/(key+'-complete.json'),state)
        except BaseException as error:
            state.update(status='failed',error=repr(error))
            raise
        finally:
            write(OUT/'queue-status.json',state)
    assert all(sha(p)==h for p,h in design['source_sha256'].items())
    print('USER_MATRIX_EXPORT_COMPLETE',arms,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('queue','convert','reference','audit','export'))
    parser.add_argument('--arm',choices=('20k-full','20k-none'))
    args, extra=parser.parse_known_args()
    if args.phase=='queue':
        queue([args.arm] if args.arm else ['20k-full','20k-none'])
    else:
        assert args.arm
        execute(args.phase,args.arm,extra)
