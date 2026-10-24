"""Convert, reference, audit, export and extract the register study's mobile drafters one arm after another, with resource guards."""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from vlm_register_mobile_export_common import ROOT, MOBI, OUT, ARMS, cache_rows, context_suffix, sha, write
ET = ROOT.parent/'executorch'; SDK = Path('/home/youngmin/qairt/2.37.1-v81')
AW = '/home/youngmin/anaconda3/envs/awbench/bin/python'; PY = '/home/youngmin/anaconda3/envs/etqnn/bin/python'


def main(arms, context_len=1024, phase_limit_s=5400, inject_rows=0):
    if inject_rows: return main_inject(arms, inject_rows, phase_limit_s)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EXECUTORCH_ROOT=str(ET), QNN_SDK_ROOT=str(SDK), PYTHONPATH=f'{ET.parent}:{ET}:{MOBI}/tools/training:{ROOT}/scripts',
               LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', REGISTER_CAP='doc20k')
    sfx = context_suffix(context_len)
    state = dict(status='running', pid=os.getpid(), arms=arms, context_len=context_len, source_sha256=sha(__file__)); S = ROOT/'scripts'
    def run(phase, cmd, log):
        assert shutil.disk_usage(OUT).free > 50*(1 << 30)
        state.update(phase=phase, command=cmd); write(OUT/'queue-status.json', state)
        with open(log, 'a') as out:
            child = subprocess.Popen(cmd, cwd=ET, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        while child.poll() is None:
            mem = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
            if shutil.disk_usage(OUT).free < 50*(1 << 30) or mem['MemAvailable'] < 24*(1 << 20) or time.monotonic()-start > phase_limit_s:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=30); raise RuntimeError(f'{phase}: resource/time guard')
            time.sleep(5)
        assert child.returncode == 0, f'{phase} failed; see {log}'
    try:
        if not (OUT/'calibration/manifest.json').exists(): run('calibration', [AW, str(S/'prepare_vlm_register_mobile_export.py'), 'calibration'], OUT/'calibration.log')
        for arm in arms:
            asset = OUT/arm; artifact = OUT/('npu-export-'+arm+sfx); context = OUT/('ctx-'+arm+sfx)
            if (context/'runtime-contract.json').exists(): continue
            if not (asset/'manifest.json').exists(): run('convert', [PY, str(S/'prepare_vlm_register_mobile_export.py'), 'convert', '--arm', arm], OUT/f'{arm}-convert.log')
            if not (asset/'native-references.json').exists(): run('references', [AW, str(S/'prepare_vlm_register_mobile_references.py'), arm], OUT/f'{arm}-references.log')
            # The FP32 audit checks the converted weights on the deployed graph geometry; an asset audited at another context is audited again for this one.
            audit = asset/'fp32-audit.json'
            if not audit.exists() or json.loads(audit.read_text()).get('max_context_len', 1024) != context_len:
                run('audit', [PY, str(S/'check_vlm_register_mobile_export.py'), '--arm', arm, '--context', str(context_len)], OUT/f'{arm}{sfx}-audit.log')
            if not artifact.exists():
                run('export', [PY, '-u', str(S/'export_vlm_register_mobile_draft.py'), '--decoder_model', 'qwen3-4b-dflash-w4', '--model_mode', 'hybrid',
                               '--prefill_ar_len', '8', '--max_context_len', str(context_len), '--soc_model', 'SM8850', '--build_folder', str(ET/'build-android'),
                               '--artifact', str(artifact), '--compile_only', '--checkpoint', str(asset/'draft-meta.pth'),
                               '--calib_samples', str(OUT/'calibration/calibration-prompts.json'), '--dflash_hidden', str(OUT/'calibration/calibration-hidden.pt'),
                               '--prompt', 'Once upon a time'], OUT/f'{arm}{sfx}-export.log')
            run('extract', [PY, str(S/'extract_variable_draft_context.py'), str(artifact), str(context), '--width', '151936',
                            '--params', str(ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json'), '--physical-block', '8', '--logical-block', '7', '--inject-rows', '16'],
                OUT/f'{arm}{sfx}-extract.log')
            manifest = json.loads((asset/'manifest.json').read_text())
            write(context/'runtime-contract.json', dict(arm=arm, mode=manifest['mode'], open_block=7, output_rows=list(range(7)), target_ar_len=32, inject_rows=16,
                                                       max_context_len=context_len, cache_rows=cache_rows(context_len),
                                                       checkpoint_sha256=manifest['checkpoint_sha256'], context_sha256=sha(context/'forward_0.bin'), params_sha256=sha(context/'params.json'),
                                                       side_files={n: sha(asset/n) for n in ('summary-f32.bin', 'registers-f32.bin', 'layer-prompts-f32.bin') if (asset/n).exists()}))
            print('REGISTER_MOBILE_EXPORTED', arm+sfx, flush=True)
        state.update(status='complete')
    except BaseException as e:
        state.update(status='failed', error=repr(e)); raise
    finally:
        write(OUT/'queue-status.json', state)


def main_inject(arms, inject_rows, phase_limit_s):
    """M4 (plan 11.2): per arm, the FP32 inject audit (G0), the inject-only prefill graph export and its extraction, artifact npu-export-<arm>-inject<N>, context ctx-<arm>-inject<N>."""
    from vlm_register_mobile_export_common import inject_context, inject_suffix, INJECT_CACHE_ROWS as CACHE_ROWS
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EXECUTORCH_ROOT=str(ET), QNN_SDK_ROOT=str(SDK), PYTHONPATH=f'{ET.parent}:{ET}:{MOBI}/tools/training:{ROOT}/scripts',
               LD_LIBRARY_PATH=str(SDK/'lib/x86_64-linux-clang'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', REGISTER_CAP='doc20k')
    sfx = inject_suffix(inject_rows); context_len = inject_context(inject_rows)
    state = dict(status='running', pid=os.getpid(), arms=arms, inject_rows=inject_rows, context_len=context_len, source_sha256=sha(__file__)); S = ROOT/'scripts'
    def run(phase, cmd, log):
        assert shutil.disk_usage(OUT).free > 50*(1 << 30)
        state.update(phase=phase, command=cmd); write(OUT/'queue-status.json', state)
        with open(log, 'a') as out:
            child = subprocess.Popen(cmd, cwd=ET, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        while child.poll() is None:
            mem = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
            if shutil.disk_usage(OUT).free < 50*(1 << 30) or mem['MemAvailable'] < 24*(1 << 20) or time.monotonic()-start > phase_limit_s:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=30); raise RuntimeError(f'{phase}: resource/time guard')
            time.sleep(5)
        assert child.returncode == 0, f'{phase} failed; see {log}'
    try:
        assert (OUT/'calibration/manifest.json').exists()
        for arm in arms:
            asset = OUT/arm; artifact = OUT/('npu-export-'+arm+sfx); context = OUT/('ctx-'+arm+sfx)
            if (context/'runtime-contract.json').exists(): continue
            assert (asset/'manifest.json').exists(), f'{arm}: convert the step-graph asset first (queue without --inject)'
            audit = asset/'fp32-inject-audit.json'
            if not audit.exists() or json.loads(audit.read_text()).get('inject_rows') != inject_rows:
                run('inject-audit', [PY, str(S/'check_vlm_register_mobile_inject.py'), '--arm', arm, '--inject-rows', str(inject_rows)], OUT/f'{arm}{sfx}-audit.log')
            if not artifact.exists():
                run('inject-export', [PY, '-u', str(S/'export_vlm_register_mobile_inject.py'), '--inject_rows', str(inject_rows), '--decoder_model', 'qwen3-4b-dflash-w4', '--model_mode', 'hybrid',
                                      '--prefill_ar_len', '8', '--max_context_len', str(context_len), '--max_seq_len', str(context_len), '--soc_model', 'SM8850', '--build_folder', str(ET/'build-android'),
                                      '--artifact', str(artifact), '--compile_only', '--checkpoint', str(asset/'draft-meta.pth'),
                                      '--calib_samples', str(OUT/'calibration/calibration-prompts.json'), '--dflash_hidden', str(OUT/'calibration/calibration-hidden.pt'),
                                      '--prompt', 'Once upon a time'], OUT/f'{arm}{sfx}-export.log')
            run('inject-extract', [PY, str(S/'extract_variable_draft_context.py'), str(artifact), str(context), '--width', '2560', '--inject-only',
                                   '--params', str(ET/'examples/models/qwen3_dflash/config/4b_dflash_config.json'), '--physical-block', '8', '--logical-block', '7', '--inject-rows', str(inject_rows)],
                OUT/f'{arm}{sfx}-extract.log')
            manifest = json.loads((asset/'manifest.json').read_text())
            write(context/'runtime-contract.json', dict(arm=arm, mode=manifest['mode'], kind='inject', inject_rows=inject_rows, cache_rows=CACHE_ROWS, max_context_len=context_len,
                                                       block_rows=8, primary_width=2560, primary='block_embd plus the mask last column, unused',
                                                       outputs='k_ctx x5 [1,8,128,N] float then v_ctx x5 [1,8,N,128] float, same order as the step graph',
                                                       step_context='ctx-'+arm+('-c4096' if manifest['mode'] == 'full' else ''),
                                                       checkpoint_sha256=manifest['checkpoint_sha256'], context_sha256=sha(context/'forward_0.bin'), params_sha256=sha(context/'params.json'),
                                                       inject_audit_sha256=sha(audit)))
            print('REGISTER_MOBILE_INJECT_EXPORTED', arm+sfx, flush=True)
        state.update(status='complete')
    except BaseException as e:
        state.update(status='failed', error=repr(e)); raise
    finally:
        write(OUT/'queue-status.json', state)


if __name__ == '__main__':
    argv = sys.argv[1:]; context_len = 1024; limit = 5400; inject_rows = 0
    if '--inject' in argv: i = argv.index('--inject'); inject_rows = int(argv[i+1]); del argv[i:i+2]
    if '--context' in argv: i = argv.index('--context'); context_len = int(argv[i+1]); del argv[i:i+2]
    if '--phase-limit-s' in argv: i = argv.index('--phase-limit-s'); limit = int(argv[i+1]); del argv[i:i+2]
    main(argv or ['none-20k-seed42', 'norm32-20k-seed42', 'dynreg8-20k-seed42', 'norm8-20k-seed42', 'deep-dynreg8-20k-seed42'], context_len, limit, inject_rows)
