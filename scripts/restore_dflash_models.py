"""Restore archived GGUF derivatives from retained checkpoints and pinned tools."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Restorer:
    def __init__(self, manifest, work):
        self.manifest_path = Path(manifest).resolve()
        self.bundle = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        self.root = Path(self.manifest['model_root'])
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.sources = {}
        self.materialized = {}
        self.environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1', HF_HOME=str(self.work/'hf-cache'),
            PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(self.bundle/'tools/quantize'))
        for relative, digest in self.manifest['tools'].items():
            assert sha256(self.bundle/relative) == digest, relative

    def source(self, key, patch=None):
        patch = patch or {}
        cache_key = (key, json.dumps(patch, sort_keys=True))
        if cache_key in self.sources:
            return self.sources[cache_key]
        spec = self.manifest['sources'][key]
        dest = self.work/'sources'/key/spec['alias']
        dest.mkdir(parents=True, exist_ok=True)
        for name, digest in spec['metadata'].items():
            source = self.bundle/'source-metadata'/key/name
            assert sha256(source) == digest, source
            shutil.copy2(source, dest/name)
        for name, record in spec['weights'].items():
            source = Path(record['path'])
            assert source.is_file() and sha256(source) == record['sha256'], source
            output = dest/name
            if not output.exists():
                output.symlink_to(source)
        if patch:
            config = dest/'config.json'
            values = json.loads(config.read_text())
            values.update(patch)
            config.write_text(json.dumps(values, indent=2)+'\n')
        self.sources[cache_key] = dest
        return dest

    def dependency(self, name):
        if name in self.materialized:
            return self.materialized[name]
        spec = self.manifest['models'].get(name) or self.manifest['retained_dependencies'][name]
        original = self.root/name
        if original.is_file() and sha256(original) == spec['sha256']:
            self.materialized[name] = original
        else:
            if name not in self.manifest['models']:
                raise FileNotFoundError(f'Retained custom dependency must be restored first: {original}')
            dest = self.work/'dependencies'/name
            dest.parent.mkdir(exist_ok=True)
            self.render(name, dest)
            self.materialized[name] = dest
        return self.materialized[name]

    def normalize_header(self, spec, pending):
        helper = self.bundle/'tools/gguf_header.py'
        module_spec = importlib.util.spec_from_file_location('recovery_header', helper)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        current = module.read_header(pending)
        original = spec['gguf']
        header = self.bundle/spec['header_file']
        assert sha256(header) == spec['header_sha256'], header
        data = header.read_bytes()
        if current['tensors'] != original['tensors']:
            mapped = {tensor['name']: tensor for tensor in current['tensors']}
            replaced = pending.with_name(pending.name+'.repacked')
            assert not replaced.exists(), replaced
            with pending.open('rb') as source, replaced.open('xb') as dest:
                dest.write(data)
                for tensor in original['tensors']:
                    name = tensor['name']
                    size = spec['tensor_bytes'][name]
                    dest.seek(original['data_offset']+tensor['offset'])
                    if name not in mapped:
                        extra = spec['retained_small_tensors'][name]
                        raw = self.bundle/extra['file']
                        assert sha256(raw) == extra['sha256'] and raw.stat().st_size == size
                        dest.write(raw.read_bytes())
                        continue
                    actual = mapped[name]
                    assert actual['shape'] == tensor['shape'] and actual['type'] == tensor['type'], name
                    source.seek(current['data_offset']+actual['offset'])
                    while size:
                        block = source.read(min(size,16*1024*1024))
                        assert block, name
                        dest.write(block)
                        size -= len(block)
                dest.truncate(spec['bytes'])
            replaced.replace(pending)
        elif current['data_offset'] == original['data_offset']:
            with pending.open('r+b') as stream:
                stream.write(data)
        else:
            replaced = pending.with_name(pending.name+'.reheader')
            assert not replaced.exists(), replaced
            with pending.open('rb') as source, replaced.open('xb') as dest:
                dest.write(data)
                source.seek(current['data_offset'])
                shutil.copyfileobj(source, dest, 16*1024*1024)
            replaced.replace(pending)

    def render(self, name, output):
        spec = self.manifest['models'][name]
        output = Path(output)
        if output.exists():
            assert sha256(output) == spec['sha256'], f'Refusing to overwrite a different file: {output}'
            return output
        recipe = spec['recipe']
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(output.name+'.restoring')
        assert not pending.exists(), f'Inspect existing partial file first: {pending}'
        if recipe['kind'] == 'convert':
            source = self.source(recipe['source'], recipe.get('config_patch'))
            command = [self.manifest['python'], str(self.bundle/'tools/converter/convert_hf_to_gguf.py'),
                       str(source), '--outtype', recipe['outtype'], '--outfile', str(pending)]
            if recipe.get('target'):
                command += ['--target-model-dir', str(self.source(recipe['target']))]
            if recipe.get('base'):
                command += ['--base-model', str(self.source(recipe['base']))]
        else:
            parent = self.dependency(recipe['parent'])
            command = [str(self.bundle/'tools/quantize/llama-quantize')]
            if recipe['output4']:
                command += ['--output-tensor-type', 'q4_0', '--token-embedding-type', 'q4_0']
            command += [str(parent), str(pending), 'q4_0', '4']
        log = self.work/(name+'.log')
        with log.open('w') as stream:
            subprocess.run(command, env=self.environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        actual = sha256(pending)
        if actual != spec['sha256']:
            self.normalize_header(spec, pending)
            actual = sha256(pending)
        if actual != spec['sha256']:
            raise ValueError(f'Rebuilt SHA mismatch for {name}: {actual}; inspect {log}. Original was not overwritten.')
        assert pending.stat().st_size == spec['bytes']
        os.link(pending, output)
        pending.unlink()
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path.home()/'dev/llm/dflash-models/_cleanup_20260929/manifest.json')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--model', action='append', default=[])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.list:
        for name, spec in manifest['models'].items():
            print(f"{spec['bytes']/2**30:7.3f} GiB  {name}")
        return
    if not args.model:
        parser.error('Specify --model NAME (repeatable), or --list')
    assert all(name in manifest['models'] for name in args.model), 'Unknown model name'
    output = args.output_dir or Path(manifest['model_root'])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gguf-restore-', dir=output) as work:
        restorer = Restorer(args.manifest, work)
        try:
            for name in args.model:
                path = restorer.render(name, output/name)
                print('RESTORED_SHA256_VERIFIED', path, flush=True)
        finally:
            logs = output/'_restore_logs'/str(time.time_ns())
            logs.mkdir(parents=True, exist_ok=True)
            for path in Path(work).glob('*.log'):
                shutil.copy2(path, logs/path.name)
            print('RESTORE_LOGS', logs, flush=True)


if __name__ == '__main__':
    main()
