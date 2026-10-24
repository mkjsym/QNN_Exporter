#!/usr/bin/env python3
"""Audit compiled full-head quantization controls without device inference."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);a=p.parse_args();root=a.artifacts
    full=json.loads((root/'ctx-full/io-contract.json').read_text())
    w4=json.loads((root/'ctx-head-w4/io-contract.json').read_text())
    assert full['inputs']==w4['inputs'] and full['outputs']==w4['outputs']
    assert len(full['inputs'])==19 and w4['width']==151936
    for name,contract in [('ctx-full',full),('ctx-head-w4',w4)]:
        assert sha(root/name/'forward_0.bin')==contract['context_sha256']
    assert w4['bytes']<full['bytes']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        input_tensors_exact=19,all_output_contracts_exact=True,
        full_w8_bytes=full['bytes'],head_w4_bytes=w4['bytes'],byte_reduction=full['bytes']-w4['bytes'],
        fractional_size_reduction=1-w4['bytes']/full['bytes'],
        full_w8_sha256=full['context_sha256'],head_w4_sha256=w4['context_sha256'],
        source_sha256=sha(Path(__file__)),export_manifest_sha256=sha(root/'head-w4-control-v2/head-w4-export-manifest.json'),
        device_inference_run=False,numerical_accuracy_established=False,sd_performance_established=False,
        scope='Compiled artifact and IO compatibility only. Lower weight precision can alter logits, candidate trees and acceptance. File-size reduction does not establish a latency reduction.')
    (root/'head-w4-contract-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
