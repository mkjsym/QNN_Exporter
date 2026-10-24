"""Preserve the actual post-PTQ graph for independent runtime verification."""
import sys,json
from pathlib import Path
import torch
from export_draft_head_w4 import sha
kind=sys.argv.pop(1)
assert kind in ['dspark','domino']
if kind=='dspark':import export_dspark8b_head_w4 as exporter
else:import export_domino8b_head_w4 as exporter
out=Path(sys.argv[sys.argv.index('--artifact')+1]).resolve()
original=exporter.llm_wrappers.to_edge_transform_and_lower_to_qnn
def freeze(**kwargs):
    module=kwargs['module']['prefill_forward'];inputs=kwargs['inputs']['prefill_forward']
    ep=torch.export.export(module,inputs,strict=True)
    path=out/'prefill-frozen.pt2';torch.export.save(ep,path)
    (out/'frozen.json').write_text(json.dumps(dict(kind=kind,sha256=sha(path),source_sha256=sha(Path(__file__)),
        inputs=[dict(shape=list(v.shape),dtype=str(v.dtype)) for v in inputs]),indent=2)+'\n')
    del ep
    print('8B_POST_PTQ_FROZEN',kind,flush=True)
    return original(**kwargs)
exporter.llm_wrappers.to_edge_transform_and_lower_to_qnn=freeze
exporter.main()
