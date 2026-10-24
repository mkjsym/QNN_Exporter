"""Record the explicit runtime contract for an audited masked-position DFlash graph."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--context',type=Path,required=True)
    parser.add_argument('--asset',type=Path,required=True)
    args=parser.parse_args()
    params=json.loads((args.context/'params.json').read_text())
    manifest=json.loads((args.asset/'manifest.json').read_text())
    audit=json.loads((args.asset/'fp32-audit.json').read_text())
    assert audit['passed'] and audit['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert (params['draft_physical_block'],params['draft_logical_block'],params['draft_inject_rows'])==(8,7,16)
    assert params['draft_output_kind']=='logits' and params['draft_output_width']==151936
    record=dict(prediction='masked_position',root_row=0,first_proposal_row=1,proposal_rows=7,visible_block_rows=8,
        injection_rows=16,required_mobispec_environment={'DFLASH_MASKED_ROW_OUTPUT':'1'},
        allowed_policy='greedy plain DFlash chain or DDTree <=31 candidates',
        checkpoint_sha256=manifest['checkpoint_sha256'],
        context_sha256=digest(args.context/'forward_0.bin'),params_sha256=digest(args.context/'params.json'),
        fp32_audit_sha256=digest(args.asset/'fp32-audit.json'),source_sha256=digest(Path(__file__)),
        note='Legacy row0/open7 and DFlash2 selector mode are not compatible with this checkpoint.')
    destination=args.context/'runtime-contract.json'
    if destination.exists():assert json.loads(destination.read_text())==record
    else:destination.write_text(json.dumps(record,indent=2)+'\n')
    print(destination)


if __name__=='__main__':main()
