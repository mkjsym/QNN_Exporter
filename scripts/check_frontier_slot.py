#!/usr/bin/env python3
"""Check physical output packing for calibration, prefill and decode shapes."""
import json
from pathlib import Path
import sys
import torch
from export_frontier_slot_w4 import pack_slot

torch.set_num_threads(4)
torch.manual_seed(260917)
checks = []
for rows in [504, 8, 1]:
    logits = torch.randn(1, rows, 151936)
    hidden = torch.randn(1, rows, 2560)
    context = torch.randn(1, 16, 2560)
    result = pack_slot(logits, hidden, context)
    assert result.shape == logits.shape
    if rows < 8:
        assert torch.equal(result, logits)
        checks.append(dict(rows=rows, unused_decode_unchanged=True))
        continue
    slot = 7
    assert torch.equal(result[:, :slot], logits[:, :slot])
    assert torch.equal(result[:, slot+1:], logits[:, slot+1:])
    count = min(rows, 7)
    assert torch.equal(result[:, slot, :count*2560], hidden[:, :count].reshape(1, -1))
    assert torch.equal(result[:, slot, 7*2560:23*2560], context.reshape(1, -1))
    assert not bool(result[:, slot, 23*2560:].count_nonzero())
    checks.append(dict(rows=rows, auxiliary_row=slot, original_other_rows_exact=True))
output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('artifacts/frontier-20260917/slot-contract-v1.json')
assert not output.exists()
output.write_text(json.dumps(checks, indent=2) + '\n')
print('SLOT_PACKING_CONTRACT_PASS', flush=True)
