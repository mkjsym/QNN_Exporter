# Recipes: RegSpec drafters at 1,000 steps (seed 43) and their deep targets for the S26 (2026-11-09 .. 11-10)

These launchers exported a second set of RegSpec drafters for the S26 device runs. The drafters were trained on doc2 for
1,000 steps with seed 43, in BF16, and handed over as a bundle (`drafters/<arm>/model.safetensors`, `complete.json`).
The recipe is the one in [vlm-register-mobile-20261019](../vlm-register-mobile-20261019/README.md); only the checkpoints and
the target widths are new. The launchers are kept verbatim as records, with the original workspace paths. The outputs are
in `/data/youngmin/qnn-artifacts/regspec-s26-20261109` (`OUT` in `scripts/run_regspec_exports.py`), and nothing there is
committed.

## Arms

| arm | drafter step graph | draft-prefill graph | target |
|---|---|---|---|
| `full-s1000-seed43` | `ctx-full-s1000-seed43-c4096` (context 4,096) | `ctx-full-s1000-seed43-inject256` | existing a32r |
| `dynreg8-deep-taps-s1000-seed43` | `ctx-dynreg8-deep-taps-s1000-seed43` (context 1,024) | `...-inject256` | new deep target, 40 verify rows (a40) |
| `dynreg8-deep-taps-copy024-s1000-seed43` | `ctx-dynreg8-deep-taps-copy024-s1000-seed43` (context 1,024) | `...-inject256` | new deep target, 56 verify rows (a56) |

`none-s1000-seed43` and `dynreg8-s1000-seed43` are registered in `scripts/vlm_register_mobile_export_common.py` but were
not exported.

Why each deep drafter needs its own target: `--layer-prompts` bakes that drafter's per-layer register prompts into the
graph, so the earlier deep targets cannot be reused. The verify width is the 31-node tree plus its root (32 rows), plus
the register rows: 8 rows for `dynreg8-deep-taps`, and 3 x 8 for `copy024`, which keeps register copies at depths 0, 2
and 4.

## Order

| step | command | builds |
|---|---|---|
| 0 | `prepare_regspec_inputs.py` | frozen device inputs (rep5 and the bundle's ood10) and their host vision fixtures |
| 0b | `prepare_regspec_mobile_calibration.py --bundle <bundle> --output <dir>` | register taps of the new drafters on the 16 calibration pages of the 10-19 recipe |
| 1 | `run_regspec_exports.py drafts` | calibration, checkpoint conversion, the step graph and the inject-256 graph of each arm |
| 2 | `run_regspec_exports.py targets` | the a40 and a56 deep targets (`export_vlm_target.py --method dflash-deep --context 4096`) |
| 3 | `repair_regspec_target_threads.py` | the same two targets again with 8 calibration threads (`*-t8`), each followed by an encoding check |
| - | `finish_regspec_targets.py --adopt-pid <pid> --original-state <json>` | adopts a target export that is already running and finishes both widths without restarting PTQ |

The targets are calibrated with:
- `--calib-registers`: the arm's own register rows.
- `--calib-common-registers`: the rows of the 10-19 `dynreg8-20k-seed42` export.
- `--calib-ar 32`.
- The 16 fixtures of `fixtures16.list`.

This makes the IO tensors they share with a32r carry a32r's encodings, so one AR reference serves every method.

The launchers read the bundle and calibration through `REGISTER_MOBILE_STUDY`, `REGISTER_MOBILE_CALIB` and
`REGISTER_MOBILE_EXPORT_OUT`. `prepare_vlm_register_mobile_export.py` takes `<study>/<arm>/model.safetensors` when it
exists (the bundle layout) and otherwise the checkpoint path in `complete.json`; the sha256 check is the same either way.

## Calibration threads must match the baseline

The first target exports ran PTQ with `OMP_NUM_THREADS=MKL_NUM_THREADS=6`; the 10-19 recipe used 8.
- **Encodings:** 246 of the 354 IO tensors shared with a32r came out with different scales: scale only, with a maximum
  relative difference of 4.96e-6, and identical offsets (`logs/encoding-check-steering40-initial.log`,
  `logs/encoding-mismatch-diagnosis.json`).
- **Outputs:** that was enough to move the outputs. On `doc2-infographic-00-1024` the AR outputs of the new targets left
  a32r's after 26 tokens.
- **Repair:** `repair_regspec_target_threads.py` re-exported only the two deep targets with 8 threads (`*-a40-t8`,
  `*-a56-t8`). Both match a32r on all 354 shared tensors, with a maximum difference of 0 (`logs/encoding-check-*-t8.log`).

`export_vlm_target.py` now records the calibration thread counts in `meta.json` (`calibration_threads`). Before reusing a
target's encodings, check that its thread count matches.

## Logs

`logs/` has the queue logs of both phases and of the repair, the calibration and fixture logs, the graph audits of the six
drafter graphs, and the encoding checks. The full target export logs, about 10 MB each, are left out.
