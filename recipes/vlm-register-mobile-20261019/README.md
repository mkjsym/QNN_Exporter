# Recipes: Qwen3-VL-4B targets, register drafter graphs and vision fixtures for the S26 register study (2026-10-19 .. 10-23)

These are the exact launchers that built the QNN models for the mobile rounds M0-M7 of the register study
(RegSpec, `docs/research/vlm-register-dyn-20261016/mobile-plan-20261019.md`). They are kept verbatim as records:
paths are those of the original workspace (`/home/youngmin/dev/llm/{QNN_Exporter,MobiSpec.cpp,executorch}`,
QAIRT at `/home/youngmin/qairt/2.37.1-v81`, a session scratch directory for lists and logs). To rerun one, set up
`env.sh` (see `env.sh.example` and `docs/export-guide.md`), apply `patch/` to the ExecuTorch checkout, and replace
those prefixes with your own.

Outputs land in `artifacts/vlm-register-mobile-20261019/` (not committed). The RegSpec harness
`tools/research/run_vlm_register_mobile.py` reads them from `../QNN_Exporter/artifacts/...`, so check this repository
out next to the RegSpec checkout.

## Order and purpose

| step | recipe | builds | notes |
|---|---|---|---|
| 0 | `probe_export_c2048.sh` | one target at context 2048 | wall time / peak RAM probe before the 4096 exports |
| 1 | `fixtures16.sh` (`fixtures16.list`) | 16 calibration fixtures (vision rows of 16 document pages, `QNN_VLM_DUMP`) | host `llama-qnn-vlm-prepare` from MobiSpec.cpp; same pages as the drafter calibration |
| 2 | `export_targets_c4096.sh` | targets a32 (ar_len 32) and a40 (ar_len 40), context 4096 | M0/M2 targets, `scripts/export_vlm_target.py` |
| 3 | `export_target_deep_c4096.sh`, `export_target_deep_v2.sh`, `export_target_deep_v3.sh` | deep-register target (per-layer prompts baked in, `vlm_prompt_slot` input) | v2/v3 calibrate with the deep arm's register rows (`--calib-registers`); v3 = v2 under `nothp.py` |
| 4 | `fixtures_m1.sh`, `fixtures_m2.sh`, `launch_m1_fixture.sh` | per-page vision fixtures for the device runs (`fixtures-m1/`) | the harness loads them with `QNN_VLM_LOAD` instead of running vision on the phone |
| 5 | `export_target_a32_r1.sh` | a32r: a32 recalibrated with the dynreg8 register rows in every fixture | M6/M7 B protocol |
| 6 | `export_target_a40_r1.sh`, `export_target_deep_a40.sh` (+ `export_mem_watchdog.sh`) | a40r and deep40 | M7 A protocol; the watchdog stops an export when the shared host's free memory runs low |
| 6b | `export_target_a40_s1.sh` | a40s: the a40 graph calibrated in 32-row chunks (`--calib-ar 32`); its IO encodings are bit-identical to a32r | tests whether a32r and a40 differ through encodings or through the 40-row graph itself |
| 6c | `export_target_deep_s1.sh` | deep32s: the deep-register target with a32r's encodings imported (`--calib-common-registers`); each calibration fixture runs twice (a32r's rows with zero prompt slots, then the deep rows) and only the 67 observers on the prompt path keep the second pass. IO encodings bit-identical to a32r on all 354 shared tensors; device AR tokens equal AR-a32r on 5/5 rep5 pages | puts Deep-Dynreg8 on the same text as the plain arms (mobile plan 16.5) |

Recipe 6c attempt logs: attempts 1 and 2 both stopped with "no observer on the prompt path" (attempt 1 before the diagnostic; attempt 2 matched observers by torch.ao class, but the graph holds torchao observer classes, so it matched none). Attempt 3 matches by the `activation_post_process_` module-name prefix and is the one that built deep32s. The message of c60d234 says attempt 2 "matched by class"; it matched nothing.
| 7 | `rebuild_inject_dynreg8_norm8.sh`, `export_inject_dynreg8_norm8.sh` | inject-only 256-row draft-prefill graphs (`ctx-<arm>-inject256`) for dynreg8 and norm8 | none / norm32 / full / deep inject graphs came from the M4 queue (`scripts/queue_vlm_register_mobile_export.py`) |
| - | `verify_restore_fixtures.sh` | - | checks fixtures restored from the phone after a server cleanup (count, size, sha256) |
| - | `nothp.py` | - | runs a command with transparent huge pages disabled (an export once starved in kernel compaction) |

Each export takes 35-95 minutes and 60-72 GB of host RAM (see the `/usr/bin/time -v` lines in the logs).

## Logs

`logs/` holds the stdout/stderr of every export, fixture and audit run listed above, including failed attempts
(`*-attempt1-*`). Logs over 1 MB are gzip-compressed (`*.log.gz`).
