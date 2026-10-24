#!/bin/bash
# Fair single table (user 2026-10-23 10:30): every SD arm prefills the drafter through its 256-row inject-only graph (iso-implementation, plan 11/12).
# dynreg8-20k-seed42 and norm8-20k-seed42 had no inject256 graph (M4 exported none/norm32/full/deep only). Same queue as M4 (plan 11.2).
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
cd /home/youngmin/dev/llm/QNN_Exporter
echo "$(date '+%H:%M') START inject256 exports: dynreg8-20k-seed42 norm8-20k-seed42"
/home/youngmin/anaconda3/envs/etqnn/bin/python scripts/queue_vlm_register_mobile_export.py --inject 256 dynreg8-20k-seed42 norm8-20k-seed42 && echo "$(date '+%H:%M') INJECT_EXPORT_OK" || echo "$(date '+%H:%M') INJECT_EXPORT_FAILED"
