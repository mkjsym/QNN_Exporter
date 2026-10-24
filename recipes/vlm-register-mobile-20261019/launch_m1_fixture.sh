#!/bin/bash
# M1 with precomputed vision: wait for the rebuilt binaries (QNN_VLM_LOAD path) and the 50 host fixtures, then stage -> smoke -> run.
S=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad
LOG=$S/m1_fixture.log
until grep -q 'ANDROID_BUILD_EXIT 0' $S/build_android_load.log 2>/dev/null && grep -q 'FIXTURES_M1_DONE' $S/fixtures_m1.log 2>/dev/null; do
  grep -q 'ANDROID_BUILD_EXIT [1-9]' $S/build_android_load.log 2>/dev/null && { echo M1F_BUILD_FAILED >> $LOG; exit 1; }
  sleep 60
done
cd /home/youngmin/dev/llm/MobiSpec.cpp
echo "$(date '+%H:%M') prepare" >> $LOG
python3 tools/research/run_vlm_register_mobile.py prepare >> $LOG 2>&1 || { echo M1F_PREPARE_FAILED >> $LOG; exit 1; }
echo "$(date '+%H:%M') smoke" >> $LOG
python3 tools/research/run_vlm_register_mobile.py smoke >> $LOG 2>&1 || { echo M1F_SMOKE_FAILED >> $LOG; exit 1; }
echo "$(date '+%H:%M') run" >> $LOG
python3 tools/research/run_vlm_register_mobile.py run --repeats 2 >> $LOG 2>&1 && echo M1F_RUN_OK >> $LOG || echo M1F_RUN_FAILED >> $LOG
