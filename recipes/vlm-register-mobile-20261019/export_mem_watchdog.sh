#!/bin/bash
# Kills the deep40 target export (then a40r) if the shared server's MemAvailable falls under 20 GiB (12 GiB), so two parallel ~70 GB exports never squeeze other users.
P1=481731; P2=481732; LOG=/tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/export_mem_watchdog.log
while kill -0 -$P1 2>/dev/null || kill -0 -$P2 2>/dev/null; do
  m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ $m -lt 20 ] && kill -0 -$P2 2>/dev/null; then echo "$(date '+%T') MemAvailable ${m} GiB < 20: killing deep40 export group $P2" >> $LOG; kill -TERM -$P2; fi
  if [ $m -lt 12 ] && kill -0 -$P1 2>/dev/null; then echo "$(date '+%T') MemAvailable ${m} GiB < 12: killing a40r export group $P1" >> $LOG; kill -TERM -$P1; fi
  echo "$(date '+%T') ${m}" > /tmp/claude-1013/-home-youngmin-dev-llm-MobiSpec-cpp/77c94dc2-d94c-47e0-9d2a-58f7bcfc791c/scratchpad/export_mem_watchdog.last; sleep 10
done
echo "$(date '+%T') both exports ended" >> $LOG
