#!/bin/bash
cd /mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
setsid bash _run_s01.sh > /tmp/kai3_s01.log 2>&1 < /dev/null &
disown
echo "launched pid=$! (setsid-detached)"
sleep 1; echo "log head:"; head -c 200 /tmp/kai3_s01.log 2>/dev/null
