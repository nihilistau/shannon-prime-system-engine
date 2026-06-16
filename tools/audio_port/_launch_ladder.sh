#!/bin/bash
cd /mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
setsid bash _run_ladder.sh > /tmp/kai3_ladder.log 2>&1 < /dev/null &
disown
echo "ladder launched pid=$! (setsid-detached)"
