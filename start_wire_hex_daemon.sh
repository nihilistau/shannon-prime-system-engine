#!/system/bin/sh
# Sprint WIRE-HEX — start the daemon with hex backend on the S22U.
# Push to /data/local/tmp via adb push, then `adb shell sh /data/local/tmp/start_wire_hex_daemon.sh`.
pkill -f sp-daemon-wire-hex 2>/dev/null
sleep 1
rm -f /data/local/tmp/wire-hex-daemon.log
export SP_ARENA=q8
export SP_DAEMON_BACKEND=hex
export ADSP_LIBRARY_PATH=/data/local/tmp/sp22u
export RUST_LOG=info
cd /data/local/tmp/sp22u
nohup ./sp-daemon-wire-hex --daemon-inner \
  --model /data/local/tmp/gemma3-1b.sp-model \
  --tokenizer /data/local/tmp/gemma3-1b.sp-tokenizer \
  --port 8087 --quic-port 0 \
  --memo-model '' --memo-tokenizer '' \
  --draft-model '' --draft-tokenizer '' \
  --pouw-ledger-path '' --peer '' --peers '' \
  > /data/local/tmp/wire-hex-daemon.log 2>&1 &
echo "started PID $!"
