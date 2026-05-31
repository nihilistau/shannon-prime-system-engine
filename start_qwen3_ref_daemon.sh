#!/system/bin/sh
pkill -f sp-daemon-wire-hex 2>/dev/null
sleep 1
rm -f /data/local/tmp/wire-hex-daemon.log
export SP_ARENA=q8
unset SP_DAEMON_BACKEND
export RUST_LOG=info
cd /data/local/tmp/sp22u
nohup ./sp-daemon-wire-hex --daemon-inner \
  --model /data/local/tmp/qwen3_rt.sp-model \
  --tokenizer /data/local/tmp/qwen3_rt.sp-tokenizer \
  --port 8087 --quic-port 0 \
  --memo-model '' --memo-tokenizer '' \
  --draft-model '' --draft-tokenizer '' \
  --pouw-ledger-path '' --peer '' --peers '' \
  > /data/local/tmp/wire-hex-daemon.log 2>&1 &
echo "started PID $!"
