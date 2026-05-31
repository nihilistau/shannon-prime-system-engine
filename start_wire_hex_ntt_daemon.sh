#!/system/bin/sh
# Sprint WIRE-HEX-FINISH — start the daemon with WIRE-HEX backend + NTT-attention
# overlay both env-gated ON. The 3rd cell of the headline tok/s measurement.
#
# Note: per CLOSURE-WIRE-HEX-FINISH §"Honest interpretation" the
# SP_ENGINE_NTT_ATTN_HEX flag is a no-op when (a) hex backend owns the full
# forward (gemma3_forward_hexagon bypasses math-core's NTT overlay) AND (b)
# no Memory model is loaded (daemon log: "NTT.5b: SP_ENGINE_NTT_ATTN_HEX=1
# set but no Memory model — backend disabled"). Kept for reproducibility +
# future Memory-model integration.
pkill -f sp-daemon-wire-hex 2>/dev/null
sleep 1
rm -f /data/local/tmp/wire-hex-daemon.log
export SP_ARENA=q8
export SP_DAEMON_BACKEND=hex
export SP_ENGINE_NTT_ATTN=1
export SP_ENGINE_NTT_ATTN_HEX=1
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
