# Morning status — the system is running. Read this first.

You went to bed thinking it might be over. It isn't. The core works. What broke tonight was
**operational, not the KEYSTONE** — leftover processes and two missing one-liners. I cleaned it
up, measured it, and left one clean daemon running. Here is the honest picture.

## It's running right now
A single plain daemon is up on **http://127.0.0.1:3000/**. Just open it (hard-refresh:
Ctrl+Shift+R). You should get coherent, fast chat that knows who it is and remembers the
conversation. Measured, just now, on your 2060:

```
T1  4.7s  "Shannon-Prime"                                       <- correct identity (system prompt)
T2  2.3s  "The ocean covers 71% of the Earth's surface."        <- coherent, stops cleanly
T3  2.1s  "The highest mountain in the world is Mount Everest."
T4  2.6s  "The Amazon River is the largest river in the world by volume of water."
T5  3.2s  "We talked about the ocean, mountains, and rivers."   <- it remembered the chat
```
~2-3 seconds a turn, 12-18 tok/s decode. That's healthy for a 12B on a 2060.

## What actually went wrong tonight (none of it was the core)
1. **"slow af" = stacked daemons.** Multiple sp-daemon processes were left running and the GPU
   was at **11966/12288 MiB (97% full)** — they were fighting over VRAM and thrashing. Killing the
   extras dropped it to 450 MiB. One daemon = fast again.
2. **"Failed to fetch" = wrong port.** Earlier in the week the console was pointed at the agent
   gateway (:8800), which wasn't running. The daemon on :3000 was fine the whole time. Fixed: the
   console now talks to whatever daemon served it.
3. **glyph-spam degeneration = missing EOT default.** `run_console.bat` never set `SP_EOT_BIAS`, so
   any client that didn't send it would ramble into repeated symbols until max_tokens (which *also*
   looks like "slow"). Fixed: the launchers set `SP_EOT_BIAS=4.0` by default now.

## What I changed (all committed + pushed)
- Both launchers now `taskkill` any old sp-daemon first, so **launches can never stack again** —
  this is the fix that prevents tonight's main symptom from recurring.
- `SP_EOT_BIAS=4.0` default in both launchers (clean stopping for every client).
- Console defaults chat to the serving daemon (:3000); the gateway is opt-in.
- `SP_AUTO_RECALL_DEFAULT=1` in the recall launcher so the librarian fires for any client.
- Persistent O(1) KV (earlier tonight) is wired in and byte-exact gated.

## The three ways to run it (pick one — only ever ONE daemon)
- **`run_console.bat`** — the daily driver. Solid chat + conversational memory + persist + clean
  stops. **This is what's running now.** Start here.
- **`run_console_recall.bat`** — adds the W_c "librarian" (autonomous long-term recall of stored
  needles). It fires and it foreign-rejects correctly. **Honest caveat:** recall accuracy is
  phrasing-sensitive out of distribution — on an ad-hoc query tonight it picked the wrong needle and
  the model confabulated a code. That's the known hard research problem (the deploy gate was 360/361
  on in-distribution queries), not a regression. Use it to *demo* recall, not to rely on it yet.
- **`run_gateway.bat`** + set `localStorage.sp_chat_endpoint = "http://127.0.0.1:8800"` — the agent
  path where the model calls tools (memory / python / web).

## If it ever feels slow again
Open Task Manager (or `nvidia-smi`) and look for more than one `sp-daemon.exe`. That's the tell.
The launchers now prevent it, but if you start things by hand, kill extras — one model fills the GPU.

## The honest state of the project
The hard parts are real and they work: byte-exact 12B forward, coherent served chat, persistent
O(1) KV, conversational memory, the memory-agency stack (store/forget/decide/merge), the tool
harness. The genuinely-unsolved frontier is the same one it's always been — **reliable autonomous
recall selection** (which stored memory to pull, on arbitrary phrasing). That's a research problem,
not a plumbing bug, and it's worth being patient with. Everything around it is solid.

It's not over. It's just late, and the desk got messy. Sleep well — it'll be here, running, when
you wake up.
