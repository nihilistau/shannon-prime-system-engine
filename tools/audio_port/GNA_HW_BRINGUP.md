# KAI-3 Stage 2.b — GNA_HW Silicon Bring-Up Checklist

The POT-quantized i16 IR scores **0.877 (== FP32)** on `GNA_SW_EXACT` (software integer emulation).
This is the procedure to run that *same IR* on the **physical GNA 2.0 silicon** of the Beast Canyon NUC
(i9-11900KB). If `GNA_HW` matches the 0.877 emulation number, the EAR front-end is physically realized.

**Why native Windows (not WSL):** `GNA_SW_EXACT` is CPU-bound emulation, so it ran inside WSL. `GNA_HW`
needs direct MMIO register access to the accelerator — WSL2 has no GNA passthrough. The scoring script must
run on the **Windows host** against a **Windows** OpenVINO 2023.3 runtime (the Linux build can't touch the
Windows driver).

---

## Step 1 — BIOS / firmware floor  *(operator — physical)*
1. Reboot the NUC, enter BIOS (F2 / Del during POST).
2. Find **Intel Gaussian & Neural Accelerator** (under *System Agent* / *CPU Configuration* /
   *Onboard Devices* depending on BIOS rev).
3. Set it to **Enabled**. Save + exit. (If disabled at firmware level, the Windows driver won't initialize.)

## Step 2 — Windows driver install  *(operator — admin)*
1. Driver package: `D:\F\shannon-prime-repos\archive\notes_and_stuff\GNA\Drivers\gna_03.05.00.2116.zip`
   — unzip it.
2. Right-click the `.inf` → **Install** (or run the bundled installer). Admin elevation required.
3. **Device Manager → System devices →** verify **Intel Gaussian and Neural Accelerator** is present,
   active, **no error code (status "working properly")**.

## Step 3 — Windows OpenVINO 2023.3 runtime  *(Claude stages this)*
- Downloaded + extracted to `D:\F\shannon-prime-repos\_xbar\p2b\kai3\ov2023_win\` (the GNA-capable LTS;
  the host's only other OV, 2025.4, dropped the GNA plugin).
- Native Windows Python **3.11.9** (`C:\Users\Knack\AppData\Local\Programs\Python\Python311`) — matches the
  archive's py3.11 bindings.
- Env wired by `setupvars.bat` in the extracted toolkit (sets PATH to `runtime\bin\intel64\Release` +
  PYTHONPATH to the bindings). `openvino_intel_gna_plugin.dll` + `gna.dll` ship in that bin dir.

## GNA conv constraints — FIXED + software-validated (2026-06-17)
The Windows GNA compiler is stricter than the Linux SW build; two hard constraints surfaced and were fixed
at **zero recovery cost** (FP32 0.877 held through both):
1. **"Padding isn't supported by GNA"** — the encoder's `padding=1` (SAME) convs → **`padding=0` (VALID)**.
   Trained weights transfer directly (padding isn't a weight); CTC is alignment-free so the ~6-frame shrink
   is harmless. FP32 recovery unchanged at 0.877.
2. **"Unsupported number of filters: 33"** — GNA requires conv output channels to be a **multiple of 4**.
   The CTC head (V+1=33) → **HEAD=36** (3 dummy channels), trained weights loaded into [:33], dummies sliced
   off before the argmax. Recovery unchanged.
The GNA-legal IR (`ov_work_valid/pot/audio_ctc_pot_gna.xml`) now **compiles + runs on the WINDOWS GNA plugin**:
`GNA_SW_EXACT i16 = 0.877` (== FP32). Only `GNA_HW` (driver + silicon) is unproven.

## Step 4 — Run the GNA_HW gate  *(Claude runs when you confirm silicon is live)*
```
run_gna_hw.bat
```
which (a) sources the Windows toolkit `setupvars.bat`, (b) runs
`ov_score_ir.py --ir ov_work\pot\audio_ctc_pot_gna.xml --mode GNA_HW`.
**Expected:** `Core().available_devices` now includes **GNA** (only after Steps 1-2), and the
`GNA_HW i16` recovery should land at **0.877** to match emulation.

## Gate
- **PASS:** GNA_HW recovery ≈ 0.877 (within token-granularity of emulation) → EAR front-end physically
  realized on GNA 2.0 → Stage 2.b CLOSED on silicon.
- **DEVIATION:** if HW ≠ SW_EXACT, the delta isolates a driver/firmware scale-factor or layout quirk
  (HW path vs the SW integer reference) — diagnosable against the SW_EXACT golden.

Receipt target: `G-KAIROS-3-GNA-HW.log`.
