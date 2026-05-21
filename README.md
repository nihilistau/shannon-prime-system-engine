# shannon-prime-system-engine

Clean from-scratch inference engine for the [shannon-prime-lattice](../shannon-prime-lattice) architecture. Uses [shannon-prime-system](../shannon-prime-system) for the math primitives.

Provides (when complete):

- GGUF model loader (small target models first — Qwen 0.5B, Gemma 0.3B class)
- Forward pass with NTT-based attention via the math core's CRT polynomial multiplication
- KSTE-encoded KV state (Phase 5+) feeding the Friedman sieve / ARM aggregation in `shannon-prime-lattice`
- Two-node sharded inference demo (Phase 6) — each node runs one Proth prime branch, driver CRT-recombines residues

## Not a fork

This is **not** a fork or extension of the older `shannon-prime-engine/` repo. Clean rebuild, only the primitives used by `shannon-prime-lattice`. The architecture is informed by the prior engine's measurements but no source is copied.

Phasing in `../shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` defines what gets built when.

## Status

**Phase 0 — empty.** README + LICENSE + .gitignore + CMake stub in place. Phase 5 lands the engine bootstrap; Phase 6 lands two-node sharded inference.

## Build (placeholder)

```bash
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```

With CUDA:

```bash
cmake -B build-cuda -G Ninja -DSP_ENGINE_WITH_CUDA=ON
cmake --build build-cuda
```

## License

AGPL-3.0-or-later. See `LICENSE`. Commercial licensing available — contact the copyright holder.
