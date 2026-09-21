# Preserved comparison snapshots

The source trees in this directory are preserved for paper reproduction and
are excluded from the `flashbob` wheel.

- `flashback/` is the FlashBack snapshot evaluated at commit `62a0e0e` from
  [lengstrom/flashback](https://github.com/lengstrom/flashback).
- `hvp_baselines/` contains the GradMem-derived HVP snapshot evaluated at
  commit `3f40f11`. The original checkout did not record a complete upstream
  repository URL.

The snapshot files are kept unchanged. Project-owned HVP call adapters live in
`benchmarks/baselines/__init__.py`; FlashBack is invoked through its preserved
`bench.py` entry point.
