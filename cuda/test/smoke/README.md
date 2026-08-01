# Release smoke test

`smoke.sh` is a binary-only sanity check: it renders one small image with an
inline `-cell` and `-default_F` (no hkl or matrix files), then asserts a clean
exit and a non-trivial output frame. It is independent of the parity harness in
`../`, which is the real correctness suite.

Run it: `./smoke.sh [path-to-nanoBraggCUDA]` (defaults to the in-repo release build).
Target a specific GPU with `CUDA_VISIBLE_DEVICES=GPU-<uuid> ./smoke.sh`.
