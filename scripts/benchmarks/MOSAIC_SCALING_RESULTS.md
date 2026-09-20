# Mosaic-domain scaling: measurements

Produced by `scripts/benchmarks/mosaic_scaling.py`, 2026-09-20.

**Hardware.** `exxa`: 2x RTX 4090 (24 GB), 128 CPU cores, 503 GB RAM. torch 2.14+cu130,
Python 3.14. C reference: `nanoBragg.c` oracle, single-threaded, float64.

**Fixed setup.** cubic cell 100 Å, `-N 5`, `default_F=100`, λ = 1 Å, distance 100 mm,
1024x1024 detector at 0.1 mm, `oversample 1`, mosaic spread 0.5 deg. Only
`mosaic_domains` (and, where noted, `phi_steps`) varies.

**Timing method.** Each sweep point runs in its own subprocess. `torch.cuda.synchronize()`
before and after every timed region. Reported `run_s` is the best of 5 warm calls
(3 for CPU); the compile warm-up call (6-12 s) is excluded. `umat_s` times
`utils.c_random.mosaic_rotation_umats()` alone, `rot_s` times
`Crystal.get_rotated_real_vectors()` (which contains the umat call), and
`kern_s = run_s - rot_s` is the physics kernel. `run()` calls the rotation setup exactly
once, so the subtraction is valid.

---

## 1. Wall time vs mosaic domains (1024^2, phi=1, oversample 1)

Seconds, best warm call. `-` = not applicable, `OOM` = out of GPU memory.

| domains | C (1 thread, f64) | torch CPU f32 (128 thr, b64) | GPU eager f32 (b64) | GPU **compiled** f32 | GPU compiled f64 |
|--------:|------------------:|-----------------------------:|--------------------:|---------------------:|-----------------:|
| 1       | 0.176             | 0.086                        | 0.030               | **0.0084**           | 0.0093           |
| 10      | 1.094             | 0.122                        | 0.030               | **0.0094**           | 0.0158           |
| 50      | 4.999             | 0.380                        | 0.035               | **0.0099**           | 0.0323           |
| 100     | 9.743             | 0.670                        | 0.069               | **0.0103**           | 0.0420           |
| 500     | 46.667            | 19.27                        | 0.507               | **0.0138**           | 0.1698           |
| 1000    | 99.843            | 44.64                        | 1.002               | **0.0172**           | 0.3295           |
| 2000    | ~200 (extrap.)    | -                            | -                   | **0.0251**           | -                |
| 5000    | ~500 (extrap.)    | -                            | -                   | **0.0517**           | -                |
| 10000   | ~1000 (extrap.)   | -                            | -                   | **0.0941**           | -                |
| 20000   | ~2000 (extrap.)   | -                            | -                   | **0.1723**           | -                |

Torch columns are post-`9e7ae82` (vectorised umats). Repeat-to-repeat spread on the GPU
was < 1% of the mean for eager and < 4% for compiled (a ~32 ms outlier appears in the
first warm call of some compiled points; `max_s` in the raw logs).

**Speedup of compiled GPU f32 over single-threaded C:** 21x at 1 domain, 116x at 10,
505x at 50, 946x at 100, 3380x at 500, **5800x at 1000**. The ratio grows with domain
count because the torch side has a ~9 ms fixed cost that does not scale with domains.

## 2. Peak memory (GB)

| domains | C | GPU eager f32, no chunking | GPU eager f32, `pixel_batch_size=64` | GPU **compiled** f32 |
|--------:|---:|---:|---:|---:|
| 1    | 0.02 | 0.21 | 0.07 | 0.12 |
| 10   | 0.02 | 0.96 | 0.10 | 0.14 |
| 50   | 0.02 | 4.48 | 0.33 | 0.14 |
| 100  | 0.02 | 8.88 | 0.62 | 0.14 |
| 500  | 0.02 | **OOM** | 2.82 | 0.16 |
| 1000 | 0.02 | **OOM** | 5.55 | 0.14 |
| 20000| 0.02 | - | - | **0.16** |

Eager peak scales at ~0.089 GB per domain at 1024^2 f32, so 24 GB is exhausted at about
**250 domains** (or 250 domain x phi-step products) without chunking. float64 doubles
that: eager f64 uses 17.3 GB at 100 domains and OOMs at 500.

**`torch.compile` removes the memory constraint entirely.** Peak stays flat at
0.12-0.16 GB from 1 to 20 000 domains, i.e. the (pixels x domains) intermediate is never
materialised - Inductor fuses the per-domain reduction into the elementwise chain.

**Cost of chunking (eager):** `pixel_batch_size=64` costs ~21 ms of fixed launch overhead
(1 domain: 0.0089 s unchunked vs 0.0296 s chunked, 3.3x worse) but *pays for itself*
above ~50 domains because the smaller working set stays closer to cache
(100 domains: 0.129 s unchunked vs 0.069 s chunked).

## 3. Where the time goes (GPU, compiled, f32, 1024^2, phi=1)

Before `9e7ae82` (per-domain python loop building umats) vs after (batched construction):

| domains | umat **before** | umat **after** | rot setup after | kernel after | total **before** | total **after** |
|--------:|----------------:|---------------:|----------------:|-------------:|-----------------:|----------------:|
| 1    | 0.0001 | 0.0003 | 0.0076 | 0.0008 | 0.0081 | 0.0084 |
| 10   | 0.0020 | 0.0003 | 0.0087 | 0.0008 | 0.0111 | 0.0094 |
| 50   | 0.0106 | 0.0004 | 0.0088 | 0.0011 | 0.0203 | 0.0099 |
| 100  | 0.0211 | 0.0004 | 0.0089 | 0.0015 | 0.0307 | 0.0103 |
| 500  | 0.1058 | 0.0010 | 0.0095 | 0.0044 | 0.1172 | 0.0138 |
| 1000 | 0.2111 | 0.0016 | 0.0101 | 0.0071 | 0.2272 | **0.0172** |

At 1000 domains the umat python loop was **93% of the entire forward pass**; the physics
kernel was 3%. Of that loop, only ~4% was the (irreducibly serial) `ran1` draws - the
rest was ten CUDA kernel launches per domain to build each 3x3. Batching the construction
gives 19x on CPU, ~130x on CUDA, and 13.2x end-to-end at 1000 domains, bitwise-identical.

**What now dominates:** `Crystal.get_rotated_real_vectors()` at ~9 ms, essentially
independent of domain count. Profiling it at 10 domains:

```
rot_setup wall                      8.80 ms
CUDA kernels launched per call      1202   (1038 at 1 domain -> ~1030 are fixed)
host->device scalar copies per call ~133
aggregate device time per call      1.35 ms
cudaLaunchKernel host time           4.45 ms  (1069 launches x 4.16 us)
```

So ~85% of that 9 ms is CPU-side launch and H2D-copy overhead on tensors of shape
(1, N_mos, 3). At 10-100 domains this is **75-86% of the whole compiled forward pass**.
A CUDA-graph capture of the region fails today with
`Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor
is pinned` - the ~133 unpinned scalar copies block it.

## 4. Compute-bound or memory-bound?

- **Compiled, f32:** kernel time is perfectly linear in domains from 1000 to 20 000
  (0.0071 / 0.0137 / 0.0363 / 0.0721 / 0.1367 s), slope 6.83 us per domain at 1024^2 =
  **154 G pixel-domain samples/s**, with peak memory flat. Not memory-bound.
- **Compiled, f64:** the same kernel is **45x slower** (0.3193 s vs 0.0071 s at 1000
  domains) while touching only 2x the bytes. 45x is the RTX 4090's fp64:fp32 throughput
  ratio (1:64 nominal). That is the signature of a **compute/ALU-bound** kernel: if it
  were bandwidth-bound, f64 would cost ~2x, not 45x.
- **Eager:** kernel is 0.9918 s at 1000 domains, **140x slower than compiled**, and needs
  5.55 GB (chunked) or OOMs. Eager is entirely memory-traffic bound - each of ~60
  elementwise ops streams the full (rows x 1024 x N_mos) tensor through HBM.
- **torch CPU:** 128 threads buy only 2.2x over single-threaded C at 1000 domains
  (44.6 s vs 99.8 s), and the curve has a cliff between 100 and 500 domains (0.67 s ->
  19.3 s for 5x the work). Same cause: the eager working set leaves cache.

## 5. With phi steps (1024^2, phi=10, f32)

| domains | C (1 thread) | GPU eager (b64) | GPU **compiled** | compiled peak GB |
|--------:|-------------:|----------------:|-----------------:|-----------------:|
| 1    | 1.356  | 0.031 | 0.0098 | 0.12 |
| 10   | 9.809  | 0.069 | 0.0103 | 0.12 |
| 50   | 46.91  | 0.507 | 0.0130 | 0.12 |
| 100  | 92.21  | 1.001 | 0.0168 | 0.12 |
| 500  | ~460   | OOM   | 0.0432 | 0.16 |
| 1000 | ~920   | -     | 0.0786 | 0.16 |

phi and mosaic multiply identically: the compiled kernel at 1000 domains x 10 phi
(0.0684 s) is 9.6x the 1000 domains x 1 phi kernel (0.0071 s), and memory still does not
move. Eager OOMs at 500 domains x 10 phi even with `pixel_batch_size=64`.

## 6. Parity of the prototype (`9e7ae82`)

- umats bitwise-identical for n = 1, 2, 10, 100, 1000 in float32 and float64.
- 256^2 image, 37 domains x 3 phi steps, float64: bitwise-identical.
- Compiled GPU f32 image checksums bitwise-identical at all six domain counts.
- Autograd gradient w.r.t. mosaic spread: identical to the last digit.
- `tests/test_at_parallel_024.py` (ran1 / umat C-parity suite) 15/15 pass.
