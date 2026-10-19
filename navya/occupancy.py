"""
p1/occupancy.py — theoretical occupancy for a kernel launch.

Occupancy = active warps per SM / maximum warps per SM. It is limited by whichever runs out first:
threads per SM, blocks per SM, registers per SM, or shared memory per SM.

Two methods:
  "driver"  — asks the CUDA driver (cuOccupancyMaxActiveBlocksPerMultiprocessor); exact.
  "formula" — the CUDA Occupancy Calculator's rules, simplified; used if the driver call isn't available.
The paper should say which method produced each number (it is written to kernel_bench.csv).
"""
import math

# Allocation granularities by compute-capability major version (simplified).
#   reg_unit: registers are allocated per warp in units of this many registers
#   smem_unit: shared memory allocated per block in units of this many bytes
#   smem_reserved: bytes the system reserves per block (sm_80+)
GRANULARITY = {6: dict(reg_unit=256, smem_unit=256, smem_reserved=0),
               7: dict(reg_unit=256, smem_unit=256, smem_reserved=0),
               8: dict(reg_unit=256, smem_unit=128, smem_reserved=1024),
               9: dict(reg_unit=256, smem_unit=128, smem_reserved=1024)}


def formula_occupancy(block, regs_per_thread, smem_per_block, limits, cc_major, warp=32):
    g = GRANULARITY.get(cc_major, GRANULARITY[7])
    max_warps_sm = limits["max_threads_sm"] // warp
    warps_per_block = math.ceil(block / warp)

    lim = {}
    lim["threads/warps"] = max_warps_sm // warps_per_block
    lim["blocks"] = limits["max_blocks_sm"]
    if regs_per_thread > 0:
        regs_per_warp = math.ceil(regs_per_thread * warp / g["reg_unit"]) * g["reg_unit"]
        if regs_per_warp * warps_per_block > limits["max_regs_block"]:
            return {"theo_occ": 0.0, "active_blocks": 0, "limiter": "registers-per-block", "method": "formula"}
        lim["registers"] = (limits["regs_sm"] // regs_per_warp) // warps_per_block
    else:
        lim["registers"] = 10**9
    if smem_per_block > 0:
        per_block = math.ceil((smem_per_block + g["smem_reserved"]) / g["smem_unit"]) * g["smem_unit"]
        lim["shared-memory"] = limits["smem_sm"] // per_block
    else:
        lim["shared-memory"] = 10**9

    active = min(lim.values())
    limiter = min(lim, key=lim.get)
    return {"theo_occ": round(active * warps_per_block / max_warps_sm, 4),
            "active_blocks": int(active), "limiter": limiter, "method": "formula"}


def driver_occupancy(func_ptr, block, dyn_smem, limits, warp=32):
    """Exact occupancy from the driver; returns None if the API isn't reachable from CuPy."""
    if func_ptr is None:
        return None
    try:
        import cupy as cp
        active = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(func_ptr, block, dyn_smem))
    except Exception:                                   # noqa: BLE001
        return None
    max_warps_sm = limits["max_threads_sm"] // warp
    return {"theo_occ": round(active * math.ceil(block / warp) / max_warps_sm, 4),
            "active_blocks": active, "limiter": "driver", "method": "driver"}


def occupancy(kernels, variant, block, dyn_smem, info):
    """Prefer the driver; fall back to the formula. `info` comes from common.gpu_info()."""
    res = driver_occupancy(kernels.function_ptr(variant), block, dyn_smem, info["limits"])
    if res is not None:
        return res
    attrs = kernels.attributes(variant)
    smem = max(attrs["shared_size_bytes"], 0) + dyn_smem
    return formula_occupancy(block, max(attrs["num_regs"], 0), smem, info["limits"], info["cc"][0])


if __name__ == "__main__":
    # Self-test against hand-worked cases.
    t4 = dict(max_threads_sm=1024, max_blocks_sm=16, regs_sm=65536, smem_sm=65536, max_regs_block=65536)
    p100 = dict(max_threads_sm=2048, max_blocks_sm=32, regs_sm=65536, smem_sm=65536, max_regs_block=65536)
    cases = [
        # (limits, cc, block, regs, smem, expected occupancy, expected limiter)
        (t4, 7, 256, 32, 1024, 1.0, "threads/warps"),     # 4 blocks x 8 warps = 32 of 32 warps
        (p100, 6, 256, 32, 1024, 1.0, "threads/warps"),   # 8 x 8 = 64 of 64
        (p100, 6, 256, 64, 1024, 0.5, "registers"),       # 2048 regs/warp -> 32 warps -> 4 blocks
        (t4, 7, 64, 16, 256, 1.0, "threads/warps"),       # 16 blocks x 2 warps = 32 of 32
        (p100, 6, 1024, 32, 8192, 1.0, "threads/warps"),  # 2 blocks x 32 warps
        (t4, 7, 256, 32, 40000, 0.25, "shared-memory"),   # 64K / 40,192 -> 1 block -> 8 of 32 warps
    ]
    ok = True
    for lim, ccm, b, r, sm, exp, exp_lim in cases:
        got = formula_occupancy(b, r, sm, lim, ccm)
        good = abs(got["theo_occ"] - exp) < 1e-9
        ok &= good
        print(f"[{'OK ' if good else 'BAD'}] block={b:4d} regs={r:3d} smem={sm:6d} -> occ={got['theo_occ']:.3f} "
              f"(expected {exp}) limiter={got['limiter']}")
    print("occupancy self-test:", "PASS" if ok else "FAIL")
