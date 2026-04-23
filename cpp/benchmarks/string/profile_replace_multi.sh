#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Drive `ncu` over the cudf replace_multiple benchmark.
#
# Usage:
#   profile_replace_multi.sh [OUTDIR]
#
# Requirements:
#   - libcudf built with BUILD_BENCHMARKS=ON (STRINGS_NVBENCH target).
#   - `ncu` (NVIDIA Compute) on PATH; we use the CUDA toolkit one at
#     /usr/local/cuda/bin/ncu if not overridden via NCU.
#   - Root / CAP_SYS_ADMIN or the persisted "allow" flag for perf counters
#     (`modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`) -- otherwise
#     NCU prints "ERR_NVGPUCTRPERM" and exits.
#
# What this collects (the "light" NCU set):
#   --section SpeedOfLight           GPU saturation (compute vs memory)
#   --section Occupancy              Theoretical / achieved occupancy
#   --section MemoryWorkloadAnalysis Global/shared/L2/DRAM traffic
#   --section LaunchStats            Block/grid dims, registers, shared mem
#   --section SchedulerStats         Issue slots & stall reasons
#   --section WarpStateStats         Active warp state distribution
#
# Kernels we target explicitly:
#   - count_targets                       (char-parallel, pass 1)
#   - cudf::detail::copy_if*              (char-parallel, pass 2: copy_if)
#   - replace_multi_parallel_fn*          (functor wrapped in thrust::transform / for_each)
#   - replace_multi_fn*                   (string-parallel kernel)
#   - make_strings_children / gather      (materialization)
#
# For the string-parallel vs character-parallel split, we run two benchmark
# configurations (row_width=128 and row_width=2048) which the benchmark file
# registers under "replace_multi_ncu".

set -euo pipefail

OUTDIR="${1:-ncu_reports_replace_multi}"
mkdir -p "$OUTDIR"

# Resolve tools
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
if [[ ! -x "$NCU" ]]; then
  echo "error: ncu not found (set NCU=/path/to/ncu)" >&2
  exit 1
fi

# Resolve benchmark binary. Search the most common build output locations.
find_bench() {
  local candidates=(
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/cpp/build/latest/benchmarks/STRINGS_NVBENCH"
    "$(git rev-parse --show-toplevel 2>/dev/null || pwd)/cpp/build/benchmarks/STRINGS_NVBENCH"
    "$(pwd)/cpp/build/latest/benchmarks/STRINGS_NVBENCH"
    "$(pwd)/cpp/build/benchmarks/STRINGS_NVBENCH"
  )
  for p in "${candidates[@]}"; do
    if [[ -x "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

BENCH="${BENCH:-$(find_bench || true)}"
if [[ -z "$BENCH" || ! -x "$BENCH" ]]; then
  echo "error: STRINGS_NVBENCH not found. Build first (build-cudf-cpp -j0 -DBUILD_BENCHMARKS=ON)" >&2
  echo "       or set BENCH=/path/to/STRINGS_NVBENCH" >&2
  exit 1
fi

echo "ncu   : $NCU"
echo "bench : $BENCH"
echo "out   : $OUTDIR"
echo

# Kernel name regexes (NCU uses POSIX ERE by default with --kernel-name-base regex).
# Cover both char-parallel and string-parallel code paths plus the post-match
# materialization. These are greedy substrings.
KERNEL_REGEX='count_targets|replace_multi|copy_if|make_strings|gather_chars|for_each_n'

# Common NCU flags.
NCU_SECTIONS=(
  --section SpeedOfLight
  --section Occupancy
  --section MemoryWorkloadAnalysis
  --section LaunchStats
  --section SchedulerStats
  --section WarpStateStats
)

run_profile() {
  local tag="$1"; shift
  local rep="$OUTDIR/${tag}.ncu-rep"
  local txt="$OUTDIR/${tag}.txt"
  echo ">>> profiling: $tag"
  "$NCU" \
    --target-processes all \
    --replay-mode kernel \
    --launch-skip-before-match 0 \
    --launch-count 16 \
    --kernel-name regex:"$KERNEL_REGEX" \
    "${NCU_SECTIONS[@]}" \
    --export "$rep" \
    --force-overwrite \
    "$BENCH" "$@" > "$txt" 2>&1 || {
      echo "  !! ncu exited non-zero; head of log:" >&2
      tail -n 40 "$txt" >&2
      return 1
    }
  echo "    report -> $rep"
  echo "    log    -> $txt"
}

# Benchmark args: nvbench takes --benchmark <name> and then --devices, axis
# filters etc. STRINGS_NVBENCH is built from many *.cpp files so --benchmark
# is required to pick just this one. --min-samples 1 keeps the run short.
#
# Use the focused registrations in replace_multi.cpp:
#   replace_multi_ncu  (row_width in {128, 2048})
BENCH_ARGS_COMMON=(
  --benchmark replace_multi_ncu
  --min-samples 1
  --min-time 0.01
  --timeout 120
)

# Run once per interesting row-width to separate the two algorithmic paths.
run_profile "string_parallel_rw128" \
  "${BENCH_ARGS_COMMON[@]}" \
  --axis "row_width=128"

run_profile "char_parallel_rw2048" \
  "${BENCH_ARGS_COMMON[@]}" \
  --axis "row_width=2048"

echo
echo "done. Open a report with: ncu-ui $OUTDIR/string_parallel_rw128.ncu-rep"
echo "     or dump to text with: ncu --import $OUTDIR/string_parallel_rw128.ncu-rep --csv"
