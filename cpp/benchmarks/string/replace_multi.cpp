/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

// Fixed target/replacement corpus ordered by (roughly) decreasing frequency in
// the synthetic data produced by `create_string_column` (see
// common/generate_input.cu). Using a fixed corpus lets us hold match density
// reasonably constant when sweeping `num_targets`, so sweeps isolate the
// "scan N targets" cost rather than mixing it with "how often do we hit".
std::vector<std::pair<std::string, std::string>> const kReplacementCorpus = {
  {"abc", "ABC"},      {"0987", "----"},      {"5W43", "qqqq"},    {"DEFGHI", "xxxxxx"},
  {"123", "@@@"},      {"9876543210", "___"}, {"xyz", "XYZ"},      {"4567890", "........"},
  {"01234", "!!!!!"},  {"56789", "~~~~~"},    {"def", "DEF"},      {"Wxyz", "wXYZ"},
  {"0123", "####"},    {"345", "???"},        {"edf", "EDF"},      {"DéFG", "DEFG"},
  {"abcdef", "ABCDEF"},{"ghijkl", "GHIJKL"},  {"mnopqr", "MNOPQR"},{"stuvwx", "STUVWX"},
  {"yz 01", "YZ-01"},  {"987", "987"},        {"Z 01", "z-01"},    {"456", "456"},
  {"789", "789"},      {"012", "012"},        {"345 6789", "X"},   {"abcdefghij", "Y"},
  {"ABC", "abc"},      {"DEF", "def"},        {"GHI", "ghi"},      {"JKL", "jkl"},
  {"MNO", "mno"},      {"PQR", "pqr"},        {"STU", "stu"},      {"VWX", "vwx"},
  {"YZ", "yz"},        {"01", "10"},          {"23", "32"},        {"45", "54"},
  {"67", "76"},        {"89", "98"},          {"éDE", "EDE"},      {"Dé", "De"},
  {" 0", " -"},        {" 1", " ."},          {" a", " A"},        {" 4", " $"},
  {" D", " d"},        {" W", " w"},          {" X", " x"},        {" 9", " 9"},
};

struct targets_and_repls {
  std::vector<std::string> targets;
  std::vector<std::string> repls;
};

// Single-byte, non-alphanumeric ASCII targets (0x01..0x7F minus digits and
// letters), each paired with its 3-byte "%XX" percent-encoded replacement.
// This matches the CtrstatEncode Java UDF's target/replacement set exactly,
// isolating the worst-case corner of the axis space:
//   - target_width = 1       (no multi-byte fast-forward on hit)
//   - num_targets  = up to 65 (full non-alnum ASCII coverage)
//   - replacement 3x expansion (every hit grows the output)
// `num_targets_cap` truncates the set if a smaller count is requested so the
// (hit-per-byte, target-count) scaling can be measured independently.
targets_and_repls build_ctrstat_targets(cudf::size_type num_targets_cap)
{
  targets_and_repls r;
  auto const is_non_alnum_ascii = [](int b) {
    return (b >= 0x01 && b < 0x30) || (b > 0x39 && b < 0x41) || (b > 0x5A && b < 0x61) ||
           (b > 0x7A && b < 0x80);
  };
  for (int b = 1; b < 128 && static_cast<cudf::size_type>(r.targets.size()) < num_targets_cap;
       ++b) {
    if (!is_non_alnum_ascii(b)) continue;
    r.targets.emplace_back(1, static_cast<char>(b));
    char buf[5];
    std::snprintf(buf, sizeof(buf), "%%%02X", b);
    r.repls.emplace_back(buf);
  }
  return r;
}

// Build targets / replacements of requested count. Replacement widths are kept
// close to target widths to hold output size roughly stable when sweeping.
targets_and_repls build_targets(cudf::size_type num_targets, cudf::size_type target_width)
{
  targets_and_repls r;
  r.targets.reserve(num_targets);
  r.repls.reserve(num_targets);

  // Additional synthetic targets that will never match real data, used when
  // num_targets exceeds the corpus size.
  auto const synth_for = [](cudf::size_type i, cudf::size_type w) {
    std::string s(w > 0 ? w : 4, 'Z');
    // Uniquely seed with i so they differ.
    s[0] = static_cast<char>('A' + (i % 26));
    s[1] = static_cast<char>('A' + ((i / 26) % 26));
    return s;
  };

  for (cudf::size_type i = 0; i < num_targets; ++i) {
    std::string t;
    std::string p;
    if (i < static_cast<cudf::size_type>(kReplacementCorpus.size())) {
      t = kReplacementCorpus[i].first;
      p = kReplacementCorpus[i].second;
    } else {
      t = synth_for(i, target_width);
      p = t;  // self-replace; never matches real data anyway
    }
    if (target_width > 0) {
      if (static_cast<cudf::size_type>(t.size()) > target_width) t.resize(target_width);
      while (static_cast<cudf::size_type>(t.size()) < target_width) t.push_back('Z');
      if (static_cast<cudf::size_type>(p.size()) > target_width) p.resize(target_width);
      while (static_cast<cudf::size_type>(p.size()) < target_width) p.push_back('_');
    }
    r.targets.push_back(std::move(t));
    r.repls.push_back(std::move(p));
  }
  return r;
}

// Host-side `replace_multiple` semantics: leftmost target wins per position.
std::string host_replace_all(std::string const& src,
                             std::vector<std::string> const& tgts,
                             std::vector<std::string> const& repls)
{
  std::string out;
  out.reserve(src.size());
  std::size_t i = 0;
  while (i < src.size()) {
    bool matched = false;
    for (std::size_t t = 0; t < tgts.size(); ++t) {
      auto const& tgt = tgts[t];
      if (!tgt.empty() && i + tgt.size() <= src.size() &&
          std::memcmp(src.data() + i, tgt.data(), tgt.size()) == 0) {
        out.append(repls[t]);
        i += tgt.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      out.push_back(src[i]);
      ++i;
    }
  }
  return out;
}

// CPU baseline: best-of-N wall time in ms over the full row set.
double measure_cpu_ms(std::vector<std::string> const& host_strings,
                      std::vector<std::string> const& tgts,
                      std::vector<std::string> const& repls,
                      int trials)
{
  double best = std::numeric_limits<double>::max();
  volatile std::size_t sink = 0;
  for (int t = 0; t < trials; ++t) {
    auto const start = std::chrono::steady_clock::now();
    for (auto const& s : host_strings) {
      auto out = host_replace_all(s, tgts, repls);
      sink += out.size();
    }
    auto const end = std::chrono::steady_clock::now();
    double const ms = std::chrono::duration<double, std::milli>(end - start).count();
    best            = std::min(best, ms);
  }
  (void)sink;
  return best;
}

// Cache CPU baselines so we don't re-measure the same (row_width, num_rows,
// num_targets, target_width, hit_rate) tuple repeatedly across nvbench
// warmup/measure phases.
using cpu_key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
std::set<cpu_key>& cpu_cache()
{
  static std::set<cpu_key> c;
  return c;
}
std::mutex& cpu_cache_mtx()
{
  static std::mutex m;
  return m;
}

bool cpu_baseline_enabled()
{
  auto const* v = std::getenv("CUDF_BENCH_CPU_BASELINE");
  return v && std::string(v) != "0" && std::string(v) != "";
}

void maybe_emit_cpu_baseline(cudf::column_view const& col,
                             std::vector<std::string> const& tgts,
                             std::vector<std::string> const& repls,
                             int64_t row_width,
                             int64_t num_rows,
                             int64_t num_targets,
                             int64_t target_width,
                             int64_t hit_rate)
{
  if (!cpu_baseline_enabled()) return;
  cpu_key const k{row_width, num_rows, num_targets, target_width, hit_rate};
  {
    std::lock_guard<std::mutex> lk(cpu_cache_mtx());
    if (!cpu_cache().insert(k).second) return;
  }
  auto host_pair        = cudf::test::to_host<std::string>(col);
  auto const& host_vec  = host_pair.first;
  std::vector<std::string> host_strings(host_vec.begin(), host_vec.end());

  int const trials = std::getenv("CUDF_BENCH_CPU_TRIALS")
                       ? std::atoi(std::getenv("CUDF_BENCH_CPU_TRIALS"))
                       : 3;
  double const ms = measure_cpu_ms(host_strings, tgts, repls, std::max(1, trials));
  // Emit one line per unique configuration for easy grepping.
  std::fprintf(stderr,
               "[cpu-baseline] row_width=%ld num_rows=%ld num_targets=%ld target_width=%ld "
               "hit_rate=%ld => %.3f ms\n",
               static_cast<long>(row_width),
               static_cast<long>(num_rows),
               static_cast<long>(num_targets),
               static_cast<long>(target_width),
               static_cast<long>(hit_rate),
               ms);
}

}  // namespace

static void bench_replace_multi(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width    = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const num_targets  = static_cast<cudf::size_type>(state.get_int64("num_targets"));
  auto const target_width = static_cast<cudf::size_type>(state.get_int64("target_width"));
  auto const hit_rate     = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Same data generator as find_multiple.cpp; plants anchor substrings in row 0
  // at a configurable hit rate.
  auto const column = create_string_column(num_rows, row_width, hit_rate);
  cudf::strings_column_view input(column->view());

  auto const tr = build_targets(num_targets, target_width);
  cudf::test::strings_column_wrapper targets(tr.targets.begin(), tr.targets.end());
  cudf::test::strings_column_wrapper repls(tr.repls.begin(), tr.repls.end());

  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  // Optional CPU baseline (enabled via CUDF_BENCH_CPU_BASELINE=1). Emitted once
  // per unique parameter tuple to stderr.
  maybe_emit_cpu_baseline(column->view(),
                          tr.targets,
                          tr.repls,
                          row_width,
                          num_rows,
                          num_targets,
                          target_width,
                          hit_rate);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::strings::replace_multiple(
      input, cudf::strings_column_view(targets), cudf::strings_column_view(repls));
  });
}

NVBENCH_BENCH(bench_replace_multi)
  .set_name("replace_multi")
  // Row-width sweep spans string- and character-parallel regimes:
  //  32  -> avg < 256 -> replace_string_parallel
  //  128 -> still < 256
  //  512 -> > 256 -> replace_character_parallel
  //  2048-> heavily character-parallel
  .add_int64_axis("row_width", {32, 128, 512, 2048})
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("num_targets", {1, 2, 8, 32, 128})
  .add_int64_axis("target_width", {0})  // 0 => use natural corpus widths
  .add_int64_axis("hit_rate", {5, 40});

// Single-config benchmark intended for focused NCU profiling. Keep the axis
// space tiny so a `--profile` run doesn't explode into dozens of reports.
static void bench_replace_multi_ncu(nvbench::state& state) { bench_replace_multi(state); }

NVBENCH_BENCH(bench_replace_multi_ncu)
  .set_name("replace_multi_ncu")
  .add_int64_axis("row_width", {128, 2048})
  .add_int64_axis("num_rows", {2097152})
  .add_int64_axis("num_targets", {32})
  .add_int64_axis("target_width", {0})
  .add_int64_axis("hit_rate", {40});

// CtrstatEncode-shape benchmark: replicate the Java UDF's exact target
// configuration (up to 65 single-byte non-alnum ASCII targets, each replaced
// with its 3-byte "%XX" percent-encoded form). Row widths and per-batch row
// counts are chosen to bracket the UDF's observed workload (~130-byte rows,
// ~166k rows per batch when splitting 1M rows into six 256 MB batches).
static void bench_replace_multi_ctrstat(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width      = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const num_targets    = static_cast<cudf::size_type>(state.get_int64("num_targets"));
  auto const hit_rate       = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto constexpr target_w   = int64_t{1};

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const column = create_string_column(num_rows, row_width, hit_rate);
  cudf::strings_column_view input(column->view());

  auto const tr = build_ctrstat_targets(num_targets);
  cudf::test::strings_column_wrapper targets(tr.targets.begin(), tr.targets.end());
  cudf::test::strings_column_wrapper repls(tr.repls.begin(), tr.repls.end());

  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  maybe_emit_cpu_baseline(column->view(),
                          tr.targets,
                          tr.repls,
                          row_width,
                          num_rows,
                          static_cast<int64_t>(tr.targets.size()),
                          target_w,
                          hit_rate);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& /*launch*/) {
    cudf::strings::replace_multiple(
      input, cudf::strings_column_view(targets), cudf::strings_column_view(repls));
  });
}

NVBENCH_BENCH(bench_replace_multi_ctrstat)
  .set_name("replace_multi_ctrstat")
  // Row widths bracket the CtrstatEncode UDF's observed range (~130 B) plus
  // one cell on each side. All values stay below AVG_CHAR_BYTES_THRESHOLD=256
  // so every cell hits the replace_string_parallel path the UDF actually uses.
  .add_int64_axis("row_width", {64, 128, 200})
  // Includes the UDF's measured per-batch row count (~166k) plus bookends to
  // see num_rows-scaling effects at fixed target shape.
  .add_int64_axis("num_rows", {166667, 262144, 2097152})
  // Sub-sweep of targets: 8 / 32 / 65. 65 = full UDF set.
  .add_int64_axis("num_targets", {8, 32, 65})
  // Anchor-density of create_string_column; effective per-byte hit density is
  // ~10-15% with our 65-char non-alnum dictionary (vs ~20-30% for real URL
  // data) but still enough to exercise the algorithmic cost.
  .add_int64_axis("hit_rate", {5, 40});
