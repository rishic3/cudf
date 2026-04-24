/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/positions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <cstdint>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or character-parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the character-parallel function is used.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the replace-multi benchmark results using an
 * RTX A6000.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 256;

/**
 * @brief Type used for holding the target position (first) and the
 * target index (second).
 */
using target_pair = cuda::std::tuple<int64_t, size_type>;

/**
 * @brief Sort key packing the target's first byte and original index.
 *
 * Packing the original index into the sort key preserves the target-list order
 * within each first-byte bucket, which is required because replace_multiple
 * uses the first matching target in target-list order at each position.
 */
constexpr uint64_t make_target_sort_key(u_char first_byte, size_type index)
{
  return (static_cast<uint64_t>(first_byte) << 32) | static_cast<uint32_t>(index);
}

constexpr u_char target_key_first_byte(uint64_t key)
{
  return static_cast<u_char>(key >> 32);
}

constexpr size_type target_key_index(uint64_t key)
{
  return static_cast<size_type>(static_cast<uint32_t>(key));
}

struct target_first_byte_lookup {
  rmm::device_uvector<u_char> first_bytes;
  rmm::device_uvector<uint64_t> sorted_target_keys;
  rmm::device_uvector<size_type> first_byte_offsets;
  size_type unique_first_bytes{};

  target_first_byte_lookup(size_type num_targets, rmm::cuda_stream_view stream)
    : first_bytes(num_targets, stream),
      sorted_target_keys(num_targets, stream),
      first_byte_offsets(num_targets, stream)
  {
  }
};

target_first_byte_lookup build_target_first_byte_lookup(device_span<string_view const> d_targets,
                                                        rmm::cuda_stream_view stream)
{
  auto lookup = target_first_byte_lookup{static_cast<size_type>(d_targets.size()), stream};

  auto const key_itr = thrust::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<uint64_t>([d_targets] __device__(size_type idx) -> uint64_t {
      auto const d_tgt      = d_targets[idx];
      auto const first_byte = d_tgt.empty() ? u_char{0} : static_cast<u_char>(d_tgt.data()[0]);
      return make_target_sort_key(first_byte, idx);
    }));

  auto const num_items = d_targets.size();
  auto const sv        = stream.value();

  std::size_t tmp_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(
    nullptr,
    tmp_bytes,
    key_itr,
    lookup.sorted_target_keys.begin(),
    num_items,
    cuda::std::less{},
    sv);
  auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
  cub::DeviceMergeSort::SortKeysCopy(tmp_stg.data(),
                                     tmp_bytes,
                                     key_itr,
                                     lookup.sorted_target_keys.begin(),
                                     num_items,
                                     cuda::std::less{},
                                     sv);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    lookup.sorted_target_keys.begin(),
                    lookup.sorted_target_keys.end(),
                    lookup.first_bytes.begin(),
                    cuda::proclaim_return_type<u_char>(
                      [] __device__(uint64_t key) { return target_key_first_byte(key); }));

  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   lookup.first_byte_offsets.begin(),
                   lookup.first_byte_offsets.end());
  auto const end =
    thrust::unique_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          lookup.first_bytes.begin(),
                          lookup.first_bytes.end(),
                          lookup.first_byte_offsets.begin());
  lookup.unique_first_bytes =
    static_cast<size_type>(cuda::std::distance(lookup.first_bytes.begin(), end.first));

  return lookup;
}

/**
 * @brief Helper functions for performing character-parallel replace
 */
struct replace_multi_parallel_fn {
  __device__ char const* get_base_ptr() const { return d_strings.head<char>(); }

  __device__ string_view const get_string(size_type idx) const
  {
    return d_strings.element<string_view>(idx);
  }

  __device__ string_view const get_replacement_string(size_type idx) const
  {
    return d_replacements.size() == 1 ? d_replacements[0] : d_replacements[idx];
  }

  __device__ bool is_valid(size_type idx) const { return d_strings.is_valid(idx); }

  __device__ bool get_target_key_range(char const* d_chars,
                                       size_type& first_idx,
                                       size_type& last_idx) const
  {
    auto const chr      = static_cast<u_char>(*d_chars);
    auto const last_ptr = d_first_bytes + unique_first_bytes;
    auto const byte_ptr = thrust::lower_bound(thrust::seq, d_first_bytes, last_ptr, chr);
    if ((byte_ptr == last_ptr) || (*byte_ptr != chr)) { return false; }

    auto const offset_idx = static_cast<size_type>(cuda::std::distance(d_first_bytes, byte_ptr));
    first_idx             = d_first_byte_offsets[offset_idx];
    last_idx              = (offset_idx + 1) < unique_first_bytes
                              ? d_first_byte_offsets[offset_idx + 1]
                              : static_cast<size_type>(d_targets.size());
    return true;
  }

  /**
   * @brief Returns the index of the target string found at the given byte position
   * in the input strings column
   *
   * @param idx Index of the byte position in the chars column
   * @param chars_bytes Number of bytes in the chars column
   */
  __device__ size_type target_index(int64_t idx, int64_t chars_bytes) const
  {
    auto const d_offsets = d_strings_offsets;
    auto const d_chars   = get_base_ptr() + d_offsets[0] + idx;
    size_type str_idx    = -1;
    string_view d_str{};
    size_type first_idx = 0;
    size_type last_idx  = 0;
    if (!get_target_key_range(d_chars, first_idx, last_idx)) { return -1; }

    for (auto key_idx = first_idx; key_idx < last_idx; ++key_idx) {
      auto const t     = target_key_index(d_target_keys[key_idx]);
      auto const d_tgt = d_targets[t];
      if (!d_tgt.empty() && (idx + d_tgt.size_bytes() <= chars_bytes) &&
          (d_tgt.compare(d_chars, d_tgt.size_bytes()) == 0)) {
        if (str_idx < 0) {
          auto const idx_itr =
            thrust::upper_bound(thrust::seq, d_offsets, d_offsets + d_strings.size(), idx);
          str_idx = cuda::std::distance(d_offsets, idx_itr) - 1;
          d_str   = get_string(str_idx - d_offsets[0]);
        }
        if ((d_chars + d_tgt.size_bytes()) <= (d_str.data() + d_str.size_bytes())) { return t; }
      }
    }
    return -1;
  }

  __device__ bool has_target(int64_t idx, int64_t chars_bytes) const
  {
    auto const d_chars = get_base_ptr() + d_strings_offsets[0] + idx;
    size_type first_idx = 0;
    size_type last_idx  = 0;
    if (!get_target_key_range(d_chars, first_idx, last_idx)) { return false; }

    for (auto key_idx = first_idx; key_idx < last_idx; ++key_idx) {
      auto const d_tgt = d_targets[target_key_index(d_target_keys[key_idx])];
      if (!d_tgt.empty() && (idx + d_tgt.size_bytes() <= chars_bytes) &&
          (d_tgt.compare(d_chars, d_tgt.size_bytes()) == 0)) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Count the number of strings that will be produced by the replace
   *
   * This includes segments of the string that are not replaced as well as those
   * that are replaced.
   *
   * @param idx Index of the row in d_strings to be processed
   * @param d_positions Positions of the targets found in the chars column
   * @param d_indices Indices into the targets array
   * @param d_targets_offsets Offsets identify which target positions go with the current string
   * @return Number of substrings resulting from the replace operations on this row
   */
  __device__ size_type count_strings(size_type idx,
                                     int64_t const* d_positions,
                                     size_type const* d_indices,
                                     cudf::detail::input_offsetalator d_targets_offsets) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const d_str     = get_string(idx);
    auto const d_str_end = d_str.data() + d_str.size_bytes();
    auto const base_ptr  = get_base_ptr();

    auto const target_offset = d_targets_offsets[idx];
    auto const targets_size  = static_cast<size_type>(d_targets_offsets[idx + 1] - target_offset);
    auto const positions     = d_positions + target_offset;
    auto const indices       = d_indices + target_offset;

    size_type count = 1;  // always at least one string
    auto str_ptr    = d_str.data();
    for (std::size_t i = 0; i < targets_size; ++i) {
      auto const tgt_idx = indices[i];
      auto const d_tgt   = d_targets[tgt_idx];
      auto const tgt_ptr = base_ptr + positions[i];
      if (str_ptr <= tgt_ptr && tgt_ptr < d_str_end) {
        auto const keep_size = static_cast<size_type>(cuda::std::distance(str_ptr, tgt_ptr));
        if (keep_size > 0) { count++; }  // don't bother counting empty strings

        auto const d_repl = get_replacement_string(tgt_idx);
        if (!d_repl.empty()) { count++; }

        str_ptr += keep_size + d_tgt.size_bytes();
      }
    }

    return count;
  }

  /**
   * @brief Retrieve the strings for each row
   *
   * This will return string segments as string_index_pair objects for
   * parts of the string that are not replaced interlaced with the
   * appropriate replacement string where replacement targets are found.
   *
   * This function is called only once to produce both the string_index_pair objects
   * and the output row size in bytes.
   *
   * @param idx Index of the row in d_strings
   * @param d_offsets Offsets to identify where to store the results of the replace for this string
   * @param d_positions The target positions found in the chars column
   * @param d_indices Indices into the targets array
   * @param d_targets_offsets The offsets to identify which target positions go with this string
   * @param d_all_strings The output of all the produced string segments
   * @return The size in bytes of the output string for this row
   */
  __device__ size_type get_strings(size_type idx,
                                   cudf::detail::input_offsetalator const d_offsets,
                                   int64_t const* d_positions,
                                   size_type const* d_indices,
                                   cudf::detail::input_offsetalator d_targets_offsets,
                                   string_index_pair* d_all_strings) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const d_output  = d_all_strings + d_offsets[idx];
    auto const d_str     = get_string(idx);
    auto const d_str_end = d_str.data() + d_str.size_bytes();
    auto const base_ptr  = get_base_ptr();

    auto const target_offset = d_targets_offsets[idx];
    auto const targets_size  = static_cast<size_type>(d_targets_offsets[idx + 1] - target_offset);
    auto const positions     = d_positions + target_offset;
    auto const indices       = d_indices + target_offset;

    size_type output_idx  = 0;
    size_type output_size = 0;
    auto str_ptr          = d_str.data();
    for (std::size_t i = 0; i < targets_size; ++i) {
      auto const tgt_idx = indices[i];
      auto const d_tgt   = d_targets[tgt_idx];
      auto const tgt_ptr = base_ptr + positions[i];
      if (str_ptr <= tgt_ptr && tgt_ptr < d_str_end) {
        auto const keep_size = static_cast<size_type>(cuda::std::distance(str_ptr, tgt_ptr));
        if (keep_size > 0) { d_output[output_idx++] = string_index_pair{str_ptr, keep_size}; }
        output_size += keep_size;

        auto const d_repl = get_replacement_string(tgt_idx);
        if (!d_repl.empty()) {
          d_output[output_idx++] = string_index_pair{d_repl.data(), d_repl.size_bytes()};
        }
        output_size += d_repl.size_bytes();

        str_ptr += keep_size + d_tgt.size_bytes();
      }
    }
    // include any leftover parts of the string
    if (str_ptr <= d_str_end) {
      auto const left_size = static_cast<size_type>(cuda::std::distance(str_ptr, d_str_end));
      d_output[output_idx] = string_index_pair{str_ptr, left_size};
      output_size += left_size;
    }
    return output_size;
  }

  replace_multi_parallel_fn(column_device_view const& d_strings,
                            cudf::detail::input_offsetalator d_strings_offsets,
                            device_span<string_view const> d_targets,
                            device_span<string_view const> d_replacements,
                            u_char const* d_first_bytes,
                            uint64_t const* d_target_keys,
                            size_type const* d_first_byte_offsets,
                            size_type unique_first_bytes)
    : d_strings(d_strings),
      d_strings_offsets(d_strings_offsets),
      d_targets{d_targets},
      d_replacements{d_replacements},
      d_first_bytes{d_first_bytes},
      d_target_keys{d_target_keys},
      d_first_byte_offsets{d_first_byte_offsets},
      unique_first_bytes{unique_first_bytes}
  {
  }

 protected:
  column_device_view d_strings;
  cudf::detail::input_offsetalator d_strings_offsets;
  device_span<string_view const> d_targets;
  device_span<string_view const> d_replacements;
  u_char const* d_first_bytes{};
  uint64_t const* d_target_keys{};
  size_type const* d_first_byte_offsets{};
  size_type unique_first_bytes{};
};

constexpr int64_t block_size         = 512;  // number of threads per block
constexpr size_type bytes_per_thread = 4;    // bytes processed per thread

/**
 * @brief Count the number of targets in a strings column
 *
 * @param fn Functor containing has_target() function
 * @param chars_bytes Number of bytes in the strings column
 * @param d_output Result of the count
 */
CUDF_KERNEL void count_targets(replace_multi_parallel_fn fn, int64_t chars_bytes, int64_t* d_output)
{
  auto const idx      = cudf::detail::grid_1d::global_thread_id();
  auto const byte_idx = static_cast<int64_t>(idx) * bytes_per_thread;
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  using block_reduce = cub::BlockReduce<int64_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  int64_t count = 0;
  // each thread processes multiple bytes
  for (auto i = byte_idx; (i < (byte_idx + bytes_per_thread)) && (i < chars_bytes); ++i) {
    count += fn.has_target(i, chars_bytes);
  }
  auto const total = block_reduce(temp_storage).Reduce(count, cuda::std::plus());

  if ((lane_idx == 0) && (total > 0)) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{*d_output};
    ref.fetch_add(total, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Used by the copy-if function to produce target_pair objects
 *
 * Using an inplace lambda caused a runtime crash in thrust::copy_if
 * (this happens sometimes when passing device lambdas to thrust algorithms)
 */
struct pair_generator {
  __device__ target_pair operator()(int64_t idx) const
  {
    return cuda::std::make_tuple(idx, fn.target_index(idx, chars_bytes));
  }
  replace_multi_parallel_fn fn;
  int64_t chars_bytes;
};

struct copy_if_fn {
  __device__ bool operator()(target_pair pos) { return cuda::std::get<1>(pos) >= 0; }
};

std::unique_ptr<column> replace_character_parallel(strings_column_view const& input,
                                                   strings_column_view const& targets,
                                                   strings_column_view const& repls,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto const strings_count = input.size();
  auto const chars_bytes =
    get_offset_value(input.offsets(), input.offset() + strings_count, stream) -
    get_offset_value(input.offsets(), input.offset(), stream);

  auto d_targets =
    create_string_vector_from_column(targets, stream, cudf::get_current_device_resource_ref());
  auto d_replacements =
    create_string_vector_from_column(repls, stream, cudf::get_current_device_resource_ref());
  auto target_lookup = build_target_first_byte_lookup(d_targets, stream);

  replace_multi_parallel_fn fn{
    *d_strings,
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset()),
    d_targets,
    d_replacements,
    target_lookup.first_bytes.data(),
    target_lookup.sorted_target_keys.data(),
    target_lookup.first_byte_offsets.data(),
    target_lookup.unique_first_bytes,
  };

  // Count the number of targets in the entire column.
  // Note this may over-count in the case where a target spans adjacent strings.
  cudf::detail::device_scalar<int64_t> d_count(0, stream, cudf::get_current_device_resource_ref());
  auto const num_blocks = util::div_rounding_up_safe(
    util::div_rounding_up_safe(chars_bytes, static_cast<int64_t>(bytes_per_thread)), block_size);
  count_targets<<<num_blocks, block_size, 0, stream.value()>>>(fn, chars_bytes, d_count.data());
  auto target_count = d_count.value(stream);
  // Create a vector of every target position in the chars column.
  // These may also include overlapping targets which will be resolved later.
  auto targets_positions = rmm::device_uvector<int64_t>(target_count, stream);
  auto targets_indices   = rmm::device_uvector<size_type>(target_count, stream);

  // cudf::detail::make_counting_transform_iterator hardcodes size_type
  auto const copy_itr = thrust::make_transform_iterator(cuda::counting_iterator<int64_t>{0},
                                                        pair_generator{fn, chars_bytes});
  auto const out_itr  = thrust::make_zip_iterator(
    cuda::std::make_tuple(targets_positions.begin(), targets_indices.begin()));
  auto const copy_end =
    cudf::detail::copy_if(copy_itr, copy_itr + chars_bytes, out_itr, copy_if_fn{}, stream);

  // adjust target count since the copy-if may have eliminated some invalid targets
  target_count = std::min(static_cast<int64_t>(std::distance(out_itr, copy_end)), target_count);
  targets_positions.resize(target_count, stream);
  targets_indices.resize(target_count, stream);
  auto d_positions       = targets_positions.data();
  auto d_targets_indices = targets_indices.data();

  // create a vector of offsets to each string's set of target positions
  auto const targets_offsets = create_offsets_from_positions(
    input, targets_positions, stream, cudf::get_current_device_resource_ref());
  auto const d_targets_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(targets_offsets->view());

  // compute the number of string segments produced by replace in each string
  auto counts = rmm::device_uvector<size_type>(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    counts.begin(),
                    cuda::proclaim_return_type<size_type>(
                      [fn, d_positions, d_targets_indices, d_targets_offsets] __device__(
                        size_type idx) -> size_type {
                        return fn.count_strings(
                          idx, d_positions, d_targets_indices, d_targets_offsets);
                      }));

  // create offsets from the counts
  auto [offsets, total_strings] =
    cudf::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr);
  auto const d_strings_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // build a vector of all the positions for all the strings
  auto indices   = rmm::device_uvector<string_index_pair>(total_strings, stream);
  auto d_indices = indices.data();
  auto d_sizes   = counts.data();  // reusing this vector to hold output sizes now
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    strings_count,
    [fn,
     d_strings_offsets,
     d_positions,
     d_targets_indices,
     d_targets_offsets,
     d_indices,
     d_sizes] __device__(size_type idx) {
      d_sizes[idx] = fn.get_strings(
        idx, d_strings_offsets, d_positions, d_targets_indices, d_targets_offsets, d_indices);
    });

  // use this utility to gather the string parts into a contiguous chars column
  auto chars      = make_strings_column(indices.begin(), indices.end(), stream, mr);
  auto chars_data = chars->release().data;

  // create offsets from the sizes
  offsets = std::get<0>(
    cudf::strings::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr));

  // build the strings columns from the chars and offsets
  return make_strings_column(strings_count,
                             std::move(offsets),
                             std::move(chars_data.release()[0]),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

/**
 * @brief Function logic for the replace_string_parallel
 *
 * Performs the multi-replace operation with a thread per string.
 * This performs best on smaller strings. @see AVG_CHAR_BYTES_THRESHOLD
 */
struct replace_multi_fn {
  column_device_view const d_strings;
  device_span<string_view const> d_targets;
  device_span<string_view const> d_repls;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;
  u_char const* d_first_bytes{};
  uint64_t const* d_target_keys{};
  size_type const* d_first_byte_offsets{};
  size_type unique_first_bytes{};

  __device__ string_view get_replacement_string(size_type idx) const
  {
    return d_repls.size() == 1 ? d_repls[0] : d_repls[idx];
  }

  __device__ bool get_target_key_range(char const* d_chars,
                                       size_type& first_idx,
                                       size_type& last_idx) const
  {
    auto const chr      = static_cast<u_char>(*d_chars);
    auto const last_ptr = d_first_bytes + unique_first_bytes;
    auto const byte_ptr = thrust::lower_bound(thrust::seq, d_first_bytes, last_ptr, chr);
    if ((byte_ptr == last_ptr) || (*byte_ptr != chr)) { return false; }

    auto const offset_idx = static_cast<size_type>(cuda::std::distance(d_first_bytes, byte_ptr));
    first_idx             = d_first_byte_offsets[offset_idx];
    last_idx              = (offset_idx + 1) < unique_first_bytes
                              ? d_first_byte_offsets[offset_idx + 1]
                              : static_cast<size_type>(d_targets.size());
    return true;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    char const* in_ptr = d_str.data();

    size_type bytes = d_str.size_bytes();
    size_type spos  = 0;
    size_type lpos  = 0;
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // check each character against each target
    while (spos < d_str.size_bytes()) {
      size_type first_idx = 0;
      size_type last_idx  = 0;
      if (get_target_key_range(in_ptr + spos, first_idx, last_idx)) {
        for (auto key_idx = first_idx; key_idx < last_idx; ++key_idx) {
          auto const tgt_idx = target_key_index(d_target_keys[key_idx]);
          auto const d_tgt   = d_targets[tgt_idx];
          if (!d_tgt.empty() && (d_tgt.size_bytes() <= (d_str.size_bytes() - spos)) &&  // check fit
              (d_tgt.compare(in_ptr + spos, d_tgt.size_bytes()) == 0))                  // and match
          {
            auto const d_repl = get_replacement_string(tgt_idx);
            bytes += d_repl.size_bytes() - d_tgt.size_bytes();
            if (out_ptr) {
              out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
              out_ptr = copy_string(out_ptr, d_repl);
              lpos    = spos + d_tgt.size_bytes();
            }
            spos += d_tgt.size_bytes() - 1;
            break;
          }
        }
      }
      ++spos;
    }
    if (out_ptr) {
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);  // copy remainder
    } else {
      d_sizes[idx] = bytes;
    }
  }
};

std::unique_ptr<column> replace_string_parallel(strings_column_view const& input,
                                                strings_column_view const& targets,
                                                strings_column_view const& repls,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto d_strings      = column_device_view::create(input.parent(), stream);
  auto d_targets =
    create_string_vector_from_column(targets, stream, cudf::get_current_device_resource_ref());
  auto d_replacements =
    create_string_vector_from_column(repls, stream, cudf::get_current_device_resource_ref());
  auto target_lookup = build_target_first_byte_lookup(d_targets, stream);

  auto [offsets_column, chars] = make_strings_children(
    replace_multi_fn{*d_strings,
                     d_targets,
                     d_replacements,
                     {},
                     {},
                     {},
                     target_lookup.first_bytes.data(),
                     target_lookup.sorted_target_keys.data(),
                     target_lookup.first_byte_offsets.data(),
                     target_lookup.unique_first_bytes},
    input.size(),
    stream,
    mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> replace_multiple(strings_column_view const& input,
                                         strings_column_view const& targets,
                                         strings_column_view const& repls,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(((targets.size() > 0) && (targets.null_count() == 0)),
               "Parameters targets must not be empty and must not have nulls");
  CUDF_EXPECTS(((repls.size() > 0) && (repls.null_count() == 0)),
               "Parameters repls must not be empty and must not have nulls");
  if (repls.size() > 1)
    CUDF_EXPECTS(repls.size() == targets.size(), "Sizes for targets and repls must match");

  return (input.size() == input.null_count() ||
          ((input.chars_size(stream) / (input.size() - input.null_count())) <
           AVG_CHAR_BYTES_THRESHOLD))
           ? replace_string_parallel(input, targets, repls, stream, mr)
           : replace_character_parallel(input, targets, repls, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_multiple(strings_column_view const& strings,
                                         strings_column_view const& targets,
                                         strings_column_view const& repls,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_multiple(strings, targets, repls, stream, mr);
}

}  // namespace strings
}  // namespace cudf
