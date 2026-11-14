"""
Scale benchmark for bioframe.read_chain / bioframe.liftover.

Builds a synthetic chain file with 100,000 aligned blocks and 100,000 sparse
queries (half fall inside one block, half span a block boundary and intersect
two blocks), then reports build and query wall time, peak resident memory and
the number of block-array positions actually examined.  The access count
growing linearly (rather than as rows x blocks) rules out an N x B scan.

Run with:  .venv/bin/python benchmarks/liftover_benchmark.py
"""

import io
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import bioframe as bf
from bioframe.io.liftover import _overlapping_blocks

N_BLOCKS = 100_000
N_QUERIES = 100_000
BLOCK_SIZE = 50
GAP = 50  # equal gap and block -> blocks tile every other base window
CHROM_SIZE = N_BLOCKS * (BLOCK_SIZE + GAP) + 1


def build_chain_text(n_blocks=N_BLOCKS):
    span_end = n_blocks * (BLOCK_SIZE + GAP) - GAP  # end of the last block
    buf = io.StringIO()
    buf.write(
        f"chain 100 c1 {CHROM_SIZE} + 0 {span_end} "
        f"q1 {CHROM_SIZE} + 0 {span_end} 42\n"
    )
    # Target and query use identical coordinates; adjacent blocks separated
    # by target/query gaps of GAP bases (dt = dq = GAP).
    for _ in range(n_blocks - 1):
        buf.write(f"{BLOCK_SIZE} {GAP} {GAP}\n")
    buf.write(f"{BLOCK_SIZE}\n")
    return buf.getvalue()


def main():
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"interpreter + numpy/pandas baseline RSS: {baseline_rss / 1024:.1f} MB"
    )

    text = build_chain_text()
    text_bytes = len(text.encode("utf-8"))
    print(f"synthetic chain: {N_BLOCKS:,} blocks, {text_bytes / 1e6:.1f} MB text")

    t0 = time.perf_counter()
    chains = bf.read_chain(io.StringIO(text))
    build_time = time.perf_counter() - t0
    build_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"read_chain build time : {build_time:.3f} s")
    print(
        f"peak RSS after build  : {build_rss / 1024:.1f} MB "
        f"(incremental {max(0, build_rss - baseline_rss) / 1024:.1f} MB)"
    )
    assert len(chains[42].blocks) == N_BLOCKS

    rng = np.random.default_rng(0)
    block_starts = np.arange(N_BLOCKS) * (BLOCK_SIZE + GAP)
    half = N_QUERIES // 2

    # Half of the queries sit inside a single block (one intersection each).
    single_starts = block_starts[:half] + rng.integers(0, 10, half)
    single_ends = single_starts + rng.integers(10, BLOCK_SIZE, half)

    # The other half start near the end of one block and run past the gap
    # into the next block, so each intersects exactly two blocks.
    boundary_idx = np.arange(1, half + 1)
    two_starts = block_starts[boundary_idx] + BLOCK_SIZE - 10
    two_ends = block_starts[boundary_idx] + BLOCK_SIZE + GAP + 10

    starts = np.concatenate([single_starts, two_starts]).astype(np.int64)
    ends = np.concatenate([single_ends, two_ends]).astype(np.int64)
    # Shuffle so block order and query order are independent.
    perm = rng.permutation(len(starts))
    starts, ends = starts[perm], ends[perm]
    df = pd.DataFrame(
        {
            "chrom": np.full(N_QUERIES, "c1", dtype=object),
            "start": starts,
            "end": ends,
        }
    )

    t0 = time.perf_counter()
    mapped, unmapped = bf.liftover(df, chains, min_match=0.1)
    query_time = time.perf_counter() - t0
    query_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"liftover query time   : {query_time:.3f} s "
        f"({N_QUERIES / query_time:,.0f} intervals/s)"
    )
    print(
        f"peak RSS after query  : {query_rss / 1024:.1f} MB "
        f"(incremental {max(0, query_rss - baseline_rss) / 1024:.1f} MB)"
    )
    print(f"mapped fragments      : {len(mapped):,}; unmapped: {len(unmapped):,}")
    # Boundary-spanning rows must produce two fragment rows.
    frags_per_row = mapped.groupby("input_row").size()
    assert frags_per_row.max() == 2
    assert (frags_per_row == 2).sum() == half

    # Directly measure the bisection access count on the same arrays.
    t_starts, t_ends, max_end, *_ = chains._block_index["c1"]
    _, _, access = _overlapping_blocks(t_starts, t_ends, max_end, starts, ends)
    naive_scan = N_QUERIES * N_BLOCKS
    print(
        "block positions examined via bisection: "
        f"{access:,} vs. naive rows*blocks scan: {naive_scan:,} "
        f"(ratio {access / naive_scan:.2e})"
    )
    assert access < naive_scan / 1000, "access pattern looks like a full scan"
    # Single-block queries examine ~1 position, boundary ones ~2.
    assert access <= 2 * N_QUERIES

    # Scale-growth check: at constant query density the examined positions
    # per query stay flat as the block array grows.
    print("\nscale growth (blocks grow, one query per block region):")
    for scale in (0.25, 0.5, 1.0):
        nb = int(N_BLOCKS * scale)
        small_map = bf.read_chain(io.StringIO(build_chain_text(nb)))
        ts, te, mend, *_ = small_map._block_index["c1"]
        nq = min(N_QUERIES, nb)
        sub_starts = (np.arange(nq) * (BLOCK_SIZE + GAP)) + 2
        sub_ends = sub_starts + 10
        _, _, acc = _overlapping_blocks(ts, te, mend, sub_starts, sub_ends)
        print(
            f"blocks={nb:>7,} queries={nq:>7,} "
            f"examined={acc:>7,} per-query={acc / nq:.3f}"
        )


if __name__ == "__main__":
    main()
