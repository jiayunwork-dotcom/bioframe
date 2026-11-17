"""
Benchmark for integer-boundary fixes in interval operations.

Measures wall time and peak RSS for overlap/closest/merge/cluster on:

- "normal" batches: 1e5 and 4e5 rows with ordinary genomic-scale coordinates
  and sparse interval relations (scalable interval queries, no all-pairs
  scans);
- "extreme" batches: 1e5 rows with coordinates near +-2**63, also sparse.

Extreme-batch results are verified against pure-Python reference
computations (O(n log n) sweep for merge/cluster; sampled exact checks for
closest/overlap) that do not call bioframe kernels.

Usage:
    python benchmarks/integer_bounds_benchmark.py                 # current bioframe
    PYTHONPATH=/path/to/other/src python benchmarks/integer_bounds_benchmark.py
"""

import gc
import json
import resource
import sys
import time

import numpy as np
import pandas as pd

import bioframe

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


def make_normal(n, seed):
    """Ordinary genomic-scale coordinates, sparse relations."""
    rng = np.random.default_rng(seed)
    chroms = np.array([f"chr{i}" for i in range(1, 20)])
    chrom = rng.choice(chroms, n)
    start = rng.integers(0, 250_000_000, n)
    length = rng.geometric(1 / 2000, n)
    return pd.DataFrame({"chrom": chrom, "start": start, "end": start + length})


def make_extreme(n, seed):
    """Coordinates near the int64 limits, sparse relations.

    A fraction of rows hug the representable boundaries: points at
    +-2**63 / 2**63-1, intervals starting at INT64_MIN or ending at
    INT64_MAX. These exercise the exact wide-arithmetic paths (and break
    the pre-fix implementation, which the verification step detects).
    """
    rng = np.random.default_rng(seed)
    chroms = np.array([f"chr{i}" for i in range(1, 20)])
    chrom = rng.choice(chroms, n)
    anchor = rng.choice(
        [INT64_MIN, INT64_MIN // 2, 0, INT64_MAX // 2, INT64_MAX - 10**9 - 100],
        n,
    )
    start = anchor + rng.integers(0, 10**9, n)
    length = rng.geometric(1 / 2000, n)
    df = pd.DataFrame({"chrom": chrom, "start": start, "end": start + length})
    k = max(4, n // 1000)
    idx = rng.choice(n, 4 * k, replace=False)
    small = rng.integers(0, 1000, k)
    # short intervals ending at INT64_MAX / starting at INT64_MIN, and
    # points exactly at the two limits
    df.loc[idx[:k], "start"] = INT64_MAX - small
    df.loc[idx[:k], "end"] = INT64_MAX
    df.loc[idx[k : 2 * k], "start"] = INT64_MIN
    df.loc[idx[k : 2 * k], "end"] = INT64_MIN + small
    df.loc[idx[2 * k : 3 * k], "start"] = INT64_MAX
    df.loc[idx[2 * k : 3 * k], "end"] = INT64_MAX
    df.loc[idx[3 * k :], "start"] = INT64_MIN
    df.loc[idx[3 * k :], "end"] = INT64_MIN
    return df


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def timed(fn, n_repeat=3):
    """Return (best wall time in s, peak RSS increase in KB, result)."""
    best_t, best_m, out = np.inf, 0, None
    for _ in range(n_repeat):
        gc.collect()
        t0, m0 = time.perf_counter(), peak_rss_kb()
        out = fn()
        dt = time.perf_counter() - t0
        dm = peak_rss_kb() - m0
        best_t, best_m = min(best_t, dt), max(best_m, dm)
    return best_t, best_m, out


# ---------------------------------------------------------------------------
# Pure-Python references (no bioframe kernels)
# ---------------------------------------------------------------------------


def ref_merge_spans(ivs, min_dist=0):
    """O(n log n) reference merge for one chromosome."""
    order = sorted(range(len(ivs)), key=lambda i: (ivs[i][0], ivs[i][1]))
    spans = []
    for i in order:
        s, e = ivs[i]
        if spans and s - spans[-1][1] <= min_dist:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [tuple(x) for x in spans]


def ref_closest_row(s1, e1, ivs2, k):
    """Exact k closest of one interval against a full set, Python ints."""

    def overlaps(a, b):
        (s1, e1), (s2, e2) = a, b
        if s1 == e1 and s2 == e2:
            return s1 == s2
        if s1 == e1:
            return s2 <= s1 < e2
        if s2 == e2:
            return s1 <= s2 < e1
        return s1 < e2 and s2 < e1

    cands = []
    for j, (s2, e2) in enumerate(ivs2):
        if overlaps((s1, e1), (s2, e2)):
            cands.append((0, 0, j))
        else:
            cands.append((1, max(s1 - e2, s2 - e1), j))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    return [(j, g) for _f, g, j in cands[:k]]


def verify_extreme(df1, df2, ov, cl, mg, clu, min_dist=100):
    """Spot-check extreme-batch results with exact Python arithmetic."""
    n_ok = 0
    mg_spans = {
        chrom: {(int(s), int(e)) for s, e in zip(grp["start"], grp["end"])}
        for chrom, grp in mg.groupby("chrom")
    }
    # merge/cluster: full O(n log n) reference per chrom
    for chrom, grp1 in df1.groupby("chrom"):
        ivs = list(zip(grp1["start"].tolist(), grp1["end"].tolist()))
        exp = set(ref_merge_spans(ivs, min_dist=min_dist))
        assert mg_spans.get(chrom, set()) == exp, f"merge mismatch on {chrom}"
        n_ok += 1
    # cluster membership consistent with merge spans
    for chrom, cs, ce in zip(clu["chrom"], clu["cluster_start"], clu["cluster_end"]):
        assert (int(cs), int(ce)) in mg_spans[chrom]
        n_ok += 1
    # closest: exact check for a sample of rows
    rng = np.random.default_rng(0)
    for chrom, grp1 in df1.groupby("chrom"):
        sub2 = df2[df2["chrom"] == chrom]
        ivs2 = list(zip(sub2["start"].tolist(), sub2["end"].tolist()))
        pos2 = sub2.index.to_numpy()
        sample = grp1.iloc[rng.choice(len(grp1), min(3, len(grp1)), replace=False)]
        got = cl[cl["chrom"] == chrom]
        for _, row in sample.iterrows():
            exp = ref_closest_row(int(row["start"]), int(row["end"]), ivs2, 1)
            grow = got[got["index"] == row.name]
            if not exp:
                assert len(grow) == 0 or grow["distance"].isna().all()
                continue
            j_exp, d_exp = exp[0]
            assert int(grow.iloc[0]["distance"]) == d_exp, (
                f"distance mismatch: {grow.iloc[0]['distance']} != {d_exp}"
            )
            assert int(grow.iloc[0]["index_"]) == int(pos2[j_exp])
            n_ok += 1
    # overlap: every reported pair must be an exact overlap (Python ints)
    for s1, e1, s2, e2 in zip(
        ov["start"].head(500),
        ov["end"].head(500),
        ov["start_"].head(500),
        ov["end_"].head(500),
    ):
        s1, e1, s2, e2 = int(s1), int(e1), int(s2), int(e2)
        assert (
            (s1 == e1 and s2 == e2 and s1 == s2)
            or (s1 == e1 and s2 <= s1 < e2)
            or (s2 == e2 and s1 <= s2 < e1)
            or (s1 < e2 and s2 < e1)
        ), f"false overlap {(s1, e1)} {(s2, e2)}"
        n_ok += 1
    return n_ok


def run(tag, n_normal=(100_000, 400_000), n_extreme=100_000):
    report = {"bioframe": bioframe.__file__, "tag": tag, "batches": {}}
    for n in n_normal:
        df1 = make_normal(n, 1)
        df2 = make_normal(n, 2)
        batch = {}
        t, m, ov = timed(
            lambda df1=df1, df2=df2: bioframe.overlap(df1, df2, how="inner")
        )
        batch["overlap"] = {"t": t, "mem_kb": m, "rows": len(ov)}
        t, m, cl = timed(lambda df1=df1, df2=df2: bioframe.closest(df1, df2, k=1))
        batch["closest"] = {"t": t, "mem_kb": m, "rows": len(cl)}
        t, m, mg = timed(lambda df1=df1: bioframe.merge(df1, min_dist=100))
        batch["merge"] = {"t": t, "mem_kb": m, "rows": len(mg)}
        t, m, clu = timed(lambda df1=df1: bioframe.cluster(df1, min_dist=100))
        batch["cluster"] = {"t": t, "mem_kb": m, "rows": len(clu)}
        report["batches"][f"normal_{n}"] = batch
        print(
            f"[{tag}] normal n={n}: "
            + ", ".join(
                f"{op} {v['t']:.3f}s/+{v['mem_kb'] // 1024}MB"
                for op, v in batch.items()
            ),
            flush=True,
        )

    df1 = make_extreme(n_extreme, 3)
    df2 = make_extreme(n_extreme, 4)
    batch = {}
    try:
        t, m, ov = timed(
            lambda: bioframe.overlap(df1, df2, how="inner", return_index=True)
        )
        batch["overlap"] = {"t": t, "mem_kb": m, "rows": len(ov)}
        t, m, cl = timed(lambda: bioframe.closest(df1, df2, k=1, return_index=True))
        batch["closest"] = {"t": t, "mem_kb": m, "rows": len(cl)}
        t, m, mg = timed(lambda: bioframe.merge(df1, min_dist=100))
        batch["merge"] = {"t": t, "mem_kb": m, "rows": len(mg)}
        t, m, clu = timed(lambda: bioframe.cluster(df1, min_dist=100))
        batch["cluster"] = {"t": t, "mem_kb": m, "rows": len(clu)}
        t0 = time.perf_counter()
        try:
            n_ok = verify_extreme(df1, df2, ov, cl, mg, clu)
            batch["verify"] = {
                "t": time.perf_counter() - t0,
                "checks": n_ok,
                "status": "ok",
            }
        except AssertionError as exc:
            batch["verify"] = {
                "t": time.perf_counter() - t0,
                "status": f"FAILED: {exc}",
            }
    except Exception as exc:  # noqa: BLE001 - record any pre-fix failure
        batch["error"] = f"{type(exc).__name__}: {exc}"
    report["batches"][f"extreme_{n_extreme}"] = batch
    print(
        f"[{tag}] extreme n={n_extreme}: "
        + ", ".join(
            f"{op} {v['t']:.3f}s/+{v.get('mem_kb', 0) // 1024}MB"
            for op, v in batch.items()
            if isinstance(v, dict) and "t" in v
        )
        or f"[{tag}] extreme n={n_extreme}: {batch}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    rep = run(tag)
    out = f"/tmp/bench_{tag}.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", out)
