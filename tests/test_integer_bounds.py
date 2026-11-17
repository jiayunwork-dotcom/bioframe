"""
Independent ground-truth tests for integer-boundary behavior of interval
operations.

The reference implementations in this file are written with unbounded
Python integers and never call bioframe's interval kernels
(:mod:`bioframe.core.arrops`) to generate expected results. They encode the
documented semantics directly:

- intervals are semi-open [start, end); an interval with start == end is a
  point, which overlaps [s, e) iff s <= p < e; two points overlap iff equal;
- overlap/cluster/merge group intervals per chromosome (and per `on` column);
- closest reports, per interval of df1, the k nearest intervals of df2:
  overlapping pairs first (distance 0), then by exact gap
  max(start1 - end2, start2 - end1); ties broken by tie_breaking_col (lower
  first) and then by the lower row position in df2;
- merge joins intervals whose gap is <= min_dist (min_dist=None joins only
  overlapping intervals).

Coordinates cover the full signed int64 range [-2**63, 2**63 - 1].
"""

import numpy as np
import pandas as pd
import pytest

import bioframe
from bioframe.core import arrops

P = 2**53
M = 2**63 - 1
MIN = -(2**63)


def bf(rows, cols=("chrom", "start", "end"), index=None):
    df = pd.DataFrame(rows, columns=list(cols))
    if index is not None:
        df.index = index
    return df


# ---------------------------------------------------------------------------
# Pure-Python reference implementations (ground truth, no bioframe kernels)
# ---------------------------------------------------------------------------


def ref_is_valid(s, e):
    return s <= e


def ref_overlaps_pair(a, b):
    (s1, e1), (s2, e2) = a, b
    p1, p2 = s1 == e1, s2 == e2
    if p1 and p2:
        return s1 == s2
    if p1:
        return s2 <= s1 < e2
    if p2:
        return s1 <= s2 < e1
    return s1 < e2 and s2 < e1


def ref_overlap_pairs(ivs1, ivs2):
    """Set of (i, j) physical-row pairs that overlap."""
    return {
        (i, j)
        for i, a in enumerate(ivs1)
        for j, b in enumerate(ivs2)
        if a[0] == b[0] and ref_overlaps_pair(a[1:], b[1:])
    }


def ref_gap(a, b):
    """bioframe distance between two intervals on the same chrom."""
    (s1, e1), (s2, e2) = a, b
    if ref_overlaps_pair(a, b):
        return 0
    return max(s1 - e2, s2 - e1)


def ref_closest(
    ivs1,
    ivs2,
    k=1,
    ignore_overlaps=False,
    ignore_upstream=False,
    ignore_downstream=False,
    strands1=None,
    tie=None,
):
    """
    For each row of ivs1, the ordered list of (j, distance) of the k closest
    rows of ivs2, following bioframe's documented contract:

    - overlapping pairs are candidates (distance 0), unless ignore_overlaps;
    - pairs touching by a single point (end2 == start1 or start2 == end1) are
      directional candidates even when they also overlap;
    - per direction, only the k nearest candidates are kept (largest end2 on
      the left, smallest start2 on the right; ties prefer lower tie values,
      then lower df2 positions);
    - final order: overlaps first, then by exact gap, then tie value, then
      df2 position; the first k win.
    """
    out = []
    for i, (c1, s1, e1) in enumerate(ivs1):
        plus = True if strands1 is None else strands1[i] != "-"
        use_left = not (ignore_upstream if plus else ignore_downstream)
        use_right = not (ignore_downstream if plus else ignore_upstream)
        ov_cands, left_cands, right_cands = [], [], []
        for j, (c2, s2, e2) in enumerate(ivs2):
            if c1 != c2:
                continue
            t = 0 if tie is None else tie[j]
            if ref_overlaps_pair((s1, e1), (s2, e2)) and not ignore_overlaps:
                ov_cands.append(j)
            if e2 <= s1:
                left_cands.append((e2, t, j))
            if s2 >= e1:
                right_cands.append((s2, t, j))
        left_cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        right_cands.sort(key=lambda c: (c[0], c[1], c[2]))
        cands = [(0, 0, 0 if tie is None else tie[j], j) for j in ov_cands]
        if use_left:
            cands += [(1, s1 - e2, t, j) for e2, t, j in left_cands[:k]]
        if use_right:
            cands += [(1, s2 - e1, t, j) for s2, t, j in right_cands[:k]]
        cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
        out.append([(j, ref_gap((s1, e1), ivs2[j][1:])) for _f, _g, _t, j in cands[:k]])
    return out


def ref_merge(ivs, min_dist=0):
    """
    Merge one chromosome's intervals [(start, end), ...].
    Returns (cluster_of_row, [(cluster_start, cluster_end), ...]).
    """
    order = sorted(range(len(ivs)), key=lambda i: (ivs[i][0], ivs[i][1]))
    clusters = []  # [start, max_end, [rows]]
    for i in order:
        s, e = ivs[i]
        if clusters:
            cur_end = clusters[-1][1]
            joins = s < cur_end if min_dist is None else s - cur_end <= min_dist
        else:
            joins = False
        if joins:
            clusters[-1][1] = max(clusters[-1][1], e)
            clusters[-1][2].append(i)
        else:
            clusters.append([s, e, [i]])
    cluster_of = {}
    for cid, (_s, _e, rows) in enumerate(clusters):
        for r in rows:
            cluster_of[r] = cid
    return [cluster_of[i] for i in range(len(ivs))], [(c[0], c[1]) for c in clusters]


# ---------------------------------------------------------------------------
# Reproductions of the reported bugs
# ---------------------------------------------------------------------------


def test_closest_prefers_truly_nearest_beyond_float_precision():
    # k=1 must select [0,1) at distance P+2, not [2P+7,2P+8) at distance P+3.
    df1 = bf([("x", P + 3, P + 4)])
    df2 = bf([("x", 2 * P + 7, 2 * P + 8), ("x", 0, 1)])
    out = bioframe.closest(df1, df2, k=1, return_index=True)
    assert out.loc[0, "index_"] == 1
    assert out.loc[0, "distance"] == P + 2
    assert out.loc[0, "start_"] == 0 and out.loc[0, "end_"] == 1


def test_closest_distance_spanning_full_int64_range():
    df1 = bf([("x", MIN, MIN + 1)])
    df2 = bf([("x", M - 1, M)])
    out = bioframe.closest(df1, df2, k=1)
    expected = (M - 1) - (MIN + 1)  # 2**64 - 3, exceeds int64
    assert out.loc[0, "distance"] == expected
    # distances that do not fit int64 are returned as exact UInt64
    assert out["distance"].dtype == pd.UInt64Dtype()
    # kernel-level agreement
    ids1, ids2 = arrops.closest_intervals(
        np.array([MIN]), np.array([MIN + 1]), np.array([M - 1]), np.array([M])
    )
    assert (list(ids1), list(ids2)) == ([0], [0])


def test_is_bedframe_boundaries():
    assert bioframe.is_bedframe(bf([("x", -1, M)]))
    assert bioframe.is_bedframe(bf([("x", MIN, M)]))
    assert bioframe.is_bedframe(bf([("x", M, M)]))  # point at M
    assert bioframe.is_bedframe(bf([("x", MIN, MIN)]))  # point at MIN
    assert not bioframe.is_bedframe(bf([("x", M, MIN)]))
    assert not bioframe.is_bedframe(bf([("x", 5, -5)]))
    assert not bioframe.is_bedframe(bf([("x", M, M - 1)]))
    assert not bioframe.is_bedframe(bf([("x", MIN + 1, MIN)]))
    with pytest.raises(ValueError):
        bioframe.is_bedframe(bf([("x", M, MIN)]), raise_errors=True)


def test_merge_min_dist_near_int64_max():
    df = bf([("x", M - 2, M - 1), ("x", M - 1, M)])
    out = bioframe.merge(df, min_dist=3)
    assert len(out) == 1
    assert out.loc[0, "start"] == M - 2 and out.loc[0, "end"] == M
    assert out.loc[0, "n_intervals"] == 2
    # cluster must agree with merge
    cl = bioframe.cluster(df, min_dist=3)
    assert cl["cluster"].nunique() == 1
    assert (cl["cluster_start"] == M - 2).all() and (cl["cluster_end"] == M).all()


def test_closest_return_index_preserves_large_labels_with_missing_chrom():
    df1 = bf([("x", 0, 1), ("y", 0, 1)])
    df2 = bf([("x", 0, 1)], index=[P + 5])
    out = bioframe.closest(df1, df2, return_index=True)
    by_chrom = out.set_index("chrom")
    assert by_chrom.loc["x", "index_"] == P + 5  # not rounded to P+4
    assert pd.isna(by_chrom.loc["y", "index_"])
    assert out["index_"].dtype == pd.Int64Dtype()


# ---------------------------------------------------------------------------
# Validation vs brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start,end,valid",
    [
        (MIN, MIN, True),
        (MIN, MIN + 1, True),
        (MIN, M, True),
        (-1, 1, True),
        (0, 0, True),
        (M, M, True),
        (M - 1, M, True),
        (M, M - 1, False),
        (0, -1, False),
        (MIN + 1, MIN, False),
        (M, MIN, False),
    ],
)
def test_is_bedframe_exhaustive_boundaries(start, end, valid):
    df = bf([("x", start, end)])
    assert bioframe.is_bedframe(df) == valid
    # nullable Int64 coordinates give the same answer
    df_nullable = df.astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()})
    assert bioframe.is_bedframe(df_nullable) == valid


def test_is_bedframe_null_rows_still_valid():
    df = pd.DataFrame(
        [[pd.NA, pd.NA, pd.NA], ["x", 1, 2]], columns=["chrom", "start", "end"]
    ).astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()})
    assert bioframe.is_bedframe(df)


# ---------------------------------------------------------------------------
# Overlap vs brute force
# ---------------------------------------------------------------------------

# Coordinates exercising both extremes, zero-crossing, points and touching.
EDGE_COORDS = [MIN, MIN + 1, -P - 1, -P, -2, -1, 0, 1, 2, P, P + 1, M - 1, M]


def _sample_intervals(seed, n, coords=EDGE_COORDS, max_len=None):
    rng = np.random.default_rng(seed)
    ivs = []
    for _ in range(n):
        s = int(rng.choice(coords))
        max_e = M
        if max_len is not None:
            max_e = min(M, s + max_len)
        hi = [c for c in coords if s <= c <= max_e]
        e = int(rng.choice(hi))
        ivs.append((s, e))
    return ivs


@pytest.mark.parametrize("seed", range(6))
def test_overlap_matches_bruteforce(seed):
    ivs1 = [("x", *iv) for iv in _sample_intervals(seed, 12)]
    ivs2 = [("x", *iv) for iv in _sample_intervals(seed + 100, 11)]
    # add a second chrom that must never cross-match
    ivs1 += [("y", *iv) for iv in _sample_intervals(seed + 200, 4)]
    ivs2 += [("y", *iv) for iv in _sample_intervals(seed + 300, 3)]
    df1, df2 = bf(ivs1), bf(ivs2)

    expected = ref_overlap_pairs(ivs1, ivs2)
    for how in ["inner", "left", "right", "outer"]:
        out = bioframe.overlap(df1, df2, how=how, return_index=True)
        got = {
            (int(r["index"]), int(r["index_"]))
            for _, r in out.iterrows()
            if not pd.isna(r["index_"]) and not pd.isna(r["index"])
        }
        assert got == expected, f"how={how}"
        # unpaired rows appear exactly on the required side(s)
        if how in ("left", "outer"):
            assert {i for i, _ in got} | {
                int(r["index"]) for _, r in out.iterrows() if pd.isna(r["index_"])
            } == set(range(len(df1)))
        if how in ("right", "outer"):
            assert {j for _, j in got} | {
                int(r["index_"]) for _, r in out.iterrows() if pd.isna(r["index"])
            } == set(range(len(df2)))

    # kernel-level agreement (the kernel is chromosome-agnostic: run it on
    # the single-chromosome subsets)
    x1 = [iv[1:] for iv in ivs1 if iv[0] == "x"]
    x2 = [iv[1:] for iv in ivs2 if iv[0] == "x"]
    k1, k2 = arrops.overlap_intervals(
        np.array([s for s, _ in x1]),
        np.array([e for _, e in x1]),
        np.array([s for s, _ in x2]),
        np.array([e for _, e in x2]),
    )
    assert set(zip(k1.tolist(), k2.tolist())) == ref_overlap_pairs(
        [("x", *iv) for iv in x1], [("x", *iv) for iv in x2]
    )


def test_overlap_return_overlap_bounds():
    ivs1 = [("x", MIN, 0), ("x", 5, M), ("x", M, M), ("x", 3, 3)]
    ivs2 = [("x", -5, 10), ("x", M - 3, M), ("y", 0, 10)]
    df1, df2 = bf(ivs1), bf(ivs2)
    out = bioframe.overlap(df1, df2, how="left", return_index=True, return_overlap=True)
    for _, r in out.iterrows():
        i, j = int(r["index"]), r["index_"]
        if pd.isna(j):
            assert pd.isna(r["overlap_start"]) and pd.isna(r["overlap_end"])
            continue
        j = int(j)
        s_exp = max(ivs1[i][1], ivs2[j][1])
        e_exp = min(ivs1[i][2], ivs2[j][2])
        assert int(r["overlap_start"]) == s_exp
        assert int(r["overlap_end"]) == e_exp


def test_overlap_custom_colnames_and_suffixes():
    cols1 = ("chr", "lo", "hi")
    cols2 = ("chromosome", "begin", "stop")
    df1 = bf([("x", MIN, MIN + 5), ("x", 0, 10)], cols=cols1)
    df2 = bf([("x", MIN + 3, 0), ("x", 9, M)], cols=cols2)
    out = bioframe.overlap(
        df1,
        df2,
        how="inner",
        cols1=cols1,
        cols2=cols2,
        suffixes=("_a", "_b"),
        return_index=True,
        return_overlap="ov",
    )
    pairs = {(int(r["index_a"]), int(r["index_b"])) for _, r in out.iterrows()}
    assert pairs == ref_overlap_pairs(
        [("x", MIN, MIN + 5), ("x", 0, 10)], [("x", MIN + 3, 0), ("x", 9, M)]
    )
    assert {"ov_lo", "ov_hi"} <= set(out.columns)
    row = out[(out["index_a"] == 0) & (out["index_b"] == 0)].iloc[0]
    # intersection of [MIN, MIN+5) and [MIN+3, 0) is [MIN+3, MIN+5)
    assert int(row["ov_lo"]) == MIN + 3 and int(row["ov_hi"]) == MIN + 5


def test_overlap_point_at_int64_max_finds_itself():
    df = bf([("x", M, M)])
    out = bioframe.overlap(df, df, how="inner", return_index=True)
    assert len(out) == 1
    assert int(out.loc[0, "index"]) == 0 and int(out.loc[0, "index_"]) == 0
    # coordinates are not shifted
    assert int(out.loc[0, "start"]) == M and int(out.loc[0, "start_"]) == M
    # kernel level
    ids1, ids2 = arrops.overlap_intervals(
        np.array([M]), np.array([M]), np.array([M]), np.array([M])
    )
    assert list(ids1) == [0] and list(ids2) == [0]
    # a point at M does not overlap an interval ending at M (semi-open)
    df2 = bf([("x", M - 5, M)])
    assert len(bioframe.overlap(df, df2, how="inner")) == 0
    # points at MIN behave as usual points
    dp = bf([("x", MIN, MIN)])
    di = bf([("x", MIN, MIN + 10)])
    assert len(bioframe.overlap(dp, di, how="inner")) == 1


def test_overlap_points_and_touching():
    ivs1 = [("x", 5, 5), ("x", 0, 5), ("x", 5, 10)]
    ivs2 = [("x", 5, 5), ("x", 5, 6), ("x", 4, 5), ("x", 10, 10)]
    df1, df2 = bf(ivs1), bf(ivs2)
    out = bioframe.overlap(df1, df2, how="inner", return_index=True)
    got = {(int(r["index"]), int(r["index_"])) for _, r in out.iterrows()}
    assert got == ref_overlap_pairs(ivs1, ivs2)
    # hand-checked: point 5 equals point 5 and lies in [5,6); [0,5) meets
    # [4,5); [5,10) contains point 5 and meets [5,6); touching pairs
    # ([0,5)/[5,6), [5,10)/point 10) do not overlap.
    assert got == {(0, 0), (0, 1), (1, 2), (2, 0), (2, 1)}


# ---------------------------------------------------------------------------
# Closest vs brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [1, 2, 5])
def test_closest_matches_bruteforce(seed, k):
    ivs1 = [("x", *iv) for iv in _sample_intervals(seed, 8, max_len=P)]
    ivs2 = [("x", *iv) for iv in _sample_intervals(seed + 50, 9, max_len=P)]
    ivs1 += [("y", *iv) for iv in _sample_intervals(seed + 60, 3, max_len=10)]
    df1, df2 = bf(ivs1), bf(ivs2)

    expected = ref_closest(ivs1, ivs2, k=k)
    out = bioframe.closest(df1, df2, k=k, return_index=True)
    got = {}
    for _, r in out.iterrows():
        i = int(r["index"])
        got.setdefault(i, []).append(
            (
                None if pd.isna(r["index_"]) else int(r["index_"]),
                None if pd.isna(r["distance"]) else int(r["distance"]),
            )
        )
    for i in range(len(ivs1)):
        exp = expected[i]
        got_i = got.get(i, [])
        if not exp:
            assert got_i == [(None, None)] or got_i == []
            continue
        assert [j for j, _ in got_i] == [j for j, _ in exp]
        assert [d for _, d in got_i] == [d for _, d in exp]


def test_closest_equidistant_and_off_by_one_neighbors():
    # truly equidistant neighbors: lowest df2 row position wins
    df1 = bf([("x", 10, 20)])
    df2 = bf([("x", 0, 5), ("x", 25, 30), ("x", 26, 31)])  # gaps 5, 5, 6
    out = bioframe.closest(df1, df2, k=1, return_index=True)
    assert int(out.loc[0, "index_"]) == 0 and int(out.loc[0, "distance"]) == 5
    out = bioframe.closest(df1, df2, k=3, return_index=True)
    assert [int(v) for v in out["index_"]] == [0, 1, 2]
    assert [int(v) for v in out["distance"]] == [5, 5, 6]

    # differ-by-one at the float-precision boundary: 2**53+1 vs 2**53+2
    df1 = bf([("x", 0, 1)])
    df2 = bf([("x", P + 2, P + 3), ("x", P + 3, P + 4)])  # gaps P+1, P+2
    out = bioframe.closest(df1, df2, k=1, return_index=True)
    assert int(out.loc[0, "index_"]) == 0
    assert int(out.loc[0, "distance"]) == P + 1


def test_closest_tie_breaking_col():
    df1 = bf([("x", 10, 20, 0)], cols=("chrom", "start", "end", "unused"))
    df2 = pd.DataFrame(
        [("x", 0, 5, 7), ("x", 25, 30, 3)], columns=["chrom", "start", "end", "score"]
    )
    out = bioframe.closest(df1, df2, k=1, tie_breaking_col="score", return_index=True)
    # both at distance 5; lower score (row 1) wins
    assert int(out.loc[0, "index_"]) == 1
    # but a truly nearer interval beats any tie-break value
    df2.loc[2] = ("x", 22, 24, 100)
    out = bioframe.closest(df1, df2, k=1, tie_breaking_col="score", return_index=True)
    assert int(out.loc[0, "index_"]) == 2
    assert int(out.loc[0, "distance"]) == 2


def test_closest_upstream_downstream_and_strand():
    df1 = pd.DataFrame(
        [("x", 100, 110, "+"), ("x", 100, 110, "-")],
        columns=["chrom", "start", "end", "strand"],
    )
    df2 = bf([("x", 0, 50), ("x", 200, 250)])
    out = bioframe.closest(
        df1,
        df2,
        k=1,
        direction_col="strand",
        ignore_upstream=True,
        return_index=True,
    )
    # '+' ignores upstream (left): gets right interval; '-' flips: gets left
    by_strand = out.set_index("strand")
    assert int(by_strand.loc["+", "index_"]) == 1
    assert int(by_strand.loc["-", "index_"]) == 0
    out = bioframe.closest(
        df1,
        df2,
        k=1,
        direction_col="strand",
        ignore_downstream=True,
        return_index=True,
    )
    by_strand = out.set_index("strand")
    assert int(by_strand.loc["+", "index_"]) == 0
    assert int(by_strand.loc["-", "index_"]) == 1
    # ignore_overlaps
    df2b = bf([("x", 100, 110), ("x", 0, 50)])
    out = bioframe.closest(df1, df2b, k=1, ignore_overlaps=True, return_index=True)
    assert (out["index_"] == 1).all()
    assert (out["distance"] == 50).all()


def test_closest_no_match_and_empty_df2():
    df1 = bf([("x", 0, 1), ("y", 2, 3)])
    df2 = bf([("x", 5, 6)])
    out = bioframe.closest(df1, df2, return_index=True)
    by_chrom = out.set_index("chrom")
    assert int(by_chrom.loc["x", "index_"]) == 0
    assert pd.isna(by_chrom.loc["y", "index_"])
    assert pd.isna(by_chrom.loc["y", "distance"])
    # empty df2: every row unmatched (previously raised IndexError)
    empty = bf([], cols=("chrom", "start", "end")).astype(
        {"start": np.int64, "end": np.int64}
    )
    out = bioframe.closest(df1, empty, return_index=True)
    assert len(out) == 2
    assert out["index_"].isna().all() and out["distance"].isna().all()
    out = bioframe.overlap(df1, empty, how="left", return_index=True)
    assert len(out) == 2 and out["index_"].isna().all()


def test_closest_duplicate_index_labels():
    df1 = bf([("x", 10, 20)], index=[100])
    df2 = bf([("x", 0, 5), ("x", 30, 40), ("x", 50, 60)], index=[7, 7, 9])
    out = bioframe.closest(df1, df2, k=3, return_index=True)
    # physical rows are reported separately despite duplicate labels
    assert [int(v) for v in out["index_"]] == [7, 7, 9]
    assert [int(v) for v in out["distance"]] == [5, 10, 30]
    assert [int(v) for v in out["start_"]] == [0, 30, 50]


def test_closest_nullable_coordinates_match_plain():
    rows1 = [("x", MIN, MIN + 10), ("x", -5, 5), ("x", M - 10, M)]
    rows2 = [("x", MIN + 3, MIN + 4), ("x", 0, 1), ("x", M - 1, M)]
    df1, df2 = bf(rows1), bf(rows2)
    expected = bioframe.closest(df1, df2, k=3, return_index=True)

    def with_nulls(df):
        df = df.astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()})
        return pd.concat(
            [
                df,
                pd.DataFrame(
                    [[pd.NA, pd.NA, pd.NA]], columns=["chrom", "start", "end"]
                ).astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()}),
            ],
            ignore_index=True,
        )

    got = bioframe.closest(with_nulls(df1), with_nulls(df2), k=3, return_index=True)
    # matched labels and distances are unchanged by the presence of null rows
    # (coordinate columns become nullable Int64 when nulls are present)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
    )
    assert list(got["index_"]) == list(expected["index_"])
    assert list(got["distance"]) == list(expected["distance"])


def test_closest_distance_dtype_policy():
    # fits int64 -> Int64
    out = bioframe.closest(bf([("x", 0, 1)]), bf([("x", 5, 6)]))
    assert out["distance"].dtype == pd.Int64Dtype()
    assert int(out.loc[0, "distance"]) == 4
    # exceeds int64 -> exact UInt64
    out = bioframe.closest(bf([("x", MIN, MIN + 1)]), bf([("x", M - 1, M)]))
    assert out["distance"].dtype == pd.UInt64Dtype()
    assert int(out.loc[0, "distance"]) == 2**64 - 3
    # largest representable distance: 2**64 - 1
    out = bioframe.closest(
        bf([("x", MIN, MIN)]), bf([("x", M, M)]), ignore_overlaps=True
    )
    assert int(out.loc[0, "distance"]) == 2**64 - 1


# ---------------------------------------------------------------------------
# Merge / cluster vs brute force, and mutual consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("min_dist", [None, 0, 1, 3, P])
def test_merge_cluster_match_bruteforce(seed, min_dist):
    per_chrom = {}
    for chrom, off in [("x", 0), ("y", 100)]:
        per_chrom[chrom] = _sample_intervals(seed + off, 10)
    rows = [(c, s, e) for c, ivs in per_chrom.items() for (s, e) in ivs]
    df = bf(rows)

    merged = bioframe.merge(df, min_dist=min_dist)
    clustered = bioframe.cluster(df, min_dist=min_dist)

    # reference grouping per chrom
    ref_groups = {}  # (chrom, ref cluster id) -> merged span
    row_ref_cluster = {}
    for c, ivs in per_chrom.items():
        ids, spans = ref_merge(ivs, min_dist=min_dist)
        for row, cid in enumerate(ids):
            row_ref_cluster[(c, row)] = cid
            ref_groups[(c, cid)] = spans[cid]

    # merge: the set of spans per chrom must match the reference exactly
    got_spans = {
        (r["chrom"], int(r["start"]), int(r["end"])) for _, r in merged.iterrows()
    }
    exp_spans = {(c, s, e) for (c, _cid), (s, e) in ref_groups.items()}
    assert got_spans == exp_spans

    # cluster: membership must equal merge grouping and reference grouping
    by_cluster = {}
    for pos, (_, r) in enumerate(clustered.iterrows()):
        by_cluster.setdefault((r["chrom"], int(r["cluster"])), set()).add(pos)
        span = (int(r["cluster_start"]), int(r["cluster_end"]))
        row_in_chrom = _row_in_chrom(per_chrom, r["chrom"], pos, rows)
        cid = row_ref_cluster[(r["chrom"], row_in_chrom)]
        assert span == ref_groups[(r["chrom"], cid)]
    # each cluster's span equals the merged span of its members and appears
    # in the merge output
    merge_spans = {(c, s, e) for c, s, e in got_spans}
    for (c, _cid), members in by_cluster.items():
        s = min(rows[m][1] for m in members)
        e = max(rows[m][2] for m in members)
        cs = clustered.iloc[sorted(members)][["cluster_start", "cluster_end"]].iloc[0]
        assert (int(cs["cluster_start"]), int(cs["cluster_end"])) == (s, e)
        assert (c, int(cs["cluster_start"]), int(cs["cluster_end"])) in merge_spans


def _row_in_chrom(per_chrom, chrom, pos, rows):
    """Map a physical row position to its index within its chromosome."""
    count = 0
    for i, (c, _s, _e) in enumerate(rows):
        if c == chrom:
            if i == pos:
                return count
            count += 1
    raise AssertionError("row not found")


def test_merge_cluster_extreme_boundaries():
    df = bf(
        [
            ("x", MIN, MIN + 2),
            ("x", MIN + 2, MIN + 3),  # touching the first
            ("x", M - 2, M - 1),
            ("x", M - 1, M),  # touching at the top
            ("x", 0, 1),
        ]
    )
    for min_dist, n_exp in [(None, 5), (0, 3), (1, 3), (2, 3)]:
        merged = bioframe.merge(df, min_dist=min_dist)
        cl = bioframe.cluster(df, min_dist=min_dist)
        assert len(merged) == n_exp
        assert cl["cluster"].nunique() == n_exp
        # cluster spans identical to merge spans
        assert {tuple(r) for r in merged[["start", "end"]].values} == {
            (int(s), int(e))
            for s, e in cl[["cluster_start", "cluster_end"]].drop_duplicates().values
        }
    merged = bioframe.merge(df, min_dist=0)
    spans = {(int(r["start"]), int(r["end"])) for _, r in merged.iterrows()}
    assert spans == {(MIN, MIN + 3), (M - 2, M), (0, 1)}


def test_merge_min_dist_float_and_large():
    df = bf([("x", 0, 5), ("x", 7, 9), ("x", 20, 21)])
    # gap of 2: merged with min_dist=2.5 (floor semantics for integer gaps)
    assert len(bioframe.merge(df, min_dist=2.5)) == 2
    assert len(bioframe.merge(df, min_dist=1.5)) == 3
    # huge min_dist merges everything, no overflow
    assert len(bioframe.merge(df, min_dist=2**63 - 1)) == 1
    assert len(bioframe.merge(df, min_dist=10**30)) == 1
    # negative min_dist is rejected at the ops level
    with pytest.raises(ValueError):
        bioframe.merge(df, min_dist=-1)
    with pytest.raises(ValueError):
        bioframe.cluster(df, min_dist=-1)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_translation_invariance():
    rng = np.random.default_rng(0)
    base = [
        (int(s), int(s) + int(length))
        for s, length in zip(
            rng.integers(-(2**40), 2**40, 40), rng.integers(0, 1000, 40)
        )
    ]
    ivs1 = sorted(base[:20])
    ivs2 = sorted(base[20:])
    delta = 2**45  # everything stays within int64

    df1, df2 = bf([("x", *iv) for iv in ivs1]), bf([("x", *iv) for iv in ivs2])
    df1s = bf([("x", s + delta, e + delta) for s, e in ivs1])
    df2s = bf([("x", s + delta, e + delta) for s, e in ivs2])

    ov_a = bioframe.overlap(df1, df2, how="inner", return_index=True)
    ov_b = bioframe.overlap(df1s, df2s, how="inner", return_index=True)
    assert {(int(r["index"]), int(r["index_"])) for _, r in ov_a.iterrows()} == {
        (int(r["index"]), int(r["index_"])) for _, r in ov_b.iterrows()
    }

    cl_a = bioframe.closest(df1, df2, k=3, return_index=True)
    cl_b = bioframe.closest(df1s, df2s, k=3, return_index=True)
    assert [int(v) for v in cl_a["index_"]] == [int(v) for v in cl_b["index_"]]
    assert [int(v) for v in cl_a["distance"]] == [int(v) for v in cl_b["distance"]]

    mg_a = bioframe.merge(df1, min_dist=7)[["start", "end"]].values
    mg_b = bioframe.merge(df1s, min_dist=7)[["start", "end"]].values
    assert (mg_b - mg_a == delta).all()


def test_row_order_permutation_physical_identity():
    rows1 = [("x", *iv) for iv in _sample_intervals(1, 10, max_len=100)]
    rows2 = [("x", *iv) for iv in _sample_intervals(2, 10, max_len=100)]
    perm1 = [3, 0, 7, 1, 9, 2, 8, 4, 6, 5]
    perm2 = [5, 2, 9, 0, 8, 1, 7, 3, 6, 4]
    df1, df2 = bf(rows1), bf(rows2)
    df1p = df1.iloc[perm1].reset_index(drop=True)
    df2p = df2.iloc[perm2].reset_index(drop=True)

    def pair_names(out, name1, name2):
        return {
            (r[name1], r[name2])
            for _, r in out.iterrows()
            if pd.notna(r[name1]) and pd.notna(r[name2])
        }

    tag1 = {i: f"a{i}" for i in range(10)}
    tag2 = {j: f"b{j}" for j in range(10)}
    df1["n"], df2["n"] = df1.index.map(tag1), df2.index.map(tag2)
    df1p["n"] = [tag1[i] for i in perm1]
    df2p["n"] = [tag2[j] for j in perm2]

    ov_a = bioframe.overlap(df1, df2, how="inner", suffixes=("_1", "_2"))
    ov_b = bioframe.overlap(df1p, df2p, how="inner", suffixes=("_1", "_2"))
    assert pair_names(ov_a, "n_1", "n_2") == pair_names(ov_b, "n_1", "n_2")

    # Closest: distances per physical df1 row are row-order invariant, and
    # every reported (n_1, n_2, distance) triple must be exact: the distance
    # must equal the true gap between those two physical intervals (ties may
    # legitimately resolve to different equidistant rows after permutation).
    iv_by_name = {}
    for i in range(10):
        iv_by_name[tag1[i]] = (rows1[i][1], rows1[i][2])
        iv_by_name[tag2[i]] = (rows2[i][1], rows2[i][2])
    dist_seqs = []
    for cl in (
        bioframe.closest(df1, df2, k=2, suffixes=("_1", "_2")),
        bioframe.closest(df1p, df2p, k=2, suffixes=("_1", "_2")),
    ):
        for _, r in cl.iterrows():
            exact = ref_gap(iv_by_name[r["n_1"]], iv_by_name[r["n_2"]])
            assert int(r["distance"]) == exact
        dist_seqs.append(
            cl.groupby("n_1")["distance"]
            .apply(lambda s: tuple(int(v) for v in s))
            .to_dict()
        )
    assert dist_seqs[0] == dist_seqs[1]


def test_inputs_are_not_mutated():
    df1 = bf([("x", 0, 5), ("x", 10, 15), ("y", 0, 3)], index=[10, 20, 30])
    df2 = bf([("x", 3, 12), ("x", M - 2, M)], index=[5, 6])
    snapshot = [(d.copy(), d.index.copy(), d.dtypes.copy()) for d in (df1, df2)]

    bioframe.overlap(df1, df2, how="outer", return_index=True, return_overlap=True)
    bioframe.closest(df1, df2, k=2, return_index=True, return_overlap=True)
    bioframe.merge(df1, min_dist=3)
    bioframe.cluster(df1, min_dist=3)
    bioframe.coverage(df1, df2)
    bioframe.count_overlaps(df1, df2)
    bioframe.is_bedframe(df1)

    for d, (snap, idx, dtypes) in zip((df1, df2), snapshot):
        pd.testing.assert_frame_equal(d, snap)
        assert d.index.equals(idx)
        assert d.dtypes.equals(dtypes)

    # low-level array entries do not mutate either
    arrs = [np.array([MIN, -1, M - 1]), np.array([MIN + 1, 0, M])]
    copies = [a.copy() for a in arrs]
    arrops.overlap_intervals(*arrs, *arrs)
    arrops.merge_intervals(*arrs, min_dist=3)
    arrops.closest_intervals(*arrs, *arrs)
    for a, c in zip(arrs, copies):
        assert (a == c).all()


def test_all_null_rows_behave_as_before():
    df1 = pd.DataFrame(
        [[pd.NA, pd.NA, pd.NA], ["x", 1, 5]], columns=["chrom", "start", "end"]
    ).astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()})
    df2 = pd.DataFrame(
        [[pd.NA, pd.NA, pd.NA], ["x", 4, 8]], columns=["chrom", "start", "end"]
    ).astype({"start": pd.Int64Dtype(), "end": pd.Int64Dtype()})
    out = bioframe.overlap(df1, df2, how="left")
    assert len(out) == 2
    assert pd.isna(out.loc[0, "chrom_"])
    assert out.loc[1, "start_"] == 4
    merged = bioframe.merge(df1)
    assert len(merged) == 2  # one real row, one NA row
    cl = bioframe.cluster(df1)
    assert len(cl) == 2


def test_is_overlapping_exact_at_extremes():
    # two huge overlapping intervals: naive int64 length sums wrap around
    df = bf([("x", MIN, M), ("x", MIN, M)])
    assert bioframe.is_overlapping(df)
    df = bf([("x", MIN, 0), ("x", 0, M)])
    assert not bioframe.is_overlapping(df)
    df = bf([("x", -1, M)])
    assert not bioframe.is_overlapping(df)
