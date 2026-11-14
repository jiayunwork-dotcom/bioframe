"""
Tests for UCSC chain reading and interval liftover (bioframe.read_chain /
bioframe.liftover).

The randomized checks build an independent, base-by-base ground truth (no
block arithmetic shared with the implementation) and compare the full
fragment tables, coverage fractions and mapped/unmapped verdicts against it.
"""

import gzip
import io

import numpy as np
import pandas as pd
import pytest

import bioframe as bf

MAPPED_COLS = [
    "input_row",
    "chain_id",
    "block",
    "source_start",
    "source_end",
    "chrom",
    "start",
    "end",
    "strand",
    "match_fraction",
]
UNMAPPED_COLS = ["input_row", "reason", "best_fraction", "candidate_count"]


# ---------------------------------------------------------------------------
# Chain construction helpers (independent of the parser)
# ---------------------------------------------------------------------------
def build_chain(
    chain_id,
    t_name,
    t_size,
    q_name,
    q_size,
    fwd_blocks,
    q_strand="+",
    score=100,
    t_pad_left=0,
    q_pad_left=0,
):
    """Build a chain text block from forward-coordinate blocks.

    fwd_blocks : list of (t_start, t_end, q_start, q_end) tuples in target
                 order with forward coordinates on both sides.
    t_pad_left / q_pad_left : shift the whole chain away from coordinate 0
                 (gaps on both sides of the header span).
    """
    fwd_blocks = [
        (ts + t_pad_left, te + t_pad_left, qs, qe)
        for (ts, te, qs, qe) in fwd_blocks
    ]
    t_start = fwd_blocks[0][0]
    t_end = fwd_blocks[-1][1]
    if q_strand == "+":
        q_start = fwd_blocks[0][2] + q_pad_left
        fwd_blocks = [
            (ts, te, qs + q_pad_left, qe + q_pad_left)
            for (ts, te, qs, qe) in fwd_blocks
        ]
        q_end = fwd_blocks[-1][3]
    else:
        q_end = fwd_blocks[0][3]
        q_start = fwd_blocks[-1][2]

    lines = [
        f"chain {score} {t_name} {t_size} + {t_start} {t_end} "
        f"{q_name} {q_size} {q_strand} {q_start} {q_end} {chain_id}"
    ]
    n = len(fwd_blocks)
    for i, (ts, te, qs, qe) in enumerate(fwd_blocks):
        size = te - ts
        if i == n - 1:
            lines.append(str(size))
            continue
        nts, _nte, nqs, nqe = fwd_blocks[i + 1]
        dt = nts - te
        if q_strand == "+":
            # Walk moves up: gap after the current block is nqs - qe.
            dq = nqs - qe
        else:
            # Walk moves down: gap is qs - nqe.
            dq = qs - nqe
        lines.append(f"{size} {dt} {dq}")
    return "\n".join(lines) + "\n"


def identical_forward_chain(blocks):
    """A simple plus-strand chain mapping t bases to identical q bases."""
    return [(s, e, s, e) for (s, e) in blocks]


# ---------------------------------------------------------------------------
# Base-by-base ground truth
# ---------------------------------------------------------------------------
def truth_for_interval(chains, chrom, start, end):
    """Per-chain truth for one interval.

    Returns {chain_id: dict(fraction=..., fragments=[(ss, se, qs, qe, block)])}
    computed base by base.
    """
    truth = {}
    for chain in chains.chains:
        if chain.t_name != chrom:
            continue
        covered = set()
        fragments = []
        for block_idx, (ts, te, qs, qe) in enumerate(chain.blocks):
            ss, se = max(ts, start), min(te, end)
            if ss >= se:
                continue
            for base in range(ss, se):
                covered.add(base)
            off = ss - ts
            span = se - ss
            if chain.q_strand == "+":
                q0, q1 = qs + off, qs + off + span
            else:
                q0, q1 = qe - off - span, qe - off
            fragments.append((ss, se, q0, q1, block_idx))
        if fragments:
            truth[chain.chain_id] = {
                "fraction": len(covered) / (end - start),
                "fragments": fragments,
                "q_name": chain.q_name,
                "q_strand": chain.q_strand,
            }
    return truth


def assert_frame_matches_truth(mapped, unmapped, df, chains, min_match, multiple):
    """Compare a full liftover result against base-by-base truth."""
    expected_mapped_rows = []
    expected_unmapped = {}
    for row, rec in df.reset_index(drop=True).iterrows():
        truth = truth_for_interval(chains, rec.chrom, rec.start, rec.end)
        qualified = {
            cid: t for cid, t in truth.items() if t["fraction"] >= min_match
        }
        best = max((t["fraction"] for t in truth.values()), default=0.0)
        if not truth:
            expected_unmapped[row] = ("no_overlap", 0.0, 0)
            continue
        if not qualified:
            expected_unmapped[row] = ("low_coverage", best, 0)
            continue
        if multiple == "unique" and len(qualified) > 1:
            expected_unmapped[row] = ("ambiguous", best, len(qualified))
            continue
        for cid in sorted(qualified):
            t = qualified[cid]
            chain = chains[cid]
            for ss, se, q0, q1, block_idx in t["fragments"]:
                expected_mapped_rows.append(
                    (
                        row,
                        cid,
                        block_idx,
                        ss,
                        se,
                        chain.q_name,
                        q0,
                        q1,
                        chain.q_strand,
                        pytest.approx(t["fraction"]),
                    )
                )

    got_mapped = [tuple(r) for r in mapped[MAPPED_COLS].itertuples(index=False)]
    # Expected rows ordered like the required output ordering.
    expected_mapped_rows.sort(key=lambda r: (r[0], r[1], r[3]))
    assert got_mapped == expected_mapped_rows

    got_unmapped = {
        r.input_row: (r.reason, r.best_fraction, r.candidate_count)
        for r in unmapped.itertuples(index=False)
    }
    assert set(got_unmapped) == set(expected_unmapped)
    for row, (reason, best, count) in expected_unmapped.items():
        assert got_unmapped[row][0] == reason
        assert got_unmapped[row][1] == pytest.approx(best)
        assert got_unmapped[row][2] == count


def check_result_shapes(mapped, unmapped):
    assert list(mapped.columns) == MAPPED_COLS
    assert list(unmapped.columns) == UNMAPPED_COLS
    assert mapped.index.equals(pd.RangeIndex(len(mapped)))
    assert unmapped.index.equals(pd.RangeIndex(len(unmapped)))
    assert list(mapped.dtypes) == [
        np.int64,
        np.int64,
        np.int64,
        np.int64,
        np.int64,
        object,
        np.int64,
        np.int64,
        object,
        np.float64,
    ]
    assert list(unmapped.dtypes) == [
        np.int64,
        object,
        np.float64,
        np.int64,
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_read_chain_basic_plus_and_minus():
    plus = build_chain(
        1,
        "chr1",
        100,
        "chrA",
        120,
        [(0, 10, 10, 20), (15, 30, 25, 40), (40, 60, 50, 70)],
        "+",
        t_pad_left=0,
        q_pad_left=0,
    )
    chains = bf.read_chain(io.StringIO(plus))
    assert isinstance(chains, bf.ChainMap)
    assert len(chains) == 1
    assert chains.chain_ids == (1,)
    assert 1 in chains
    assert chains[1].blocks == (
        (0, 10, 10, 20),
        (15, 30, 25, 40),
        (40, 60, 50, 70),
    )
    assert chains.target_chromsizes == {"chr1": 100}
    assert chains.query_chromsizes == {"chrA": 120}

    minus = build_chain(
        2,
        "chr2",
        80,
        "chrB",
        90,
        [(10, 18, 52, 60), (22, 34, 36, 48), (40, 50, 20, 30)],
        "-",
    )
    cm = bf.read_chain(io.StringIO(minus))
    assert cm[2].q_strand == "-"
    assert cm[2].blocks == (
        (10, 18, 52, 60),
        (22, 34, 36, 48),
        (40, 50, 20, 30),
    )


def test_read_chain_empty_blank_separated_no_trailing_newline():
    assert len(bf.read_chain(io.StringIO(""))) == 0
    text = (
        "\n\n"
        + build_chain(1, "c1", 50, "x1", 50, [(0, 10, 0, 10)], "+")
        + "\n"
        + build_chain(2, "c2", 50, "x2", 50, [(0, 5, 0, 5)], "+").rstrip("\n")
    )
    chains = bf.read_chain(io.StringIO(text))
    assert chains.chain_ids == (1, 2)
    # No trailing blank line after the last chain is legal.
    no_trailing = build_chain(
        3, "c3", 50, "x3", 50, [(0, 5, 0, 5)], "+"
    ).rstrip("\n")
    assert bf.read_chain(io.StringIO(no_trailing)).chain_ids == (3,)


def test_read_chain_gzip_and_stream_ownership(tmp_path):
    text = build_chain(7, "c1", 50, "x1", 50, [(0, 10, 0, 10)], "+")
    plain = tmp_path / "map.chain"
    plain.write_text(text)
    gz = tmp_path / "map.chain.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(text)

    assert bf.read_chain(plain).chain_ids == (7,)
    assert bf.read_chain(gz).chain_ids == (7,)

    stream = io.StringIO(text)
    chains = bf.read_chain(stream)
    assert chains.chain_ids == (7,)
    assert not stream.closed  # borrowed streams stay open
    stream.read()  # still usable


def test_read_chain_io_error_propagates(tmp_path):
    with pytest.raises(OSError):
        bf.read_chain(tmp_path / "does_not_exist.chain")


@pytest.mark.parametrize(
    "bad_text, fragment",
    [
        # header with the wrong number of fields
        ("chain 1 c1 50 + 0 10 x1 50 + 0 10\n10\n", "header"),
        # non-integer field
        ("chain score c1 50 + 0 10 x1 50 + 0 10 1\n10\n", "integer"),
        # negative id
        ("chain 1 c1 50 + 0 10 x1 50 + 0 10 -1\n10\n", "non-negative"),
        # unsupported tStrand
        ("chain 1 c1 50 - 0 10 x1 50 + 0 10 1\n10\n", "tStrand"),
        # bad qStrand
        ("chain 1 c1 50 + 0 10 x1 50 ? 0 10 1\n10\n", "qStrand"),
        # empty header span
        ("chain 1 c1 50 + 10 10 x1 50 + 0 10 1\n", "non-empty"),
        # span beyond chromosome length
        ("chain 1 c1 5 + 0 10 x1 50 + 0 10 1\n10\n", "tSize"),
        # block size must be positive
        ("chain 1 c1 50 + 0 10 x1 50 + 0 10 1\n0\n", "positive"),
        # negative gap
        ("chain 1 c1 50 + 0 15 x1 50 + 0 15 1\n5 -1 5\n10\n", "non-negative"),
        # data line with wrong field count
        ("chain 1 c1 50 + 0 10 x1 50 + 0 10 1\n10 5\n", "1 field"),
        # truncated: chain without terminal block
        ("chain 1 c1 50 + 0 15 x1 50 + 0 15 1\n5 5 5\n", "truncated"),
        # blank line inside a chain
        ("chain 1 c1 50 + 0 15 x1 50 + 0 15 1\n5 5 5\n\n10\n", "blank line"),
        # walk ends before the declared t endpoint: two valid blocks whose
        # union stops one base short of tEnd
        (
            "chain 1 c1 50 + 0 16 x1 50 + 0 16 1\n5 5 5\n5\n",
            "target walk",
        ),
        # block walks outside the header query span
        ("chain 1 c1 50 + 0 10 x1 50 + 45 50 1\n10\n", "query block"),
    ],
)
def test_read_chain_malformed(bad_text, fragment):
    with pytest.raises(ValueError) as excinfo:
        bf.read_chain(io.StringIO(bad_text))
    message = str(excinfo.value)
    assert "line" in message
    assert fragment in message


def test_read_chain_errors_carry_line_and_id():
    text = (
        build_chain(1, "c1", 50, "x1", 50, [(0, 5, 0, 5)], "+")
        + "chain 2 c2 50 + 0 10 x2 50 + 0 10 2\n5 1 1\n3\n"
    )
    # chain 2 walks 5 + 1 + 3 = 9 on the target, short of tEnd 10
    with pytest.raises(ValueError) as excinfo:
        bf.read_chain(io.StringIO(text))
    message = str(excinfo.value)
    assert "line 5" in message
    assert "chain id 2" in message


def test_read_chain_duplicate_id_and_inconsistent_sizes():
    a = build_chain(1, "c1", 50, "x1", 50, [(0, 5, 0, 5)], "+")
    b = build_chain(1, "c2", 50, "x2", 50, [(0, 5, 0, 5)], "+")
    with pytest.raises(ValueError, match="duplicate chain id"):
        bf.read_chain(io.StringIO(a + b))

    c = build_chain(2, "c1", 60, "x3", 50, [(0, 5, 0, 5)], "+")
    with pytest.raises(ValueError, match="inconsistent target chromosome"):
        bf.read_chain(io.StringIO(a + c))

    d = build_chain(3, "c9", 50, "x1", 70, [(0, 5, 0, 5)], "+")
    with pytest.raises(ValueError, match="inconsistent query chromosome"):
        bf.read_chain(io.StringIO(a + d))


def test_malformed_file_never_returns_partial_map():
    good = build_chain(1, "c1", 50, "x1", 50, [(0, 5, 0, 5)], "+")
    # The second chain repeats id 1 and is therefore invalid.
    bad = "chain 100 c2 50 + 0 5 x2 50 + 0 5 1\n5\n"
    result = None
    with pytest.raises(ValueError):
        result = bf.read_chain(io.StringIO(good + bad))
    assert result is None
    # The same good prefix on its own is still parseable.
    assert bf.read_chain(io.StringIO(good)).chain_ids == (1,)


def test_read_chain_bad_source_type():
    with pytest.raises(ValueError):
        bf.read_chain(12345)


# ---------------------------------------------------------------------------
# Liftover: hand-built scenarios
# ---------------------------------------------------------------------------
@pytest.fixture
def plus_chains():
    text = build_chain(
        1,
        "chr1",
        100,
        "chrA",
        120,
        [(0, 10, 10, 20), (15, 30, 25, 40), (40, 60, 50, 70)],
        "+",
    )
    return bf.read_chain(io.StringIO(text))


@pytest.fixture
def minus_chains():
    text = build_chain(
        2,
        "chr2",
        80,
        "chrB",
        90,
        [(10, 18, 52, 60), (22, 34, 36, 48), (40, 50, 20, 30)],
        "-",
    )
    return bf.read_chain(io.StringIO(text))


def test_liftover_plus_fragments_and_gaps(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "start": [0, 5, 12, 40],
            "end": [5, 20, 30, 50],
        }
    )
    mapped, unmapped = bf.liftover(df, plus_chains, min_match=0.5)
    check_result_shapes(mapped, unmapped)
    expected = [
        (0, 1, 0, 0, 5, "chrA", 10, 15, "+", 1.0),
        (1, 1, 0, 5, 10, "chrA", 15, 20, "+", 2 / 3),
        (1, 1, 1, 15, 20, "chrA", 25, 30, "+", 2 / 3),
        (2, 1, 1, 15, 30, "chrA", 25, 40, "+", 15 / 18),
        (3, 1, 2, 40, 50, "chrA", 50, 60, "+", 1.0),
    ]
    assert [tuple(r) for r in mapped.itertuples(index=False)] == expected
    assert unmapped.empty


def test_liftover_minus_forward_coordinates(minus_chains):
    df = pd.DataFrame(
        {"chrom": ["chr2"] * 2, "start": [10, 30], "end": [20, 45]}
    )
    mapped, unmapped = bf.liftover(df, minus_chains, min_match=0.5)
    # Row 0: block 0 covers 8 of 10 bases; row 1 crosses blocks 1 and 2.
    rows = [tuple(r) for r in mapped.itertuples(index=False)]
    assert rows == [
        (0, 2, 0, 10, 18, "chrB", 52, 60, "-", 0.8),
        (1, 2, 1, 30, 34, "chrB", 36, 40, "-", 0.6),
        (1, 2, 2, 40, 45, "chrB", 25, 30, "-", 0.6),
    ]
    assert all(r.start < r.end for r in mapped.itertuples(index=False))
    assert unmapped.empty


def test_liftover_unmapped_reasons(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chrZZ"],
            "start": [60, 10, 0, 0],
            "end": [90, 15, 12, 10],
        }
    )
    mapped, unmapped = bf.liftover(df, plus_chains, min_match=0.95)
    assert mapped.empty
    reasons = dict(
        zip(unmapped.input_row, zip(unmapped.reason, unmapped.best_fraction))
    )
    assert reasons[0] == ("no_overlap", 0.0)  # beyond the chain span
    # [10, 15): 5 bases gap, no block intersects
    assert reasons[1] == ("no_overlap", 0.0)
    # [0, 12): 10 covered of 12 -> 0.833 < 0.95
    assert reasons[2] == ("low_coverage", pytest.approx(10 / 12))
    assert reasons[3] == ("no_overlap", 0.0)  # unknown chromosome
    assert (unmapped.candidate_count == 0).all()


def test_liftover_threshold_exact_equality(plus_chains):
    # [0, 12): block 0 covers exactly 10 of 12 bases -> fraction 5/6
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [12]})
    mapped, unmapped = bf.liftover(df, plus_chains, min_match=5 / 6)
    assert len(mapped) == 1 and unmapped.empty
    mapped, unmapped = bf.liftover(df, plus_chains, min_match=5 / 6 + 1e-12)
    assert mapped.empty
    assert unmapped.reason.iloc[0] == "low_coverage"
    # A fragment crossing a gap stays split: [0, 30) covers 25 of 30 bases
    # across two blocks, so two rows share fraction 25/30.
    mapped, unmapped = bf.liftover(
        pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [30]}),
        plus_chains,
        min_match=0.5,
    )
    assert len(mapped) == 2 and unmapped.empty
    assert all(v == pytest.approx(25 / 30) for v in mapped.match_fraction)
    # Exactly-at-threshold check with an easy fraction: [0,10) fully inside
    # block 0, so fraction 1.0.
    full = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})
    mapped, unmapped = bf.liftover(full, plus_chains, min_match=1.0)
    assert len(mapped) == 1 and unmapped.empty
    # Just above 1.0 must be rejected as an invalid parameter.
    with pytest.raises(ValueError):
        bf.liftover(full, plus_chains, min_match=1.0 + 1e-12)


def test_liftover_ambiguous_unique_vs_all():
    a = build_chain(
        1, "c1", 100, "q1", 100, identical_forward_chain([(0, 50)]), "+"
    )
    b = build_chain(
        2, "c1", 100, "q2", 100, identical_forward_chain([(0, 50)]), "+"
    )
    chains = bf.read_chain(io.StringIO(a + b))
    df = pd.DataFrame({"chrom": ["c1"], "start": [0], "end": [50]})

    mapped, unmapped = bf.liftover(df, chains, multiple="unique")
    assert mapped.empty
    assert unmapped.reason.iloc[0] == "ambiguous"
    assert unmapped.candidate_count.iloc[0] == 2
    assert unmapped.best_fraction.iloc[0] == 1.0

    mapped, unmapped = bf.liftover(df, chains, multiple="all")
    assert unmapped.empty
    assert set(mapped.chain_id) == {1, 2}
    assert len(mapped) == 2


def test_liftover_same_coordinates_different_ids_remain_candidates():
    # Two chains mapping to the identical coordinates but with distinct ids:
    # scores must not be used to pick a winner.
    a = build_chain(
        1, "c1", 100, "q1", 100, identical_forward_chain([(0, 50)]), "+",
        score=1,
    )
    b = build_chain(
        9, "c1", 100, "q1", 100, identical_forward_chain([(0, 50)]), "+",
        score=999,
    )
    chains = bf.read_chain(io.StringIO(a + b))
    df = pd.DataFrame({"chrom": ["c1"], "start": [0], "end": [50]})
    mapped, unmapped = bf.liftover(df, chains, multiple="unique")
    assert unmapped.reason.iloc[0] == "ambiguous"
    mapped, _ = bf.liftover(df, chains, multiple="all")
    # Numeric (not lexicographic) chain id ordering in the output.
    assert list(mapped.chain_id) == [1, 9]


def test_liftover_chains_are_not_combined(plus_chains):
    # A second chain covers the gap [10, 15) only.  Neither chain alone
    # reaches 0.95 for [0, 20); their fractions must not be summed.
    extra = build_chain(
        2, "chr1", 100, "chrZ", 100, [(10, 15, 0, 5)], "+"
    )
    text = "\n".join(
        [
            build_chain(
                1,
                "chr1",
                100,
                "chrA",
                120,
                [(0, 10, 10, 20), (15, 30, 25, 40), (40, 60, 50, 70)],
                "+",
            ).rstrip(),
            extra.rstrip(),
            "",
        ]
    )
    chains = bf.read_chain(io.StringIO(text))
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [20]})
    mapped, unmapped = bf.liftover(df, chains, min_match=0.95)
    assert mapped.empty
    row = unmapped.iloc[0]
    assert row.reason == "low_coverage"
    assert row.best_fraction == pytest.approx(0.75)  # 15/20, the better chain
    assert row.candidate_count == 0


def test_liftover_duplicate_inputs_and_index(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 0, 0],
            "end": [5, 5, 5],
            "payload": ["a", "b", "c"],
        },
        index=[7, 7, 2],  # duplicated, non-sequential index is allowed
    )
    mapped, unmapped = bf.liftover(df, plus_chains)
    assert unmapped.empty
    assert list(mapped.input_row) == [0, 1, 2]
    assert list(mapped.index) == [0, 1, 2]


def test_liftover_custom_columns():
    text = build_chain(1, "c1", 50, "q1", 50, [(0, 10, 0, 10)], "+")
    chains = bf.read_chain(io.StringIO(text))
    df = pd.DataFrame(
        {"CHR": ["c1"], "FROM": [2], "TO": [8], "note": ["x"]}
    )
    mapped, unmapped = bf.liftover(
        df, chains, cols=("CHR", "FROM", "TO")
    )
    assert unmapped.empty
    assert list(mapped.source_start) == [2]
    assert list(mapped.end) == [8]


def test_liftover_strand_column(minus_chains, plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr2", "chr2", "chr2", "chr2"],
            "start": [10, 10, 10, 30],
            "end": [18, 18, 18, 40],
            "st": ["+", "-", ".", "-"],
        }
    )
    mapped, _ = bf.liftover(df, minus_chains, strand="st", min_match=0.5)
    by_row = mapped.groupby("input_row").strand.unique()
    assert list(by_row[0]) == ["-"]  # plus input flipped by minus chain
    assert list(by_row[1]) == ["+"]  # minus input flipped to plus
    assert list(by_row[2]) == ["."]  # dot stays dot

    # Without a strand column the output strand is the chain qStrand.
    mapped2, _ = bf.liftover(df.drop(columns="st"), minus_chains, min_match=0.5)
    assert (mapped2.strand == "-").all()


def test_liftover_empty_frame(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": pd.Series([], dtype=object),
            "start": pd.Series([], dtype=np.int64),
            "end": pd.Series([], dtype=np.int64),
        }
    )
    mapped, unmapped = bf.liftover(df, plus_chains)
    check_result_shapes(mapped, unmapped)
    assert mapped.empty and unmapped.empty

    # Even an empty frame must carry valid dtypes.
    bad = pd.DataFrame(
        {
            "chrom": pd.Series([1], dtype=np.int64).iloc[:0],
            "start": pd.Series([], dtype=np.float64),
            "end": pd.Series([], dtype=np.float64),
        }
    )
    with pytest.raises(ValueError):
        bf.liftover(bad, plus_chains)

    # Empty ChainMap: everything is no_overlap.
    empty_map = bf.read_chain(io.StringIO(""))
    df2 = pd.DataFrame({"chrom": ["c"], "start": [0], "end": [3]})
    mapped, unmapped = bf.liftover(df2, empty_map)
    assert mapped.empty
    assert unmapped.reason.iloc[0] == "no_overlap"


def test_liftover_dtypes_and_alternative_pandas_types():
    text = build_chain(1, "c1", 100, "q1", 100, [(0, 50, 0, 50)], "+")
    chains = bf.read_chain(io.StringIO(text))
    base = {"start": [0, 10], "end": [10, 20]}

    for chrom_kwargs in (
        {"dtype": object},
        {"dtype": "string"},
        {"dtype": pd.CategoricalDtype(["c1", "c2"])},
    ):
        df = pd.DataFrame(
            {
                "chrom": pd.Series(["c1", "c1"], **chrom_kwargs),
                "start": pd.array(base["start"], dtype="Int64"),
                "end": pd.array(base["end"], dtype="Int64"),
            }
        )
        mapped, unmapped = bf.liftover(df, chains)
        check_result_shapes(mapped, unmapped)
        assert len(mapped) == 2 and unmapped.empty


def test_liftover_big_integers():
    big = 2**63 - 1
    text = build_chain(
        1, "c1", big, "q1", big, [(0, big, 0, big)], "+"
    )
    chains = bf.read_chain(io.StringIO(text))
    df = pd.DataFrame(
        {"chrom": ["c1"], "start": [big - 2], "end": [big]}
    )
    mapped, unmapped = bf.liftover(df, chains)
    assert unmapped.empty
    assert list(mapped.start) == [big - 2]
    assert list(mapped.end) == [big]
    assert mapped.match_fraction.iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_arguments(plus_chains):
    good = pd.DataFrame(
        {"chrom": ["chr1"], "start": [0], "end": [5]}
    )
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, min_match=0.0)
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, min_match=1.5)
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, min_match=float("nan"))
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, min_match=float("inf"))
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, min_match=True)  # bool rejected
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, multiple="first")
    with pytest.raises(ValueError):
        bf.liftover(good, "not a chainmap")
    with pytest.raises(ValueError):
        bf.liftover("not a dataframe", plus_chains)
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, cols=("chrom", "chrom", "end"))
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, cols=("chrom", "start", "nope"))
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, strand="start")  # overlaps coords
    with pytest.raises(ValueError):
        bf.liftover(good, plus_chains, strand="missing")


def test_invalid_data_messages_carry_physical_rows(plus_chains):
    def expect_row(df, match, **kwargs):
        with pytest.raises(ValueError, match=match):
            bf.liftover(df, plus_chains, **kwargs)

    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1", ""], "start": [0, 0], "end": [5, 5]}
        ),
        "row 1",
    )
    expect_row(
        pd.DataFrame(
            {
                "chrom": ["chr1", None],
                "start": [0, 0],
                "end": [5, 5],
            }
        ),
        "row 1",
    )
    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1", "chr1"], "start": [0, 5], "end": [5, 5]}
        ),
        "row 1",
    )
    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1", "chr1"], "start": [-1, 0], "end": [5, 5]}
        ),
        "row 0",
    )
    expect_row(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 0],
                "end": [5, 2**63],
            }
        ),
        "row 1",
    )
    # Interval beyond the declared target chromosome length.
    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1", "chr1"], "start": [0, 95], "end": [5, 101]}
        ),
        "row 1",
    )
    # Nullable integer with missing data.
    expect_row(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": pd.array([0, None], dtype="Int64"),
                "end": pd.array([5, 5], dtype="Int64"),
            }
        ),
        "row 1",
    )
    # Boolean coordinate column rejected.
    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1"], "start": [True], "end": [False]}
        ),
        "integer",
    )
    # Float coordinates rejected.
    expect_row(
        pd.DataFrame(
            {"chrom": ["chr1"], "start": [0.5], "end": [1.5]}
        ),
        "integer",
    )
    # Bad chrom dtype.
    expect_row(
        pd.DataFrame({"chrom": [1], "start": [0], "end": [5]}),
        "dtype",
    )
    # Duplicated column names.
    expect_row(
        pd.DataFrame(
            [["chr1", 0, 5, 1]],
            columns=["chrom", "start", "end", "end"],
        ),
        "unique",
    )
    # Bad strand values.
    expect_row(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 0],
                "end": [5, 5],
                "st": ["+", "?"],
            }
        ),
        "row 1",
        strand="st",
    )


def test_invalid_strand_missing_value(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [5, 5],
            "st": ["+", None],
        }
    )
    with pytest.raises(ValueError, match="row 1"):
        bf.liftover(df, plus_chains, strand="st")


# ---------------------------------------------------------------------------
# Purity, reuse and ordering invariance
# ---------------------------------------------------------------------------
def test_inputs_not_modified_and_reuse(plus_chains):
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chrZZ"],
            "start": [0, 12, 0],
            "end": [5, 30, 10],
        }
    )
    snapshot = df.copy(deep=True)
    m1, u1 = bf.liftover(df, plus_chains, min_match=0.8)
    # A failing call must not change subsequent results.
    with pytest.raises(ValueError):
        bf.liftover(
            pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [0]}),
            plus_chains,
        )
    m2, u2 = bf.liftover(df, plus_chains, min_match=0.8)
    pd.testing.assert_frame_equal(df, snapshot)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(u1, u2)
    # Interleaved repeated calls stay consistent.  The reversed frame is a
    # different input, so its physical rows number the reversed records:
    # the original rows were [0: chr1 [0,5)], [1: chr1 [12,30)],
    # [2: chrZZ unknown]; after reversal the unknown record is physical 0
    # and the two mappable ones become physical rows 1 and 2.
    df_rev = df.iloc[::-1].reset_index(drop=True)
    m3, u3 = bf.liftover(df_rev, plus_chains, min_match=0.8)
    assert set(m3.input_row) == {1, 2}
    assert set(u3.input_row) == {0}
    # Internal arrays are read-only.
    for arrays in plus_chains._block_index.values():
        for arr in arrays:
            assert not arr.flags.writeable


def test_chain_order_does_not_matter():
    blocks_a = [(0, 10, 0, 10), (20, 30, 40, 50)]
    blocks_b = [(5, 25, 100, 120)]
    t1 = build_chain(1, "c1", 100, "q1", 200, blocks_a, "+")
    t2 = build_chain(2, "c1", 100, "q2", 200, blocks_b, "+")
    chains_ab = bf.read_chain(io.StringIO(t1 + t2))
    chains_ba = bf.read_chain(io.StringIO(t2 + t1))
    # The stored records are the same set, regardless of file order.
    assert sorted(chains_ab.chains) == sorted(chains_ba.chains)
    assert chains_ab.chain_ids == chains_ba.chain_ids
    assert chains_ab.target_chromsizes == chains_ba.target_chromsizes
    assert chains_ab.query_chromsizes == chains_ba.query_chromsizes

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "chrom": ["c1"] * 30,
            "start": rng.integers(0, 40, 30),
            "end": 0,
        }
    )
    df["end"] = df["start"] + rng.integers(1, 20, 30)
    m1, u1 = bf.liftover(df, chains_ab, multiple="all", min_match=0.1)
    m2, u2 = bf.liftover(df, chains_ba, multiple="all", min_match=0.1)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(u1, u2)


def test_zero_gap_block_split_invariance():
    # One block of 20 bases vs. the same block split in two with dt=dq=0.
    one = build_chain(1, "c1", 100, "q1", 100, [(0, 20, 0, 20)], "+")
    split_text = (
        "chain 100 c1 100 + 0 20 q1 100 + 0 20 1\n"
        "8 0 0\n"
        "12\n"
    )
    chains_one = bf.read_chain(io.StringIO(one))
    chains_split = bf.read_chain(io.StringIO(split_text))

    df = pd.DataFrame(
        {"chrom": ["c1"] * 3, "start": [0, 5, 8], "end": [20, 12, 18]}
    )
    m1, u1 = bf.liftover(df, chains_one, min_match=0.1)
    m2, u2 = bf.liftover(df, chains_split, min_match=0.1)

    # Coverage fractions per input must be identical (one fraction per
    # qualifying chain per input).
    frac1 = m1.groupby("input_row").match_fraction.unique()
    frac2 = m2.groupby("input_row").match_fraction.unique()
    for row in frac1.index:
        assert list(frac1[row]) == list(frac2[row])

    # Base-level mapped coordinate spans must be identical after merging the
    # split fragments back together.
    merged2 = (
        m2.groupby("input_row")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    merged1 = (
        m1.groupby("input_row")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    pd.testing.assert_frame_equal(merged1, merged2)
    # Splitting may only add fragment rows, never remove coverage.
    assert len(m2) >= len(m1)
    pd.testing.assert_frame_equal(u1, u2)
    # The split version reports two block numbers.
    assert set(m2.block) == {0, 1}


def test_zero_gap_block_split_minus_strand():
    # Same minus-strand chain as one block vs. two zero-gap blocks.
    one_text = (
        "chain 1 c1 100 + 0 20 q1 100 - 30 50 1\n20\n"
    )
    split_text = (
        "chain 1 c1 100 + 0 20 q1 100 - 30 50 1\n"
        "8 0 0\n12\n"
    )
    chains_one = bf.read_chain(io.StringIO(one_text))
    chains_split = bf.read_chain(io.StringIO(split_text))
    df = pd.DataFrame(
        {"chrom": ["c1"] * 2, "start": [0, 7], "end": [20, 15]}
    )
    m1, u1 = bf.liftover(df, chains_one, min_match=0.1)
    m2, u2 = bf.liftover(df, chains_split, min_match=0.1)
    merged1 = (
        m1.groupby("input_row")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    merged2 = (
        m2.groupby("input_row")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    pd.testing.assert_frame_equal(merged1, merged2)
    pd.testing.assert_frame_equal(u1, u2)
    assert len(m2) > len(m1)  # the split creates extra fragment rows
    assert (m2.strand == "-").all()


# ---------------------------------------------------------------------------
# Randomized base-by-base differential tests
# ---------------------------------------------------------------------------
def _random_chain_text(rng, chain_id, t_size, q_size, strand):
    """Generate a valid random chain by walking both sides in chain order."""
    n_blocks = int(rng.integers(1, 6))
    t_pos = int(rng.integers(0, 4))
    if strand == "+":
        q_pos = int(rng.integers(0, 4))
    else:
        q_pos = q_size - int(rng.integers(0, 4))  # walk starts near q_end
    walk_blocks = []
    for _ in range(n_blocks):
        size = int(rng.integers(1, 8))
        t_s, t_e = t_pos, t_pos + size
        if t_e >= t_size - 1:
            break
        if strand == "+":
            q_s, q_e = q_pos, q_pos + size
            if q_e >= q_size - 1:
                break
        else:
            q_e, q_s = q_pos, q_pos - size
            if q_s <= 0:
                break
        walk_blocks.append((t_s, t_e, q_s, q_e))
        dt = int(rng.integers(0, 4))
        dq = int(rng.integers(0, 4))
        t_pos = t_e + dt
        if strand == "+":
            q_pos = q_e + dq
        else:
            q_pos = q_s - dq
    if not walk_blocks:
        walk_blocks = [(0, 1, 0, 1)]
    return build_chain(
        chain_id, "c1", t_size, f"q{chain_id}", q_size, walk_blocks, strand
    )


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("multiple", ["unique", "all"])
def test_randomized_matches_base_truth(strand, multiple):
    rng = np.random.default_rng(2024 + (strand == "-") + 2 * (multiple == "all"))
    t_size, q_size = 60, 60
    texts = [
        _random_chain_text(rng, cid, t_size, q_size, strand)
        for cid in range(1, 5)
    ]
    chains = bf.read_chain(io.StringIO("\n".join(texts)))

    intervals = []
    for _ in range(60):
        s = int(rng.integers(0, t_size - 1))
        e = int(rng.integers(s + 1, min(t_size, s + 25) + 1))
        intervals.append(("c1", s, e))
    # Include intervals on an unknown chromosome and fully outside spans.
    intervals += [("ghost", 0, 5), ("c1", t_size - 4, t_size)]
    df = pd.DataFrame(intervals, columns=["chrom", "start", "end"])

    for min_match in (0.25, 0.5, 0.75, 1.0):
        mapped, unmapped = bf.liftover(
            df, chains, min_match=min_match, multiple=multiple
        )
        check_result_shapes(mapped, unmapped)
        assert_frame_matches_truth(
            mapped, unmapped, df, chains, min_match, multiple
        )
        # Failed rows leave no fragments; successful rows are not unmapped.
        assert set(mapped.input_row).isdisjoint(set(unmapped.input_row))


def test_randomized_mixed_strands_and_input_strand():
    rng = np.random.default_rng(77)
    texts = [
        _random_chain_text(rng, 1, 60, 60, "+"),
        _random_chain_text(rng, 2, 60, 60, "-"),
    ]
    chains = bf.read_chain(io.StringIO("\n".join(texts)))
    df = pd.DataFrame(
        {
            "chrom": ["c1"] * 40,
            "start": rng.integers(0, 45, 40),
            "end": 0,
            "st": rng.choice(["+", "-", "."], 40),
        }
    )
    df["end"] = df["start"] + rng.integers(1, 12, 40)
    mapped, unmapped = bf.liftover(
        df, chains, strand="st", multiple="all", min_match=0.2
    )
    check_result_shapes(mapped, unmapped)

    # Independent strand transform check.
    for row in mapped.itertuples(index=False):
        in_st = df.st.iloc[row.input_row]
        q_strand = chains[row.chain_id].q_strand
        if in_st == ".":
            assert row.strand == "."
        elif q_strand == "+":
            assert row.strand == in_st
        else:
            assert row.strand == ("-" if in_st == "+" else "+")
