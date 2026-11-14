"""
Coordinate liftover between genome assemblies via UCSC ``chain`` files.

This module implements reading of UCSC chain files
(see https://genome.ucsc.edu/goldenPath/help/chain.html) and remapping of
zero-based, left-closed/right-open genomic intervals from the *target* (``t``)
coordinate system of the chains to the *query* (``q``) coordinate system.

Only the fixed ``t -> q`` direction is supported: chain headers must carry
``tStrand +`` while ``qStrand`` may be ``+`` or ``-``.  All input and output
coordinates are given on the forward strand, including fragments that originate
from minus-strand chains.

The two public entry points are:

* :func:`read_chain` -- parse a chain file into a reusable, immutable
  :class:`ChainMap`.
* :func:`liftover` -- remap a bedframe of intervals against a
  :class:`ChainMap`, returning the mapped fragments and the unmapped records.

The implementation works on aligned blocks (no base-level expansion), uses
sorted block arrays with bisection lookups (no rows-by-blocks dense matrix and
no full scan of all blocks per query) and never mutates its inputs.
"""

import gzip
import os
from collections import namedtuple
from contextlib import ExitStack

import numpy as np
import pandas as pd

__all__ = [
    "Chain",
    "ChainMap",
    "liftover",
    "read_chain",
]

# Maximum value allowed for any coordinate (see the task specification).
_COORD_MAX = 2**63 - 1

# Columns of the two returned data frames, in their fixed order.
_MAPPED_COLUMNS = [
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
_UNMAPPED_COLUMNS = ["input_row", "reason", "best_fraction", "candidate_count"]


# ---------------------------------------------------------------------------
# Chain container
# ---------------------------------------------------------------------------
class Chain(
    namedtuple(
        "Chain",
        [
            "chain_id",
            "score",
            "t_name",
            "t_size",
            "t_strand",
            "t_start",
            "t_end",
            "q_name",
            "q_size",
            "q_strand",
            "q_start",
            "q_end",
            "blocks",
        ],
    )
):
    """A single chain record.

    Attributes
    ----------
    chain_id : int
        Non-negative chain ``id`` from the header; unique within a
        :class:`ChainMap`.
    score : int
        Non-negative chain ``score`` from the header.  Never used to break
        ties.
    t_name, q_name : str
        Target (source) and query (destination) chromosome names.
    t_size, q_size : int
        Declared chromosome lengths.
    t_strand : str
        Always ``"+"``.
    q_strand : str
        ``"+"`` or ``"-"``.
    t_start, t_end : int
        Zero-based, half-open target span covered by the chain.
    q_start, q_end : int
        Forward-strand, zero-based, half-open query span covered by the
        chain.  For a minus-strand chain the alignment walks from ``q_end``
        towards ``q_start``.
    blocks : tuple of (t_start, t_end, q_start, q_end) tuples
        Aligned, gap-free blocks in chain order.  Target and query
        coordinates are forward-strand, zero-based, half-open, including for
        minus-strand chains.
    """

    __slots__ = ()


class ChainMap:
    """An immutable, reusable collection of parsed UCSC chains.

    Instances are created with :func:`read_chain`.  Besides exposing the
    individual :class:`Chain` records, they hold sorted per-chromosome block
    indices that allow interval remapping without scanning every chain block.

    Notes
    -----
    A :class:`ChainMap` is immutable from the caller's perspective.  The
    internal arrays are read-only numpy arrays and :func:`liftover` never
    modifies the object, so the same instance can be shared across repeated
    and interleaved calls.
    """

    def __init__(self, chains):
        """Build a ChainMap from an iterable of validated :class:`Chain` objects."""
        chains = tuple(chains)
        seen_ids = set()
        t_sizes = {}
        q_sizes = {}
        for chain in chains:
            if chain.chain_id in seen_ids:
                # read_chain already guards against this; keep the container
                # self-consistent if constructed directly.
                raise ValueError(f"duplicate chain id: {chain.chain_id}")
            seen_ids.add(chain.chain_id)
            if chain.t_name in t_sizes and t_sizes[chain.t_name] != chain.t_size:
                raise ValueError(
                    f"inconsistent target chromosome length for {chain.t_name}: "
                    f"{t_sizes[chain.t_name]} != {chain.t_size}"
                )
            t_sizes[chain.t_name] = chain.t_size
            if chain.q_name in q_sizes and q_sizes[chain.q_name] != chain.q_size:
                raise ValueError(
                    f"inconsistent query chromosome length for {chain.q_name}: "
                    f"{q_sizes[chain.q_name]} != {chain.q_size}"
                )
            q_sizes[chain.q_name] = chain.q_size

        self._chains = chains
        # id -> position in self._chains (chain ids are arbitrary non-negative
        # integers and must not be assumed dense or ordered).
        self._id_pos = {chain.chain_id: pos for pos, chain in enumerate(chains)}
        self._t_sizes = t_sizes
        self._q_sizes = q_sizes

        # Concatenate every block, grouped by target chromosome.  Blocks of a
        # single chain are disjoint on the target; blocks of different chains
        # may overlap, but the intervals can still be sorted together.
        per_chrom = {}
        for chain in chains:
            entry = per_chrom.setdefault(chain.t_name, [])
            for block_idx, (ts, te, qs, qe) in enumerate(chain.blocks):
                entry.append((ts, te, qs, qe, chain.chain_id, block_idx))

        block_index = {}
        for t_name, rows in per_chrom.items():
            dt = np.array([(r[0], r[1]) for r in rows], dtype=np.int64)
            dq = np.array([(r[2], r[3]) for r in rows], dtype=np.int64)
            order = np.lexsort((dt[:, 1], dt[:, 0]))
            dt = dt[order]
            dq = dq[order]
            chain_ids = np.array([r[4] for r in rows], dtype=np.int64)[order]
            block_idxs = np.array([r[5] for r in rows], dtype=np.int64)[order]
            # Running maximum of block ends supports the bisection-based
            # overlap lookup used by liftover().
            max_end = np.maximum.accumulate(dt[:, 1])
            arrays = (
                dt[:, 0],
                dt[:, 1],
                max_end,
                dq[:, 0],
                dq[:, 1],
                chain_ids,
                block_idxs,
            )
            for arr in arrays:
                arr.setflags(write=False)
            block_index[t_name] = arrays

        self._block_index = block_index

    # ------------------------------------------------------------------
    # Mapping-style read-only access
    # ------------------------------------------------------------------
    @property
    def chains(self):
        """Tuple of all :class:`Chain` records, in parse order."""
        return self._chains

    @property
    def chain_ids(self):
        """Sorted tuple of all chain ids."""
        return tuple(sorted(self._id_pos))

    @property
    def target_chromsizes(self):
        """Dict mapping target chromosome names to their declared lengths."""
        return dict(self._t_sizes)

    @property
    def query_chromsizes(self):
        """Dict mapping query chromosome names to their declared lengths."""
        return dict(self._q_sizes)

    def __len__(self):
        return len(self._chains)

    def __contains__(self, chain_id):
        return chain_id in self._id_pos

    def __iter__(self):
        return iter(chain.chain_id for chain in self._chains)

    def __getitem__(self, chain_id):
        return self._chains[self._id_pos[chain_id]]

    def get(self, chain_id, default=None):
        """Return the :class:`Chain` with the given id, or ``default``."""
        pos = self._id_pos.get(chain_id)
        return default if pos is None else self._chains[pos]

    def __eq__(self, other):
        if not isinstance(other, ChainMap):
            return NotImplemented
        return self._chains == other._chains

    def __repr__(self):
        return f"ChainMap(n_chains={len(self._chains)})"


# ---------------------------------------------------------------------------
# Chain file parsing
# ---------------------------------------------------------------------------
def _parse_int(token, line_no, chain_id, field):
    """Parse an integer field, raising ValueError with file context on failure."""
    try:
        return int(token)
    except (TypeError, ValueError):
        where = f"line {line_no}"
        if chain_id is not None:
            where += f", chain id {chain_id}"
        raise ValueError(
            f"invalid chain file at {where}: field {field!r} is not an "
            f"integer: {token!r}"
        ) from None


def _chain_error(message, line_no, chain_id=None):
    where = f"line {line_no}"
    if chain_id is not None:
        where += f", chain id {chain_id}"
    return ValueError(f"invalid chain file at {where}: {message}")


def _read_chain_stream(handle):
    """Parse chain records from an open text stream into a ChainMap."""
    chains = []
    seen_ids = set()
    t_sizes = {}
    q_sizes = {}

    # Parser state.
    header = None  # dict describing the chain currently being parsed
    blocks = None  # list of fully resolved block tuples
    t_pos = q_pos = 0
    line_no = 0

    def finalize(end_line_no):
        """Validate and materialize the chain that just ended."""
        chain_id = header["chain_id"]
        if not blocks:
            raise _chain_error(
                "chain has no alignment blocks", end_line_no, chain_id
            )
        if t_pos != header["t_end"]:
            raise _chain_error(
                f"target walk ends at {t_pos}, expected header tEnd "
                f"{header['t_end']}",
                end_line_no,
                chain_id,
            )
        # The plus-strand walk finishes at qEnd, the minus-strand walk at
        # qStart (both given as forward coordinates in the header).
        q_goal = (
            header["q_end"] if header["q_strand"] == "+" else header["q_start"]
        )
        if q_pos != q_goal:
            raise _chain_error(
                f"query walk ends at {q_pos}, expected header boundary {q_goal}",
                end_line_no,
                chain_id,
            )
        chains.append(
            Chain(
                chain_id=chain_id,
                score=header["score"],
                t_name=header["t_name"],
                t_size=header["t_size"],
                t_strand=header["t_strand"],
                t_start=header["t_start"],
                t_end=header["t_end"],
                q_name=header["q_name"],
                q_size=header["q_size"],
                q_strand=header["q_strand"],
                q_start=header["q_start"],
                q_end=header["q_end"],
                blocks=tuple(blocks),
            )
        )

    while True:
        raw_line = handle.readline()
        if raw_line == "":
            break
        line_no += 1
        line = raw_line.strip()

        if line == "":
            # Blank lines may separate chains; one inside a chain means the
            # chain was truncated before its terminal block.
            if header is not None:
                raise _chain_error(
                    "unexpected blank line; chain truncated before its "
                    "terminal block",
                    line_no,
                    header["chain_id"],
                )
            continue

        if header is None:
            # Expect a new chain header: "chain" + 12 fields.
            tokens = line.split()
            if tokens[0] != "chain" or len(tokens) != 13:
                raise _chain_error(
                    "expected a chain header ('chain' followed by 12 "
                    f"fields), got {len(tokens) - 1} field(s)",
                    line_no,
                )
            chain_id = _parse_int(tokens[12], line_no, None, "id")
            score = _parse_int(tokens[1], line_no, chain_id, "score")
            t_size = _parse_int(tokens[3], line_no, chain_id, "tSize")
            t_start = _parse_int(tokens[5], line_no, chain_id, "tStart")
            t_end = _parse_int(tokens[6], line_no, chain_id, "tEnd")
            q_size = _parse_int(tokens[8], line_no, chain_id, "qSize")
            q_start = _parse_int(tokens[10], line_no, chain_id, "qStart")
            q_end = _parse_int(tokens[11], line_no, chain_id, "qEnd")
            t_name, t_strand = tokens[2], tokens[4]
            q_name, q_strand = tokens[7], tokens[9]

            if score < 0:
                raise _chain_error(
                    "score must be non-negative", line_no, chain_id
                )
            if chain_id < 0:
                raise _chain_error(
                    "id must be non-negative", line_no, chain_id
                )
            if score > _COORD_MAX or chain_id > _COORD_MAX:
                raise _chain_error(
                    "score and id must not exceed 2**63 - 1",
                    line_no,
                    chain_id,
                )
            if chain_id in seen_ids:
                raise _chain_error(
                    f"duplicate chain id {chain_id}", line_no, chain_id
                )
            if t_strand != "+":
                raise _chain_error(
                    f"only tStrand '+' is supported, got {t_strand!r}",
                    line_no,
                    chain_id,
                )
            if q_strand not in ("+", "-"):
                raise _chain_error(
                    f"qStrand must be '+' or '-', got {q_strand!r}",
                    line_no,
                    chain_id,
                )
            if t_size <= 0 or q_size <= 0:
                raise _chain_error(
                    "chromosome sizes must be positive", line_no, chain_id
                )
            if t_size > _COORD_MAX or q_size > _COORD_MAX:
                raise _chain_error(
                    "chromosome sizes exceed 2**63-1", line_no, chain_id
                )
            if not (0 <= t_start < t_end <= t_size):
                raise _chain_error(
                    "target span must be non-empty and within tSize "
                    f"([{t_start}, {t_end}) in size {t_size})",
                    line_no,
                    chain_id,
                )
            if not (0 <= q_start < q_end <= q_size):
                raise _chain_error(
                    "query span must be non-empty and within qSize "
                    f"([{q_start}, {q_end}) in size {q_size})",
                    line_no,
                    chain_id,
                )
            if t_name in t_sizes and t_sizes[t_name] != t_size:
                raise _chain_error(
                    f"inconsistent target chromosome length for {t_name!r}: "
                    f"{t_sizes[t_name]} != {t_size}",
                    line_no,
                    chain_id,
                )
            if q_name in q_sizes and q_sizes[q_name] != q_size:
                raise _chain_error(
                    f"inconsistent query chromosome length for {q_name!r}: "
                    f"{q_sizes[q_name]} != {q_size}",
                    line_no,
                    chain_id,
                )

            seen_ids.add(chain_id)
            t_sizes[t_name] = t_size
            q_sizes[q_name] = q_size
            header = {
                "chain_id": chain_id,
                "score": score,
                "t_name": t_name,
                "t_size": t_size,
                "t_strand": t_strand,
                "t_start": t_start,
                "t_end": t_end,
                "q_name": q_name,
                "q_size": q_size,
                "q_strand": q_strand,
                "q_start": q_start,
                "q_end": q_end,
            }
            blocks = []
            t_pos = t_start
            # Plus-strand walks start at qStart and move up; minus-strand
            # walks start at qEnd and move down.  q_pos is the next boundary
            # to consume.
            q_pos = q_start if q_strand == "+" else q_end
            continue

        # Inside a chain: alignment data line(s).
        tokens = line.split()
        if len(tokens) not in (1, 3):
            raise _chain_error(
                "alignment data line must have 1 field (terminal block) or "
                f"3 fields, got {len(tokens)}",
                line_no,
                header["chain_id"],
            )

        size = _parse_int(tokens[0], line_no, header["chain_id"], "size")
        if size <= 0:
            raise _chain_error(
                "block size must be a positive integer",
                line_no,
                header["chain_id"],
            )
        dt = dq = 0
        if len(tokens) == 3:
            dt = _parse_int(tokens[1], line_no, header["chain_id"], "dt")
            dq = _parse_int(tokens[2], line_no, header["chain_id"], "dq")
            if dt < 0 or dq < 0:
                raise _chain_error(
                    "gap sizes dt and dq must be non-negative",
                    line_no,
                    header["chain_id"],
                )

        t_block_start = t_pos
        t_block_end = t_pos + size
        if header["q_strand"] == "+":
            q_block_start = q_pos
            q_block_end = q_pos + size
        else:
            q_block_end = q_pos
            q_block_start = q_pos - size

        # Every block must lie within the declared header spans.
        t_inside = (
            header["t_start"] <= t_block_start
            and t_block_end <= header["t_end"]
        )
        if not t_inside:
            raise _chain_error(
                f"target block [{t_block_start}, {t_block_end}) falls outside "
                f"header span [{header['t_start']}, {header['t_end']})",
                line_no,
                header["chain_id"],
            )
        q_inside = (
            header["q_start"] <= q_block_start
            and q_block_end <= header["q_end"]
        )
        if not q_inside:
            raise _chain_error(
                f"query block [{q_block_start}, {q_block_end}) falls outside "
                f"header span [{header['q_start']}, {header['q_end']})",
                line_no,
                header["chain_id"],
            )

        blocks.append((t_block_start, t_block_end, q_block_start, q_block_end))

        if len(tokens) == 1:
            # Terminal block: advance the walk to the block ends and
            # finalize.  Plus-strand walks finish at qEnd, minus-strand
            # walks at qStart.
            t_pos = t_block_end
            q_pos = q_block_end if header["q_strand"] == "+" else q_block_start
            finalize(line_no)
            header = None
            blocks = None
        else:
            t_pos = t_block_end + dt
            if header["q_strand"] == "+":
                q_pos = q_block_end + dq
            else:
                q_pos = q_block_start - dq
            # Gaps must stay inside the header spans; the next block also has
            # to fit.
            if not (header["t_start"] <= t_pos < header["t_end"]):
                raise _chain_error(
                    f"target gap walks to {t_pos}, outside header span "
                    f"[{header['t_start']}, {header['t_end']})",
                    line_no,
                    header["chain_id"],
                )
            if not (header["q_start"] <= q_pos < header["q_end"]):
                raise _chain_error(
                    f"query gap walks to {q_pos}, outside header span "
                    f"[{header['q_start']}, {header['q_end']})",
                    line_no,
                    header["chain_id"],
                )

    if header is not None:
        # File ended while a chain was still open.
        raise _chain_error(
            "chain truncated: missing terminal block (a final line with a "
            "single size field)",
            line_no,
            header["chain_id"],
        )

    return ChainMap(chains)


def read_chain(source):
    """Read a UCSC chain file into a reusable :class:`ChainMap`.

    Chain files are parsed in the fixed ``t -> q`` direction (target to
    query); see https://genome.ucsc.edu/goldenPath/help/chain.html for the
    file format.  Supported headers carry ``tStrand +`` and ``qStrand`` of
    ``+`` or ``-``.  All coordinates are interpreted as zero-based,
    half-open intervals on forward-strand chromosomes.

    Parameters
    ----------
    source : str, os.PathLike or text file-like
        A local path to a chain file, optionally gzip-compressed (``.gz``),
        or an already opened text stream (e.g. an :class:`io.StringIO` or a
        file opened with :func:`open` in text mode).  Files opened by this
        function are closed by it; streams provided by the caller are left
        open.  An empty file yields an empty :class:`ChainMap`.

    Returns
    -------
    ChainMap
        Immutable, reusable chain collection.

    Raises
    ------
    ValueError
        If the chain file is malformed (wrong field counts, truncated
        chains, missing terminal block, duplicate chain ids, inconsistent
        chromosome lengths, out-of-range coordinates, blocks outside the
        declared header spans, or failure to reach the declared header
        endpoints).  Error messages include the 1-based file line number and,
        once known, the offending chain id.  A malformed file never yields a
        partial :class:`ChainMap`.
    OSError
        I/O errors raised while opening or reading the file are propagated
        unchanged.

    Notes
    -----
    The order of chains in the file does not affect the result.  The last
    chain does not require a trailing blank line.
    """
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with ExitStack() as stack:
            if path.endswith(".gz"):
                handle = stack.enter_context(gzip.open(path, "rt"))
            else:
                handle = stack.enter_context(open(path, encoding="utf-8"))
            return _read_chain_stream(handle)
    if hasattr(source, "readline"):
        # Borrowed stream: leave its lifecycle to the caller.
        return _read_chain_stream(source)
    raise ValueError(
        "source must be a local file path (optionally .gz) or an open text "
        f"stream, got {type(source).__name__}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _first_invalid_row(mask):
    """Return the first physical (0-based) row index where mask is True."""
    (idx,) = np.nonzero(mask)
    return int(idx[0])


def _validate_liftover_args(df, chains, min_match, multiple, cols, strand):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(chains, ChainMap):
        raise ValueError("chains must be a ChainMap produced by read_chain")
    if isinstance(min_match, bool) or not isinstance(min_match, (int, float)):
        raise ValueError("min_match must be a finite real number in (0, 1]")
    min_match = float(min_match)
    if not np.isfinite(min_match) or not (0.0 < min_match <= 1.0):
        raise ValueError("min_match must be a finite real number in (0, 1]")
    if multiple not in ("unique", "all"):
        raise ValueError("multiple must be either 'unique' or 'all'")

    if cols is None:
        from ..core.specs import _get_default_colnames

        cols = _get_default_colnames()
    cols = tuple(cols)
    if len(cols) != 3:
        raise ValueError("cols must be a triplet of column names")
    ccol, scol, ecol = cols
    if ccol == scol or ccol == ecol or scol == ecol:
        raise ValueError("chrom, start and end column names must be distinct")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("df must have unique column names")
    for name in cols:
        if name not in df.columns:
            raise ValueError(f"column {name!r} not found in df")

    if strand is not None:
        if not isinstance(strand, str):
            raise ValueError("strand must be a column name or None")
        if strand in cols:
            raise ValueError(
                "strand column must not overlap the coordinate columns"
            )
        if strand not in df.columns:
            raise ValueError(f"strand column {strand!r} not found in df")

    # Empty frames: dtypes still have to be of the right kind.
    from ..core.specs import is_chrom_dtype

    if not is_chrom_dtype(df[ccol].dtype):
        raise ValueError(
            "chrom column must have object, string or categorical dtype"
        )
    for name, label in ((scol, "start"), (ecol, "end")):
        series = df[name]
        if not pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(
            series
        ):
            raise ValueError(
                f"{label} column must have a non-boolean integer dtype"
            )

    n = len(df)
    if n == 0:
        return cols, min_match, n

    # Chrom values: non-null, non-empty strings.
    chrom_series = df[ccol]
    if isinstance(chrom_series.dtype, pd.CategoricalDtype):
        null_mask = chrom_series.isna().to_numpy()
        values = chrom_series.astype(object).to_numpy()
    else:
        values = chrom_series.to_numpy()
        null_mask = pd.isna(np.asarray(chrom_series))
    if null_mask.any():
        raise ValueError(
            "chrom column contains a missing value at physical row "
            f"{_first_invalid_row(null_mask)}"
        )
    empty_mask = np.fromiter(
        ((not isinstance(v, str)) or (len(v) == 0) for v in values),
        dtype=bool,
        count=len(values),
    )
    if empty_mask.any():
        raise ValueError(
            "chrom column must contain non-empty strings; first offending "
            f"physical row {_first_invalid_row(empty_mask)}"
        )

    # Integer coordinate columns, no missing data (covers nullable Int64).
    if df[scol].isna().any():
        bad = int(df[scol].isna().to_numpy().nonzero()[0][0])
        raise ValueError(
            f"start column contains missing data at physical row {bad}"
        )
    if df[ecol].isna().any():
        bad = int(df[ecol].isna().to_numpy().nonzero()[0][0])
        raise ValueError(
            f"end column contains missing data at physical row {bad}"
        )
    start = df[scol].to_numpy(dtype=np.int64)
    end = df[ecol].to_numpy(dtype=np.int64)

    bad_order = start >= end
    if bad_order.any():
        raise ValueError(
            "require 0 <= start < end; first offending physical row "
            f"{_first_invalid_row(bad_order)}"
        )
    bad_lo = start < 0
    if bad_lo.any():
        raise ValueError(
            "start must be >= 0; first offending physical row "
            f"{_first_invalid_row(bad_lo)}"
        )
    bad_hi = end > _COORD_MAX
    if bad_hi.any():
        raise ValueError(
            "end must be <= 2**63 - 1; first offending physical row "
            f"{_first_invalid_row(bad_hi)}"
        )

    # Intervals on known target chromosomes must respect declared sizes.
    t_sizes = chains.target_chromsizes
    known = np.fromiter(
        (v in t_sizes for v in values), dtype=bool, count=len(values)
    )
    if known.any():
        limits = np.array(
            [t_sizes[v] for v in values[known]], dtype=np.int64
        )
        over = end[known] > limits
        if over.any():
            local = int(over.nonzero()[0][0])
            physical = int(np.nonzero(known)[0][local])
            raise ValueError(
                f"interval on {values[physical]!r} exceeds declared target "
                f"chromosome length {t_sizes[values[physical]]} at physical "
                f"row {physical}"
            )

    if strand is not None:
        strand_series = df[strand]
        if strand_series.isna().any():
            bad = int(strand_series.isna().to_numpy().nonzero()[0][0])
            raise ValueError(
                f"strand column contains missing data at physical row {bad}"
            )
        sv = strand_series.astype(object).to_numpy()
        bad_strand = np.fromiter(
            (v not in ("+", "-", ".") for v in sv),
            dtype=bool,
            count=len(sv),
        )
        if bad_strand.any():
            raise ValueError(
                "strand values must be one of '+', '-', '.'; first offending "
                f"physical row {_first_invalid_row(bad_strand)}"
            )

    return cols, min_match, n


def _empty_results():
    """Return correctly typed empty mapped/unmapped data frames."""
    mapped = pd.DataFrame(
        {
            "input_row": pd.Series(dtype=np.int64),
            "chain_id": pd.Series(dtype=np.int64),
            "block": pd.Series(dtype=np.int64),
            "source_start": pd.Series(dtype=np.int64),
            "source_end": pd.Series(dtype=np.int64),
            "chrom": pd.Series(dtype=object),
            "start": pd.Series(dtype=np.int64),
            "end": pd.Series(dtype=np.int64),
            "strand": pd.Series(dtype=object),
            "match_fraction": pd.Series(dtype=np.float64),
        }
    )[_MAPPED_COLUMNS]
    unmapped = pd.DataFrame(
        {
            "input_row": pd.Series(dtype=np.int64),
            "reason": pd.Series(dtype=object),
            "best_fraction": pd.Series(dtype=np.float64),
            "candidate_count": pd.Series(dtype=np.int64),
        }
    )[_UNMAPPED_COLUMNS]
    return mapped, unmapped


def _overlapping_blocks(
    t_starts, t_ends, max_end, starts, ends
):
    """Find block overlaps for queries on a single target chromosome.

    Block arrays are sorted by start.  A left bisection on starts finds the
    first block that could begin before a query ends; a right bisection on
    the running-maximum of ends discards blocks that end before the query
    starts.  Only that contiguous slice of the block array is examined per
    query, so the cost is O((N + B) log B + K) rather than O(N * B).

    Returns
    -------
    query_idx, block_idx : numpy.ndarray
        Pairs of local query indices and global block-array indices that
        overlap with positive length.
    access_count : int
        Number of block-array positions examined while narrowing the search;
        benchmarks use this to rule out an N x B scan.
    """
    n = len(starts)
    if n == 0 or len(t_starts) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, 0

    right_cut = np.searchsorted(t_starts, ends, side="left")
    left_cut = np.searchsorted(max_end, starts, side="right")

    query_idx_parts = []
    block_idx_parts = []
    access_count = 0
    for i in range(n):
        lo, hi = int(left_cut[i]), int(right_cut[i])
        access_count += hi - lo
        if hi <= lo:
            continue
        cand = np.arange(lo, hi, dtype=np.int64)
        keep = (t_starts[cand] < ends[i]) & (t_ends[cand] > starts[i])
        cand = cand[keep]
        if cand.size:
            query_idx_parts.append(np.full(cand.size, i, dtype=np.int64))
            block_idx_parts.append(cand)

    if query_idx_parts:
        query_idx = np.concatenate(query_idx_parts)
        block_idx = np.concatenate(block_idx_parts)
    else:
        query_idx = np.empty(0, dtype=np.int64)
        block_idx = np.empty(0, dtype=np.int64)
    return query_idx, block_idx, access_count


# ---------------------------------------------------------------------------
# Public liftover entry point
# ---------------------------------------------------------------------------
def liftover(df, chains, *, min_match=0.95, multiple="unique", cols=None,
             strand=None):
    """Remap genomic intervals from the target to the query coordinate system.

    Each input interval is intersected with the aligned blocks of every
    chain in ``chains``.  A chain is a *candidate* for an interval when at
    least one of its blocks overlaps it with positive length; its match
    fraction is the union length of the input bases covered by that chain's
    blocks divided by the interval length.  Coverage is evaluated
    independently per chain: target gaps have no corresponding bases, query
    gaps never add coverage, and fractions from different chains are never
    combined to reach the threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Intervals in the target (``t``) coordinate system.  Must contain the
        coordinate columns (``('chrom', 'start', 'end')`` by default), with
        unique column names, non-empty string chromosome values, non-boolean
        integer coordinates without missing data (nullable ``Int64`` is
        accepted) and ``0 <= start < end <= 2**63 - 1``.  Intervals on
        chromosomes known to ``chains`` must not exceed the declared target
        chromosome length; intervals on unknown chromosomes are reported as
        unmapped with reason ``no_overlap``.  The frame is never modified.
    chains : ChainMap
        Chain collection returned by :func:`read_chain`.
    min_match : float, default 0.95
        Finite real number in the half-open interval ``(0, 1]`` (booleans
        are rejected).  A chain qualifies only when its match fraction is
        greater than or equal to this value.
    multiple : {'unique', 'all'}, default 'unique'
        With ``'unique'`` an interval is mapped only when exactly one chain
        qualifies; otherwise it is unmapped with reason ``ambiguous``.  With
        ``'all'`` fragments from every qualifying chain are retained.  Chain
        scores are never used to break ties and chains with different ids
        remain distinct candidates even when they produce identical
        coordinates.
    cols : (str, str, str) or None
        Names of the chrom, start, end columns.  Defaults to
        ``('chrom', 'start', 'end')``; the three names must be distinct.
    strand : str or None
        Name of a column holding input strand labels (``'+'``, ``'-'`` or
        ``'.'``).  When omitted, output fragments carry the chain
        ``qStrand``.  When given, the input strand is transformed through
        the mapping (minus-strand chains flip ``'+'``/``'-'``; ``'.'``
        stays ``'.'``).

    Returns
    -------
    mapped : pandas.DataFrame
        One row per intersecting block of every qualifying chain, with
        columns (in this order):

        ``input_row`` (int64)
            Physical row number of the input interval, starting at 0.
        ``chain_id`` (int64)
            Chain that produced the fragment.
        ``block`` (int64)
            Zero-based block number within that chain.
        ``source_start``, ``source_end`` (int64)
            Clipped fragment of the input interval on the forward target
            chromosome.
        ``chrom``, ``start``, ``end`` (object, int64, int64)
            Corresponding fragment in query coordinates; always forward
            coordinates with ``start < end``, including for minus-strand
            chains.  Gaps are not filled and adjacent chain blocks stay on
            separate rows.
        ``strand`` (object)
            Output strand (see the ``strand`` parameter).
        ``match_fraction`` (float64)
            Chain-level match fraction; identical for every fragment of the
            same (input row, chain) hit.

        Sorted by ``input_row``, numeric ``chain_id`` and
        ``source_start``; backed by a fresh :class:`pandas.RangeIndex`.
    unmapped : pandas.DataFrame
        One row per input interval without a usable mapping, with columns
        ``input_row`` (int64), ``reason`` (object), ``best_fraction``
        (float64) and ``candidate_count`` (int64).  ``reason`` is one of:

        ``no_overlap``
            No chain block intersects the interval; ``best_fraction`` is 0.
        ``low_coverage``
            At least one chain intersects but none reaches ``min_match``;
            ``best_fraction`` is the largest fraction among intersecting
            chains and ``candidate_count`` counts qualifying (coverage >=
            ``min_match``) chains, which is zero here.
        ``ambiguous``
            ``multiple='unique'`` with more than one qualifying chain;
            ``candidate_count`` is their number.

        Failed inputs never leave fragments in ``mapped``.  Both frames have
        independent :class:`pandas.RangeIndex` objects and retain their
        columns and dtypes when empty.

    Raises
    ------
    ValueError
        For invalid parameters or invalid input data.  Data-related messages
        identify the 0-based physical row.  Empty DataFrames are valid and
        yield two empty results.

    Notes
    -----
    The call is a pure function: it neither modifies ``df``/``chains`` nor
    keeps state between calls, so repeated, interleaved and previously
    failed calls do not affect later results.  Other columns of ``df`` are
    not copied; rejoin them later through ``input_row``.
    """
    (ccol, scol, ecol), min_match, n_rows = _validate_liftover_args(
        df, chains, min_match, multiple, cols, strand
    )

    if n_rows == 0:
        return _empty_results()

    chrom_values = df[ccol].astype(object).to_numpy()
    starts = df[scol].to_numpy(dtype=np.int64)
    ends = df[ecol].to_numpy(dtype=np.int64)
    input_strands = None
    if strand is not None:
        input_strands = df[strand].astype(object).to_numpy()

    # Per-input bookkeeping, indexed by physical row.
    best_fraction = np.zeros(n_rows, dtype=np.float64)
    candidate_count = np.zeros(n_rows, dtype=np.int64)
    # bit 0: at least one chain block intersects; bit 1: mapped successfully
    status = np.zeros(n_rows, dtype=np.int8)
    intersected, mapped_flag = 1, 2

    frag_parts = []

    # Group query rows by target chromosome so each block array is searched
    # at most once.
    unique_chroms, row_groups = np.unique(chrom_values, return_inverse=True)
    for t_name in unique_chroms:
        group_id = np.searchsorted(unique_chroms, t_name)
        local_rows = np.nonzero(row_groups == group_id)[0].astype(np.int64)
        block_arrays = chains._block_index.get(t_name)
        if block_arrays is None:
            continue
        (
            t_starts,
            t_ends,
            max_end,
            q_starts,
            q_ends,
            chain_ids,
            block_idxs,
        ) = block_arrays
        local_starts = starts[local_rows]
        local_ends = ends[local_rows]

        q_local, q_block, _access = _overlapping_blocks(
            t_starts, t_ends, max_end, local_starts, local_ends
        )
        if q_local.size == 0:
            continue

        physical_rows = local_rows[q_local]
        hit_chain_ids = chain_ids[q_block]
        hit_block_idxs = block_idxs[q_block]
        overlap_len = np.minimum(
            t_ends[q_block], ends[physical_rows]
        ) - np.maximum(t_starts[q_block], starts[physical_rows])

        # Aggregate covered input bases per (input row, chain).  Blocks of
        # one chain are disjoint on the target, so summing clipped overlap
        # lengths already gives the union length; different chains carry
        # different chain ids and are summed apart.
        order = np.lexsort((hit_chain_ids, physical_rows))
        physical_rows = physical_rows[order]
        hit_chain_ids = hit_chain_ids[order]
        hit_block_idxs = hit_block_idxs[order]
        q_block = q_block[order]
        overlap_len = overlap_len[order]

        group_key_change = np.ones(len(physical_rows), dtype=bool)
        if len(physical_rows) > 1:
            group_key_change[1:] = (
                physical_rows[1:] != physical_rows[:-1]
            ) | (hit_chain_ids[1:] != hit_chain_ids[:-1])
        group_starts_idx = np.nonzero(group_key_change)[0]
        covered = np.add.reduceat(overlap_len, group_starts_idx)

        group_rows = physical_rows[group_starts_idx]
        group_chains = hit_chain_ids[group_starts_idx]
        lengths = (ends - starts)[group_rows]
        fractions = covered.astype(np.float64) / lengths.astype(np.float64)

        # Best fraction among *intersecting* chains.
        np.maximum.at(best_fraction, group_rows, fractions)
        status[group_rows] |= intersected

        qualified = fractions >= min_match
        if not qualified.any():
            continue
        q_idx = np.nonzero(qualified)[0]
        qual_rows = group_rows[q_idx]
        qual_chains = group_chains[q_idx]
        qual_fractions = fractions[q_idx]

        # Each group is one intersecting chain, so this counts qualifying
        # chains per input row.
        np.add.at(candidate_count, qual_rows, 1)

        row_counts = np.zeros(n_rows, dtype=np.int64)
        np.add.at(row_counts, qual_rows, 1)
        if multiple == "unique":
            keep_group = row_counts[qual_rows] == 1
        else:
            keep_group = np.ones(len(qual_rows), dtype=bool)
        status[qual_rows[keep_group]] |= mapped_flag

        kept_pairs = {
            (int(r), int(cid))
            for r, cid, keep in zip(qual_rows, qual_chains, keep_group)
            if keep
        }
        if not kept_pairs:
            continue

        # Emit one fragment per intersecting block of each kept pair.
        keep_hit = np.fromiter(
            ((int(r), int(cid)) in kept_pairs
             for r, cid in zip(physical_rows, hit_chain_ids)),
            dtype=bool,
            count=len(physical_rows),
        )
        if not keep_hit.any():
            continue
        ph = physical_rows[keep_hit]
        cids = hit_chain_ids[keep_hit]
        bidx = hit_block_idxs[keep_hit]
        gidx = q_block[keep_hit]
        frac_lookup = {
            (int(r), int(cid)): float(f)
            for r, cid, f in zip(qual_rows, qual_chains, qual_fractions)
        }

        src_s = np.maximum(t_starts[gidx], starts[ph])
        src_e = np.minimum(t_ends[gidx], ends[ph])
        offset = src_s - t_starts[gidx]
        span = src_e - src_s
        # Block query coordinates are stored as forward coordinates, so the
        # strand comes from the owning chain, not coordinate ordering.
        qs_forward = q_starts[gidx]
        qe_forward = q_ends[gidx]
        q_strand_of = np.array(
            [chains.get(int(cid)).q_strand for cid in cids], dtype=object
        )
        plus_mask = q_strand_of == "+"
        out_start = np.empty(len(ph), dtype=np.int64)
        out_end = np.empty(len(ph), dtype=np.int64)
        out_start[plus_mask] = qs_forward[plus_mask] + offset[plus_mask]
        out_end[plus_mask] = (
            qs_forward[plus_mask] + offset[plus_mask] + span[plus_mask]
        )
        out_start[~plus_mask] = (
            qe_forward[~plus_mask] - offset[~plus_mask] - span[~plus_mask]
        )
        out_end[~plus_mask] = qe_forward[~plus_mask] - offset[~plus_mask]

        if input_strands is None:
            out_strand = q_strand_of
        else:
            in_strand = input_strands[ph]
            out_strand = np.empty(len(ph), dtype=object)
            dot = in_strand == "."
            flipped_plus = (~plus_mask) & (in_strand == "+")
            flipped_minus = (~plus_mask) & (in_strand == "-")
            same_plus = plus_mask & (in_strand == "+")
            same_minus = plus_mask & (in_strand == "-")
            out_strand[dot] = "."
            out_strand[same_plus | flipped_minus] = "+"
            out_strand[same_minus | flipped_plus] = "-"

        out_chrom = np.array(
            [chains.get(int(cid)).q_name for cid in cids], dtype=object
        )
        out_frac = np.array(
            [frac_lookup[(int(r), int(cid))] for r, cid in zip(ph, cids)],
            dtype=np.float64,
        )

        frag_parts.append(
            {
                "input_row": ph,
                "chain_id": cids,
                "block": bidx,
                "source_start": src_s,
                "source_end": src_e,
                "chrom": out_chrom,
                "start": out_start,
                "end": out_end,
                "strand": out_strand,
                "match_fraction": out_frac,
            }
        )

    # ------------------------------------------------------------------
    # Assemble the mapped frame
    # ------------------------------------------------------------------
    if frag_parts:
        combined = {
            key: np.concatenate([p[key] for p in frag_parts])
            for key in _MAPPED_COLUMNS
        }
        sort_order = np.lexsort(
            (
                combined["source_start"],
                combined["chain_id"],
                combined["input_row"],
            )
        )
        mapped = pd.DataFrame(
            {
                "input_row": combined["input_row"][sort_order].astype(np.int64),
                "chain_id": combined["chain_id"][sort_order].astype(np.int64),
                "block": combined["block"][sort_order].astype(np.int64),
                "source_start": combined["source_start"][sort_order].astype(
                    np.int64
                ),
                "source_end": combined["source_end"][sort_order].astype(
                    np.int64
                ),
                "chrom": pd.Series(
                    combined["chrom"][sort_order], dtype=object
                ),
                "start": combined["start"][sort_order].astype(np.int64),
                "end": combined["end"][sort_order].astype(np.int64),
                "strand": pd.Series(
                    combined["strand"][sort_order], dtype=object
                ),
                "match_fraction": combined["match_fraction"][sort_order].astype(
                    np.float64
                ),
            }
        )[_MAPPED_COLUMNS]
    else:
        mapped, _ = _empty_results()

    # ------------------------------------------------------------------
    # Assemble the unmapped frame
    # ------------------------------------------------------------------
    unmapped_rows = np.nonzero((status & mapped_flag) == 0)[0].astype(np.int64)
    if unmapped_rows.size:
        reasons = np.empty(unmapped_rows.size, dtype=object)
        for k, r in enumerate(unmapped_rows):
            if candidate_count[r] > 1:
                reasons[k] = "ambiguous"
            elif status[r] & intersected:
                reasons[k] = "low_coverage"
            else:
                reasons[k] = "no_overlap"
        unmapped = pd.DataFrame(
            {
                "input_row": unmapped_rows.astype(np.int64),
                "reason": pd.Series(reasons, dtype=object),
                "best_fraction": best_fraction[unmapped_rows].astype(np.float64),
                "candidate_count": candidate_count[unmapped_rows].astype(
                    np.int64
                ),
            }
        )[_UNMAPPED_COLUMNS]
    else:
        _, unmapped = _empty_results()

    return mapped, unmapped
