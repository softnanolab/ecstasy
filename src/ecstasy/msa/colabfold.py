"""ColabFold MMseqs2 API client + paired-A3M parser.

Mirrors the official MSA Pairformer + MMseqs2 colab notebook's MMseqs2
integration. This module is the canonical home for:

  - submit/poll/download against ``api.colabfold.com/ticket/pair``
  - parsing the paired pair.a3m that the server returns
    (two mini-a3ms separated by a NUL byte at byte level)
  - notebook-style ``save_msa`` post-filters (coverage, identity,
    genomic distance — see ``apply_save_msa_filters``)

Future datasets that need paired MSAs from the ColabFold server should
import this module rather than re-implementing the parser.
"""
from __future__ import annotations

import re
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import requests

HOST = "https://api.colabfold.com"
DEFAULT_MODE = "paircomplete-pairfilterprox_20"  # notebook default

CHECK_INTERVAL_S = 5
SUBMIT_TIMEOUT_S = 30
POLL_TIMEOUT_S = 60
DOWNLOAD_TIMEOUT_S = 120
MAX_RETRIES = 6
MAX_WAIT_S = 1800
RETRY_BACKOFF_BASE_S = 8


# ---- HTTP client ----------------------------------------------------------


def submit_pair(
    session: requests.Session,
    fasta: str,
    *,
    mode: str = DEFAULT_MODE,
    max_retries: int = MAX_RETRIES,
) -> str:
    """POST a paired FASTA to /ticket/pair; return the server job id.

    Backs off exponentially on 429 (rate limit) and transient network errors.
    """
    for attempt in range(max_retries):
        try:
            r = session.post(
                f"{HOST}/ticket/pair",
                data={"q": fasta, "mode": mode},
                timeout=SUBMIT_TIMEOUT_S,
            )
            if r.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
                continue
            r.raise_for_status()
            j = r.json()
            if "id" not in j:
                raise RuntimeError(f"no id in submit response: {j}")
            return j["id"]
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"submit failed: {e}")
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
    raise RuntimeError("submit retries exhausted")


def submit_msa(
    session: requests.Session,
    fasta: str,
    *,
    mode: str = "env",
    max_retries: int = MAX_RETRIES,
) -> str:
    """POST a single-sequence FASTA to /ticket/msa; return the server job id.

    Unpaired MSA (uniref + env), used for homodimer tiling. Mirrors submit_pair.
    """
    for attempt in range(max_retries):
        try:
            r = session.post(f"{HOST}/ticket/msa", data={"q": fasta, "mode": mode},
                             timeout=SUBMIT_TIMEOUT_S)
            if r.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
                continue
            r.raise_for_status()
            j = r.json()
            if "id" not in j:
                raise RuntimeError(f"no id in submit response: {j}")
            return j["id"]
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"submit_msa failed: {e}")
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
    raise RuntimeError("submit_msa retries exhausted")


def parse_unpaired_a3m_bytes(tar_bytes: bytes) -> list[str]:
    """Extract uniref.a3m (+ bfd…a3m) from the /ticket/msa tarball; matched-only seqs.

    Returns deduped matched-only (insertions stripped) sequences, query first.
    Used to tile a homodimer's paired MSA from its single-chain alignment.
    """
    names = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
    seqs: list[str] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tf:
        members = {m.name.split("/")[-1]: m for m in tf.getmembers()}
        for nm in names:
            m = members.get(nm)
            if m is None:
                continue
            text = tf.extractfile(m).read().decode("utf-8", errors="replace")
            cur: list[str] = []
            for line in text.splitlines():
                if line.startswith(">"):
                    if cur:
                        s = _strip_inserts("".join(cur))
                        if s and s not in seen:
                            seen.add(s); seqs.append(s)
                    cur = []
                elif line:
                    cur.append(line.rstrip())
            if cur:
                s = _strip_inserts("".join(cur))
                if s and s not in seen:
                    seen.add(s); seqs.append(s)
    return seqs


def poll_until_done(
    session: requests.Session,
    job_id: str,
    *,
    max_wait_s: int = MAX_WAIT_S,
) -> None:
    """Block until /ticket/{job_id} reports COMPLETE; raise on ERROR or timeout."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = session.get(f"{HOST}/ticket/{job_id}", timeout=POLL_TIMEOUT_S)
            if r.status_code == 429:
                time.sleep(CHECK_INTERVAL_S * 2)
                continue
            r.raise_for_status()
            j = r.json()
            status = j.get("status", "")
            if status == "COMPLETE":
                return
            if status == "ERROR":
                raise RuntimeError(f"server reported ERROR for job {job_id}: {j}")
        except requests.RequestException:
            pass  # transient
        time.sleep(CHECK_INTERVAL_S)
    raise TimeoutError(f"job {job_id} did not complete within {max_wait_s}s")


def download_results(
    session: requests.Session,
    job_id: str,
    *,
    max_retries: int = MAX_RETRIES,
) -> bytes:
    """GET the .tar.gz result blob; raw bytes for the caller to extract."""
    for attempt in range(max_retries):
        try:
            r = session.get(f"{HOST}/result/download/{job_id}", timeout=DOWNLOAD_TIMEOUT_S)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("download retries exhausted")


def make_query_fasta(seqs: list[str], *, start_id: int = 101) -> str:
    """Build the FASTA query string the colab notebook uses.

    IDs start at 101 (notebook convention; the server keys per-chain output
    on these numeric labels in `pair.a3m`).
    """
    return "\n".join(f">{i}\n{s}" for i, s in enumerate(seqs, start=start_id)) + "\n"


def clean_sequence(seq: str) -> str:
    """Notebook prepare_sequences: uppercase + strip non-canonical chars."""
    return re.sub(r"[^A-Z]", "", seq.upper())


# ---- Parser ---------------------------------------------------------------


@dataclass
class A3mEntry:
    """One row in a paired MSA, parsed from the colabfold server output.

    `coverage` and `identity` are from the header's tab-separated metadata
    (`alnscore  identity  evalue  qstart  qend  qlen  tstart  tend  tlen  alnlen`);
    the query row gets coverage=identity=1.0. `uid` is the extracted UniProt
    accession when the header is a UniRef entry; `uniprot_num` is the
    notebook's encoded ordering used for the genomic-distance filter.
    """
    header: str
    sequence: str        # matched-only (insertions stripped)
    coverage: float
    identity: float
    uid: str = ""
    uniprot_num: int = 0
    has_uniref: bool = False
    is_query: bool = False


def _strip_inserts(seq: str) -> str:
    """A3M format: lowercase letters and '.' are insertions; remove."""
    return re.sub(r"[a-z.]", "", seq)


def parse_chain_a3m_chunk(text: str, *, extract_metadata: bool = False) -> list[A3mEntry]:
    """Parse a single chain's a3m chunk (after byte-level NUL split).

    The first record is the query (label `>101` / `>102` / ...); we keep it
    as the row-0 entry. Subsequent records are UniRef hits.

    `extract_metadata=True` populates coverage/identity/uid/uniprot_num
    from the header (required by the notebook `save_msa` post-filters).
    Cheaper to skip when you only want the stitched sequences.
    """
    entries: list[A3mEntry] = []
    cur_header: str | None = None
    cur_seq_parts: list[str] = []
    is_first = True

    def _flush() -> None:
        nonlocal is_first
        if cur_header is None:
            return
        seq = _strip_inserts("".join(cur_seq_parts))
        e = A3mEntry(header=cur_header.lstrip(">"), sequence=seq,
                     coverage=1.0, identity=1.0, is_query=is_first)
        if extract_metadata:
            _populate_metadata(e, cur_header)
        entries.append(e)
        is_first = False

    for line in text.splitlines():
        if line.startswith(">"):
            _flush()
            cur_header = line
            cur_seq_parts = []
        elif line:
            cur_seq_parts.append(line.rstrip())
    _flush()
    return entries


def parse_paired_a3m_bytes(tar_bytes: bytes, *, extract_metadata: bool = False) -> list[list[A3mEntry]]:
    """Extract `pair.a3m` from the colabfold tarball and parse per-chain.

    The server returns a single `pair.a3m` containing N mini-a3ms (one per
    chain in the query) separated by a NUL byte at the byte level. Returns
    a list of length N; each item is the parsed entry list for that chain.
    """
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("pair.a3m"):
                raw = tf.extractfile(m).read()
                break
        else:
            raise FileNotFoundError("pair.a3m not found in tarball")
    raw = raw.rstrip(b"\x00")
    chunks = [c for c in raw.split(b"\x00") if c.strip()]
    return [
        parse_chain_a3m_chunk(c.decode("utf-8", errors="replace"),
                              extract_metadata=extract_metadata)
        for c in chunks
    ]


def _populate_metadata(entry: A3mEntry, raw_header: str) -> None:
    """Fill coverage/identity/uid/uniprot_num/has_uniref from the FASTA header."""
    entry.has_uniref = "UniRef" in raw_header
    entry.uid = _extract_uniprot_id(raw_header)
    if entry.uid:
        entry.uniprot_num = _uniprot_to_number([entry.uid])[0]
    if entry.is_query:
        return
    parts = raw_header.split("\t")
    if len(parts) < 10:
        entry.coverage = 0.0
        entry.identity = 0.0
        return
    try:
        entry.identity = float(parts[2])
        q_start = int(parts[4])
        q_end = int(parts[5])
        q_len = int(parts[6])
        entry.coverage = (q_end - q_start + 1) / q_len
    except Exception:  # noqa: BLE001
        entry.coverage = 0.0
        entry.identity = 0.0


# ---- Stitching ------------------------------------------------------------


def stitch_paired_msa(
    per_chain: list[list[A3mEntry]],
    expected_chain_lens: list[int],
) -> list[tuple[str, str]]:
    """Row-by-row stitch per-chain entries into a single concatenated MSA.

    Takes `min(depth_per_chain)` rows so the paired structure is preserved.
    Skips rows where any chain piece doesn't match its expected matched-only
    length (defensive; should never happen for well-formed server output but
    can after aggressive filtering).
    """
    depth = min(len(c) for c in per_chain)
    combined: list[tuple[str, str]] = []
    for r in range(depth):
        entries = [per_chain[c][r] for c in range(len(per_chain))]
        if any(len(e.sequence) != expected_chain_lens[ci] for ci, e in enumerate(entries)):
            continue
        header = "|".join(e.header for e in entries) if not entries[0].is_query else "query"
        combined.append((header, "".join(e.sequence for e in entries)))
    return combined


# ---- Save-MSA filters (notebook port) ------------------------------------


@dataclass
class SaveMsaFilters:
    """MSA Pairformer paper (SI) pairing/filter defaults. None disables a filter.

    From the SI (Akiyama et al. 2025, bioRxiv 2025.08.02.668173): the accessible
    MMseqs2 interface-contact route uses proximity-based pairing with **distance
    ≤ 20** (paircomplete-pairfilterprox_20), **coverage ≥ 75%**, **minimum query
    identity ≥ 30%**, and **512 sequences** (hhfilter, applied at model load).
    (The interactive notebook UI used the more aggressive cov=0.75/id=0.15/Δgene=1;
    those are demo defaults, not the benchmark settings. The paper's HEADLINE
    interface numbers used pre-computed MSAs from Ovchinnikov et al.; this MMseqs2
    path is the authors' provided accessible reproduction.)
    """
    min_coverage: float | None = 0.75
    min_identity: float | None = 0.30
    max_genomic_distance: int | None = 20


@dataclass
class FilterStats:
    raw: int = 0
    filtered_cov: int = 0
    filtered_id: int = 0
    filtered_dist: int = 0
    filtered_len: int = 0
    kept: int = 0


def apply_save_msa_filters(
    per_chain: list[list[A3mEntry]],
    expected_chain_lens: list[int],
    filters: SaveMsaFilters = SaveMsaFilters(),
) -> tuple[list[tuple[str, str]], FilterStats]:
    """Notebook ``save_msa`` post-filter, then stitch.

    Returns (stitched_entries, stats). Stats are useful for the per-entry
    log written by ``11_apply_notebook_filters.py``.
    """
    depth = min(len(c) for c in per_chain)
    stats = FilterStats(raw=depth)
    kept: list[tuple[str, str]] = []
    for r in range(depth):
        entries = [per_chain[c][r] for c in range(len(per_chain))]
        seqs = [e.sequence for e in entries]
        if any(len(s) != expected_chain_lens[c] for c, s in enumerate(seqs)):
            stats.filtered_len += 1
            continue
        if not entries[0].is_query:
            if filters.min_coverage is not None and any(e.coverage < filters.min_coverage for e in entries):
                stats.filtered_cov += 1
                continue
            if filters.min_identity is not None and any(e.identity < filters.min_identity for e in entries):
                stats.filtered_id += 1
                continue
            if filters.max_genomic_distance is not None:
                # Notebook save_msa applies the genomic-distance (operon-proximity) test
                # ONLY to rows with a UniRef accession on every chain. Rows lacking UniRef
                # have no operon evidence and are KEPT (not dropped) — matching the notebook.
                if all(e.has_uniref and e.uid for e in entries):
                    dists = _calc_distances([e.uniprot_num for e in entries])
                    if dists and dists[0] != -1 and dists[0] > filters.max_genomic_distance:
                        stats.filtered_dist += 1
                        continue
        header = "|".join(e.header for e in entries) if not entries[0].is_query else "query"
        kept.append((header, "".join(seqs)))
        stats.kept += 1
    return kept, stats


def write_a3m(entries: Iterable[tuple[str, str]], dest) -> None:
    """Write [(header, sequence), ...] to a plain A3M file at `dest`."""
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for header, seq in entries:
            f.write(f">{header}\n{seq}\n")


# ---- UniProt ID -> notebook-encoded number (for genomic distance) --------
#
# Ported verbatim from `ColabFoldPairedMSA._uniprot_to_number` in the colab
# notebook. The exact encoding is a noisy proxy for "are these two UniRef
# accessions from adjacent genes on the same genome?" — the actual ordering
# is a hash of UniProt's accession assignment policy. The notebook authors
# treat `|dist| <= max_genomic_distance` as the operon proximity rule.
from string import ascii_uppercase

_PA = {a: 0 for a in ascii_uppercase}
for _a in ["O", "P", "Q"]:
    _PA[_a] = 1

_MA: list[list[dict]] = [[{} for _ in range(6)], [{} for _ in range(6)]]
for _n, _t in enumerate(range(10)):
    for _i in [0, 1]:
        for _j in [0, 4]:
            _MA[_i][_j][str(_t)] = _n

for _n, _t in enumerate(list(ascii_uppercase) + list(range(10))):
    for _i in [0, 1]:
        for _j in [1, 2]:
            _MA[_i][_j][str(_t)] = _n
    _MA[1][3][str(_t)] = _n

for _n, _t in enumerate(ascii_uppercase):
    _MA[0][3][str(_t)] = _n
    for _i in [0, 1]:
        _MA[_i][5][str(_t)] = _n

_UPI_ENCODING: dict[str, int] = {}
_HEX = list(range(10)) + ["A", "B", "C", "D", "E", "F"]
for _n, _c in enumerate(_HEX):
    _UPI_ENCODING[str(_c)] = _n


def _extract_uniprot_id(header: str) -> str:
    pos = header.find("UniRef")
    if pos == -1:
        return ""
    start = header.find("_", pos)
    if start == -1:
        return ""
    start += 1
    end = start
    while end < len(header) and header[end] not in " _\t":
        end += 1
    uid = header[start:end]
    if len(uid) >= 3 and uid[:3] == "UPI":
        return uid
    if len(uid) not in (6, 10):
        return ""
    if not uid[0].isalpha():
        return ""
    return uid


def _uniprot_to_number(uniprot_ids: list[str]) -> list[int]:
    numbers: list[int] = []
    for uni in uniprot_ids:
        if not uni or not uni[0].isalpha():
            numbers.append(0)
            continue
        if uni.startswith("UPI") and len(uni) == 13:
            hex_part = uni[3:]
            num = 0
            tot = 1
            for u in reversed(hex_part):
                if str(u) in _UPI_ENCODING:
                    num += _UPI_ENCODING[str(u)] * tot
                    tot *= 16
                else:
                    num = 0
                    break
            numbers.append(num + 10**15)
            continue
        p = _PA.get(uni[0], 0)
        tot, num = 1, 0
        if len(uni) == 10:
            for n, u in enumerate(reversed(uni[-4:])):
                if str(u) in _MA[p][n]:
                    num += _MA[p][n][str(u)] * tot
                    tot *= len(_MA[p][n].keys())
        for n, u in enumerate(reversed(uni[:6])):
            if n < len(_MA[p]) and str(u) in _MA[p][n]:
                num += _MA[p][n][str(u)] * tot
                tot *= len(_MA[p][n].keys())
        numbers.append(num)
    return numbers


def _calc_distances(uniprot_nums: list[int]) -> list[int]:
    out = []
    for i in range(1, len(uniprot_nums)):
        if uniprot_nums[i - 1] and uniprot_nums[i]:
            out.append(abs(uniprot_nums[i] - uniprot_nums[i - 1]))
        else:
            out.append(-1)
    return out
