"""CPU unit coverage for the aarch64 DeepInteract feature reimplementations
(_di_aarch64_feats): the PSAIA-format CX .tbl and the DSSP-dict replacement.

These are golden-format tests — they assert the exact shapes DeepInteract's parsers
read (`.tbl` 6 CX stats at whitespace-split indices [3:9] after a `chain` header;
DSSP dict keyed (chain, (' ', resseq_int, ' ')) with SS at [2], RSA at [3]). No GPU.
pydssp is optional (SS falls back to '-' when absent), so SS content is not asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("Bio")

from ecstasy.models._runners import _di_aarch64_feats as F  # noqa: E402


def _synthetic_pdb(path, n_res=6):
    """A minimal valid single-chain poly-ALA PDB (N/CA/C/O/CB per residue), residues
    strung ~3.8 A apart along x so neighbour counts (hence CX) vary by position."""
    offs = {"N": (0.0, 0.0, 0.0), "CA": (0.7, 0.8, 0.0), "C": (1.4, 0.0, 0.6),
            "O": (1.5, -1.1, 0.4), "CB": (0.7, 1.6, 1.0)}
    lines, serial = [], 1
    for r in range(1, n_res + 1):
        bx = (r - 1) * 3.8
        for name, (dx, dy, dz) in offs.items():
            x, y, z = bx + dx, dy, dz
            elem = name[0]
            lines.append(
                f"ATOM  {serial:>5} {name:<4} ALA A{r:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}")
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_write_psaia_cx_tbl_format(tmp_path):
    pdb = _synthetic_pdb(tmp_path / "x.pdb", n_res=6)
    out = tmp_path / "x.tbl"
    n = F.write_psaia_cx_tbl(str(pdb), str(out))
    assert n == 6

    text = out.read_text().splitlines()
    # the parser turns on only after a line whose first token is exactly 'chain'
    assert "chain" in [ln.split()[0] for ln in text if ln.split()]
    hdr = next(i for i, ln in enumerate(text) if ln.split() and ln.split()[0] == "chain")
    rows = [ln.split() for ln in text[hdr + 1:] if ln.strip()]
    assert len(rows) == 6
    for tok in rows:
        assert len(tok) >= 9                       # chain resid resname + 6 CX stats
        cx = [float(v) for v in tok[3:9]]          # indices [3:9] = the 6 CX statistics
        assert all(np.isfinite(c) and c >= 0 for c in cx)
        assert tok[0] == "A"                       # chain id preserved
    # CX is not a flat constant across residues (ends protrude more than the middle)
    avg_cx = np.array([float(t[3]) for t in rows])
    assert avg_cx.std() > 0


def test_write_psaia_cx_tbl_empty_input(tmp_path):
    empty = tmp_path / "empty.pdb"
    empty.write_text("END\n")
    out = tmp_path / "empty.tbl"
    assert F.write_psaia_cx_tbl(str(empty), str(out)) == 0
    # still a valid stub the parser can consume (header + 'chain', no rows)
    assert "chain" in out.read_text()


def test_dssp_dict_shape_and_rsa_range(tmp_path):
    pdb = _synthetic_pdb(tmp_path / "y.pdb", n_res=5)
    d = F.dssp_dict_for_pdb(str(pdb))
    assert len(d) == 5
    for key, val in d.items():
        chain, resid = key
        assert chain == "A"
        assert resid[0] == " " and isinstance(resid[1], int) and resid[2] == " "
        ss, rsa = val[2], val[3]                   # the only indices the consumer reads
        assert isinstance(ss, str)
        assert (isinstance(rsa, float) and 0.0 <= rsa <= 1.0) or (rsa != rsa)  # in [0,1] or NaN
