"""Tests for ecstasy.structure.pdb — the atom37 <-> PDB serialisation.

The point of this module is byte-compatibility with the PDBs the MENTOS DockQ
checkpoint series was scored on, so the tests pin the record layout itself (column
positions, the +1 on residue_index, masking rules) rather than just round-tripping.
A change that keeps the round-trip working but shifts a column would move every DockQ
score away from the published series, and these tests are what catches it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.structure.pdb import (
    ATOM_TYPES,
    CA_INDEX,
    RESNAMES,
    atom_coords,
    chain_of,
    load_structure_npz,
    read_atom_lines,
    render_structure_npz,
    replace_coords,
    write_atom37_pdb,
    write_structure_npz,
)


def _bundle(n_a: int = 3, n_b: int = 2, atoms: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Two-chain atom37 bundle with `atoms` present on every residue."""
    n = n_a + n_b
    pos = np.zeros((n, len(ATOM_TYPES), 3), dtype=np.float32)
    for i in range(n):
        for a in atoms:
            pos[i, a] = (i + 1, a, 0.5)
    mask = np.zeros((n, len(ATOM_TYPES)), dtype=bool)
    mask[:, list(atoms)] = True
    return {
        "atom37_positions": pos,
        "atom37_mask": mask,
        "aatype": np.zeros(n, dtype=np.int8),               # ALA
        "asym_id": np.array([0] * n_a + [1] * n_b, dtype=np.int8),
        "residue_index": np.array(list(range(n_a)) + list(range(n_b)), dtype=np.int32),
    }


def _write(tmp_path, bundle, **kw):
    return write_atom37_pdb(
        tmp_path / "s.pdb",
        positions=bundle["atom37_positions"], atom_mask=bundle["atom37_mask"],
        aatype=bundle["aatype"], asym_id=bundle["asym_id"],
        residue_index=bundle["residue_index"], **kw)


class TestWriteAtom37Pdb:
    def test_record_layout_matches_mentos(self, tmp_path):
        """Columns are the ones DockQ reads; this is the byte-compatibility pin."""
        lines = read_atom_lines(_write(tmp_path, _bundle(n_a=1, n_b=0, atoms=(1,))))
        assert len(lines) == 1
        ln = lines[0]
        assert ln[:6] == "ATOM  "
        assert ln[6:11] == "    1"          # serial, right-aligned in 5
        assert ln[12:16] == " CA "          # atom name
        assert ln[17:20] == "ALA"           # resname
        assert ln[21] == "A"                # chain
        assert ln[22:26] == "   1"          # resnum = residue_index + 1
        assert ln[30:38] == "   1.000"      # x
        assert ln[54:] == "  1.00  0.00           C"

    def test_residue_index_is_offset_by_one_and_resets_per_chain(self, tmp_path):
        lines = read_atom_lines(_write(tmp_path, _bundle(n_a=3, n_b=2, atoms=(1,))))
        a = [int(ln[22:26]) for ln in lines if chain_of(ln) == "A"]
        b = [int(ln[22:26]) for ln in lines if chain_of(ln) == "B"]
        assert a == [1, 2, 3]
        assert b == [1, 2]

    def test_chain_letter_follows_asym_id(self, tmp_path):
        lines = read_atom_lines(_write(tmp_path, _bundle(atoms=(1,))))
        assert sorted({chain_of(ln) for ln in lines}) == ["A", "B"]

    def test_unmasked_atoms_are_omitted(self, tmp_path):
        b = _bundle(n_a=2, n_b=0, atoms=(0, 1, 2))
        b["atom37_mask"][0, 2] = False                       # drop one C
        lines = read_atom_lines(_write(tmp_path, b))
        assert len(lines) == 5                               # 3 + 2

    def test_residue_mask_skips_whole_residues(self, tmp_path):
        b = _bundle(n_a=3, n_b=0, atoms=(1,))
        keep = np.array([True, False, True])
        lines = read_atom_lines(_write(tmp_path, b, residue_mask=keep))
        assert [int(ln[22:26]) for ln in lines] == [1, 3]

    def test_negative_asym_id_is_skipped(self, tmp_path):
        b = _bundle(n_a=2, n_b=1, atoms=(1,))
        b["asym_id"] = np.array([-1, 0, 1], dtype=np.int8)
        assert len(read_atom_lines(_write(tmp_path, b))) == 2

    def test_serials_are_contiguous_across_skips(self, tmp_path):
        b = _bundle(n_a=3, n_b=0, atoms=(0, 1))
        b["atom37_mask"][1, :] = False
        lines = read_atom_lines(_write(tmp_path, b))
        assert [int(ln[6:11]) for ln in lines] == [1, 2, 3, 4]

    def test_file_ends_with_END(self, tmp_path):
        text = _write(tmp_path, _bundle(atoms=(1,))).read_text()
        assert text.endswith("END\n")

    def test_every_resname_is_writable(self, tmp_path):
        n = len(RESNAMES)
        b = _bundle(n_a=n, n_b=0, atoms=(1,))
        b["aatype"] = np.arange(n, dtype=np.int8)
        names = [ln[17:20] for ln in read_atom_lines(_write(tmp_path, b))]
        assert names == list(RESNAMES)

    @pytest.mark.parametrize("field", ["atom37_mask", "aatype", "asym_id",
                                       "residue_index"])
    def test_shape_mismatch_is_rejected(self, tmp_path, field):
        b = _bundle()
        b[field] = b[field][:-1]
        with pytest.raises(ValueError, match="shape"):
            _write(tmp_path, b)


class TestStructureNpz:
    def test_round_trip(self, tmp_path):
        b = _bundle()
        p = write_structure_npz(tmp_path / "structure.npz", **b)
        got = load_structure_npz(p)
        for k, v in b.items():
            np.testing.assert_allclose(got[k], v)

    def test_missing_key_is_rejected_on_write(self, tmp_path):
        b = _bundle()
        b.pop("asym_id")
        with pytest.raises(KeyError, match="asym_id"):
            write_structure_npz(tmp_path / "structure.npz", **b)

    def test_missing_key_is_rejected_on_read(self, tmp_path):
        p = tmp_path / "structure.npz"
        np.savez_compressed(p, atom37_positions=np.zeros((1, 37, 3)))
        with pytest.raises(KeyError, match="missing"):
            load_structure_npz(p)

    def test_render_matches_the_direct_writer(self, tmp_path):
        """The npz path and the array path must produce identical bytes — that identity
        is what lets predictions and natives be compared without a serialisation gap."""
        b = _bundle()
        npz = write_structure_npz(tmp_path / "structure.npz", **b)
        via_npz = render_structure_npz(npz, tmp_path / "from_npz.pdb").read_text()
        direct = _write(tmp_path, b).read_text()
        assert via_npz == direct


class TestPlainPdbHelpers:
    def test_atom_coords_reads_the_written_positions(self, tmp_path):
        b = _bundle(n_a=2, n_b=0, atoms=(CA_INDEX,))
        lines = read_atom_lines(_write(tmp_path, b))
        np.testing.assert_allclose(atom_coords(lines),
                                   b["atom37_positions"][:, CA_INDEX], atol=1e-3)

    def test_replace_coords_changes_only_the_coordinate_columns(self, tmp_path):
        lines = read_atom_lines(_write(tmp_path, _bundle(atoms=(1,))))
        new = replace_coords(lines, np.zeros((len(lines), 3)))
        for old, fresh in zip(lines, new):
            assert fresh[:30] == old[:30]
            assert fresh[54:] == old[54:]
        np.testing.assert_allclose(atom_coords(new), 0.0)

    def test_replace_coords_rejects_a_length_mismatch(self, tmp_path):
        lines = read_atom_lines(_write(tmp_path, _bundle(atoms=(1,))))
        with pytest.raises(ValueError):
            replace_coords(lines, np.zeros((len(lines) - 1, 3)))

    def test_atom_coords_of_nothing_is_empty(self):
        assert atom_coords([]).shape == (0, 3)
