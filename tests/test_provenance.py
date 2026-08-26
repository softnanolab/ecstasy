"""Provenance capture — what code and what bytes produced a result.

The motivating failure is concrete: the MiniFold runner takes `minifold_src` as a PATH,
and whether the residx patch is applied inside that tree is the entire difference between
the intended chain break and the linker-only variant. `params.json` records only the path,
so today those two *different experiments* serialise identically. These tests pin that a
source-tree parameter is recorded as a commit plus a dirty flag, which is what makes them
distinguishable.

The second rule throughout: provenance must never be the thing that fails a run. Every
degraded case (no git, missing file, unreadable path) must record a reason, not raise.
"""
from __future__ import annotations

import subprocess

import pytest

from ecstasy import provenance


def _git_repo(path, *, commit=True):
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(a, cwd=str(path), capture_output=True, check=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (path / "model.py").write_text("def forward(): ...\n")
    if commit:
        run("git", "add", "-A")
        run("git", "commit", "-qm", "initial")
    return path


class TestGitState:
    def test_reports_sha_and_clean_tree(self, tmp_path):
        repo = _git_repo(tmp_path / "src")
        st = provenance.git_state(repo)
        assert st is not None
        assert len(st["sha"]) == 40
        assert st["dirty"] is False

    def test_detects_a_dirty_tree(self, tmp_path):
        """A patched-but-uncommitted vendored tree is exactly the MiniFold case."""
        repo = _git_repo(tmp_path / "src")
        (repo / "model.py").write_text("def forward(residx=None): ...\n")
        st = provenance.git_state(repo)
        assert st["dirty"] is True
        assert "model.py" in st["dirty_files"]

    def test_patched_and_unpatched_trees_are_distinguishable(self, tmp_path):
        """The whole point of the module, stated as a test."""
        repo = _git_repo(tmp_path / "src")
        clean = provenance.git_state(repo)
        (repo / "model.py").write_text("def forward(residx=None): ...\n")
        patched = provenance.git_state(repo)
        assert clean != patched
        assert clean["dirty"] != patched["dirty"]

    def test_untracked_file_leaves_tree_clean(self, tmp_path):
        """The question is 'was the code that ran the committed code' — scratch files in
        a working directory do not change the answer, and flagging them would make
        `dirty` so noisy it stopped being read."""
        repo = _git_repo(tmp_path / "src")
        (repo / "scratch.log").write_text("noise")
        assert provenance.git_state(repo)["dirty"] is False

    def test_porcelain_filenames_are_not_shifted(self, tmp_path):
        """`git status --porcelain` emits 'XY path', so an unstaged change starts with a
        literal space. Stripping it truncates every filename by one character."""
        repo = _git_repo(tmp_path / "src")
        (repo / "model.py").write_text("changed\n")
        assert provenance.git_state(repo)["dirty_files"] == ["model.py"]

    def test_non_repo_is_none_not_an_error(self, tmp_path):
        assert provenance.git_state(tmp_path / "not_a_repo") is None

    def test_accepts_a_file_inside_the_tree(self, tmp_path):
        repo = _git_repo(tmp_path / "src")
        assert provenance.git_state(repo / "model.py")["sha"] == provenance.git_state(repo)["sha"]


class TestFileIdentity:
    def test_records_size_and_hash(self, tmp_path):
        f = tmp_path / "w.ckpt"
        f.write_bytes(b"weights" * 100)
        rec = provenance.file_identity(f)
        assert rec["size"] == 700
        assert "sha256_ends" in rec and "mtime_utc" in rec

    def test_different_bytes_give_different_hashes(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"x" * 500)
        b.write_bytes(b"y" * 500)
        assert (provenance.file_identity(a)["sha256_ends"]
                != provenance.file_identity(b)["sha256_ends"])

    def test_same_size_different_content_is_caught(self, tmp_path):
        """Size alone would collide; the end-window hash must separate these."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"A" + b"0" * 999)
        b.write_bytes(b"B" + b"0" * 999)
        assert (provenance.file_identity(a)["sha256_ends"]
                != provenance.file_identity(b)["sha256_ends"])

    def test_follows_symlinks_and_records_the_target(self, tmp_path):
        """Weight paths under ${ECSTASY_ROOT} are symlinks; a target can be repointed
        without any artifact changing, so the resolved path is recorded."""
        real = tmp_path / "real.ckpt"
        real.write_bytes(b"z" * 64)
        link = tmp_path / "link.ckpt"
        link.symlink_to(real)
        rec = provenance.file_identity(link)
        assert rec["path"].endswith("link.ckpt")
        assert rec["resolved"].endswith("real.ckpt")
        assert rec["size"] == 64

    def test_missing_file_records_an_error_rather_than_raising(self, tmp_path):
        rec = provenance.file_identity(tmp_path / "nope.ckpt")
        assert "error" in rec and "size" not in rec

    def test_full_hash_is_opt_in(self, tmp_path):
        f = tmp_path / "w"
        f.write_bytes(b"q" * 32)
        assert "sha256" in provenance.file_identity(f, full_hash=True)
        assert "sha256_ends" in provenance.file_identity(f)


class TestParamsProvenance:
    def test_source_tree_param_records_a_commit(self, tmp_path):
        repo = _git_repo(tmp_path / "minifold")
        got = provenance.params_provenance({"minifold_src": str(repo)})
        assert got["minifold_src"]["kind"] == "source_tree"
        assert len(got["minifold_src"]["git"]["sha"]) == 40

    def test_weight_param_records_bytes(self, tmp_path):
        ckpt = tmp_path / "m.ckpt"
        ckpt.write_bytes(b"w" * 128)
        got = provenance.params_provenance({"checkpoint": str(ckpt)})
        assert got["checkpoint"]["kind"] == "file"
        assert got["checkpoint"]["size"] == 128

    def test_non_path_params_are_ignored(self, tmp_path):
        got = provenance.params_provenance(
            {"num_recycles": 3, "contact_cutoff_bin": 17, "name": "not/a/real/path"})
        assert got == {}

    def test_nonexistent_paths_are_skipped_not_guessed_at(self):
        assert provenance.params_provenance({"checkpoint": "/no/such/file.ckpt"}) == {}

    def test_empty_params_are_fine(self):
        assert provenance.params_provenance({}) == {}
        assert provenance.params_provenance(None) == {}


class TestCapture:
    def test_captures_the_expected_shape(self):
        rec = provenance.capture(params={"num_recycles": 3})
        assert "captured_utc" in rec
        assert "ecstasy" in rec and "submodules" in rec
        assert rec["env"]["python"]

    def test_records_scheduler_job_id_when_present(self, monkeypatch):
        monkeypatch.setenv("PBS_JOBID", "3911588.pbs-7")
        rec = provenance.capture()
        assert rec["job"]["PBS_JOBID"] == "3911588.pbs-7"

    def test_no_scheduler_means_no_job_key(self, monkeypatch):
        for k in ("PBS_JOBID", "SLURM_JOB_ID", "JOB_ID"):
            monkeypatch.delenv(k, raising=False)
        assert "job" not in provenance.capture()

    def test_never_raises_on_a_bogus_root(self, tmp_path):
        rec = provenance.capture(repo_root=tmp_path / "nowhere")
        assert "error" in rec["ecstasy"]

    def test_summarise_is_a_single_line(self, tmp_path):
        repo = _git_repo(tmp_path / "src")
        rec = provenance.capture(params={"minifold_src": str(repo)})
        line = provenance.summarise(rec)
        assert "\n" not in line
        assert "ecstasy" in line and "minifold_src" in line

    def test_summarise_flags_a_dirty_tree(self, tmp_path):
        repo = _git_repo(tmp_path / "src")
        (repo / "model.py").write_text("patched\n")
        rec = provenance.capture(params={"minifold_src": str(repo)})
        assert "-dirty" in provenance.summarise(rec)
