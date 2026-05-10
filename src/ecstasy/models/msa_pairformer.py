from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter, register_model

_HERE = Path(__file__).resolve().parent


@register_model
class MSAPairformerAdapter(ModelAdapter):
    """MSA Pairformer (Akiyama et al.) — single-MSA contact prediction.

    Marked `needs_msa = True` because the value of MSA Pairformer is the MSA;
    a single-sequence input gives a degenerate baseline. The adapter loads the
    MSA from `complex_a3m_dir`/`<entry_id>.a3m` (ColabFold format with the
    `#L1,L2<TAB>copies1,copies2` header) — generate via the
    softnanolab/colabfold-local pipeline.
    """

    name = "msa_pairformer"
    needs_msa = True
    runner_script = _HERE / "_runners" / "msa_pairformer_runner.py"
