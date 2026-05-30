"""MSA-generation backends, one module per `kind`, uniform interface.

Each backend exposes ``prepare(datasets)``, ``submit(datasets)``, and
``ingest(datasets, out_dir=None)``. ``generate.py`` dispatches by kind via the
``BACKENDS`` registry — adding a kind is a new module + one registry entry, not
another branch in a shared flow.
"""
from ecstasy.msa.backends import boltz_csv
from ecstasy.msa.backends import complex as _complex  # avoid shadowing the builtin

BACKENDS = {
    "boltz_csv": boltz_csv,   # Boltz-2: paired+unpaired per-chain CSVs (local colabfold_search)
    "complex": _complex,      # MSA Pairformer: SI-faithful paired a3m (ColabFold API)
}
