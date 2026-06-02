"""MSA-generation backends, one module per `kind`, uniform interface.

Each backend exposes ``prepare(datasets)``, ``submit(datasets)``, and
``ingest(datasets, out_dir=None)``. ``generate.py`` dispatches by kind via the
``BACKENDS`` registry — adding a kind is a new module + one registry entry, not
another branch in a shared flow.
"""
from ecstasy.msa.backends import boltz_csv
from ecstasy.msa.backends import complex as _complex          # avoid shadowing the builtin
from ecstasy.msa.backends import complex_api as _complex_api

# kind -> backend. Two models, two distinct local pipelines — never conflate them
# (see msa/README.md):
BACKENDS = {
    "boltz_csv": boltz_csv,       # Boltz-2: LOCAL colabfold_search -> paired+unpaired per-chain CSVs
    "complex": _complex,          # MSA-Pairformer: LOCAL colabfold-local -> stitched complex a3m  (DEFAULT)
    "complex_api": _complex_api,  # MSA-Pairformer: ColabFold API fallback (NOT used for the eval data)
}
