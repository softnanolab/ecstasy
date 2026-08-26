"""Named, reusable metrics.

Metrics used to be reachable only from the one place they happened to be written. The
sharpest example: tolerant inter-chain P@K — GT dilated by a Chebyshev radius so a
near-miss counts — existed only inside
``scripts/mentos-perf-benchmarking/plot_pak_vs_flops.py``. It worked, it was used to make
published figures, and ``ecstasy score`` could not call it. Anything else that wanted
tolerance had to copy it.

So metrics are registered under a name and requested by name, from scoring, from
plotting, from a manifest, or from the CLI:

    from ecstasy.metrics import registry
    registry.compute(["P@K", "P@K(tol=2)"], ev)

Registering a metric is the only way to make it available; there is no second path that
scoring reaches and plotting does not. A metric declares whether higher is better, so
downstream code can rank without a hardcoded table of exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

#: A metric's input kind. "contact" metrics take a `ContactEval`; "structure" metrics take
#: a `StructureEval`. The kind is what lets scoring ask "which metrics can I even run
#: against what this model produced" instead of guessing from the model name.
KINDS = ("contact", "structure")


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str
    fn: Callable[..., float]
    description: str
    higher_is_better: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, ev) -> float:
        return self.fn(ev, **self.params)


_REGISTRY: dict[str, Metric] = {}


def register(name: str, kind: str, fn: Callable[..., float], description: str,
             higher_is_better: bool = True, **params: Any) -> Metric:
    """Register a metric under `name`. Re-registering the same name is an error.

    Silently overwriting would let two different definitions of "P@K" coexist across a
    codebase, which is the failure this registry exists to prevent.
    """
    if kind not in KINDS:
        raise ValueError(f"metric {name!r} has unknown kind {kind!r}; expected one of {KINDS}")
    if name in _REGISTRY:
        raise ValueError(
            f"metric {name!r} is already registered ({_REGISTRY[name].description!r}). "
            f"Pick a distinct name — two definitions under one name is exactly the "
            f"ambiguity this registry prevents.")
    m = Metric(name=name, kind=kind, fn=fn, description=description,
               higher_is_better=higher_is_better, params=dict(params))
    _REGISTRY[name] = m
    return m


def get(name: str) -> Metric:
    if name not in _REGISTRY:
        raise KeyError(f"unknown metric {name!r}; registered: {names()}")
    return _REGISTRY[name]


def names(kind: str | None = None) -> list[str]:
    return sorted(n for n, m in _REGISTRY.items() if kind is None or m.kind == kind)


def describe(kind: str | None = None) -> list[dict]:
    """Machine-readable catalogue — what `ecstasy metrics` prints and an agent reads."""
    return [
        {"name": m.name, "kind": m.kind, "description": m.description,
         "higher_is_better": m.higher_is_better, "params": dict(m.params)}
        for m in sorted(_REGISTRY.values(), key=lambda m: (m.kind, m.name))
        if kind is None or m.kind == kind
    ]


def compute(metric_names: Iterable[str], ev, skip_errors: bool = False) -> dict[str, float]:
    """Run each named metric against `ev`.

    With ``skip_errors`` a metric that raises is omitted rather than killing the whole
    scoring pass — one broken metric should not cost a run every other number it earned.
    """
    out: dict[str, float] = {}
    for name in metric_names:
        m = get(name)
        if m.kind != ev.KIND:
            raise TypeError(
                f"metric {name!r} is a {m.kind!r} metric but was given a {ev.KIND!r} input")
        try:
            out[name] = float(m(ev))
        except Exception:  # noqa: BLE001
            if not skip_errors:
                raise
    return out


def _reset_for_tests() -> None:
    """Drop all registrations. Tests only — never call this from library code."""
    _REGISTRY.clear()
