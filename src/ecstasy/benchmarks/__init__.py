from ecstasy.benchmarks.base import Benchmark, Entry, BENCHMARKS, register_benchmark, load_benchmark
from ecstasy.benchmarks import mint_seqid30  # noqa: F401  (registers MintSeqid30Bench)

__all__ = ["Benchmark", "Entry", "BENCHMARKS", "register_benchmark", "load_benchmark"]
