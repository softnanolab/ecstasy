from ecstasy.benchmarks.base import Benchmark, Entry, BENCHMARKS, register_benchmark, load_benchmark
from ecstasy.benchmarks import mentos_seqid30  # noqa: F401  (registers MentosSeqid30Bench)

__all__ = ["Benchmark", "Entry", "BENCHMARKS", "register_benchmark", "load_benchmark"]
from ecstasy.benchmarks import ecstasy_v1  # noqa: F401  (registers EcstasyV1Bench)
