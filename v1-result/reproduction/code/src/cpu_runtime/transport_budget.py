"""Shared defaults for the Edge/OpenCLI durable-job transport, in seconds.

These are per-request safety deadlines, never a total TTT training deadline.
The installed OpenCLI eval path has no per-command timeout option. Its locally
inspected defaults are extension 115 < daemon 120 < daemon HTTP 130.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TransportBudget:
    fetch: float = 20
    extension: float = 115
    daemon: float = 120
    daemon_http: float = 130
    cli: float = 150
    lock: float = 60
    worker: float = 600
    recovery: float = 60

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in self.__dict__.values()):
            raise ValueError("transport deadlines must be finite and positive")
        if not self.fetch < self.extension < self.daemon < self.daemon_http < self.cli:
            raise ValueError("transport deadlines must expire from inner to outer")

    @property
    def transport_call(self):
        return self.lock + self.cli

    def bridge(self, worker=None):
        return (self.worker if worker is None else worker) + 2 * self.transport_call + self.recovery

    @property
    def client(self):
        return self.bridge() + 60

    @property
    def barrier(self):
        return self.client + 60


DEFAULT_BUDGET = TransportBudget()
