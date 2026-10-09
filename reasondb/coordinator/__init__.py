"""Distributed job-coordination system for reasondb benchmark sweeps.

A coordinator (``scripts/run_coordinator.py``) enumerates a sweep's jobs once at
startup and persists them in a SQLite queue (:mod:`reasondb.coordinator.db`); workers
(``scripts/run_worker.py``) started on other networked machines register
with a capability flag, pull matching jobs, execute them, and forward telemetry back.
See ``reasondb.coordinator.producers`` for the per-script job producers.
"""
