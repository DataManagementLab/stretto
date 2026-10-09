"""Shared parsing for REASONDB_PRECOMPUTE_SKIP_MODALITIES.

Read by both `Executor._precompute_pipeline` (skips `run_outside_db` for a
skipped modality's operators) and `PlanConfigurator.prepare` (skips KV cache
materialization for them), so a `--precompute` pass can run with only some of
the KV cache servers a mixed-modality benchmark needs up at once (e.g.
ecommerce needs both text *and* image servers; this lets one pass cover text
with only the text server up, and a separate pass cover image with only the
image server up).

Referenced via `precompute_modalities.PRECOMPUTE_SKIP_MODALITIES` (module
attribute access, not `from ... import PRECOMPUTE_SKIP_MODALITIES`) in both
consumers, so tests can monkeypatch this one module attribute and have it
take effect in both places.
"""

import os


def parse_precompute_skip_modalities(raw: str) -> frozenset:
    """Parse the env var value into a lowercased modality set.

    Comma-separated (e.g. "image,audio"); entries are stripped and lowercased,
    blanks (including an unset/empty variable) dropped.
    """
    return frozenset(m.strip().lower() for m in raw.split(",") if m.strip())


PRECOMPUTE_SKIP_MODALITIES = parse_precompute_skip_modalities(
    os.environ.get("REASONDB_PRECOMPUTE_SKIP_MODALITIES", "")
)
